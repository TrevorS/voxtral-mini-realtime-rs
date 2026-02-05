#!/usr/bin/env -S uv run --with playwright --python 3.11
# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright"]
# ///
"""
End-to-end transcription test in headless Chrome with WebGPU.

Uses VoxtralPhased with streamed shards to fit within wasm32's 4GB limit.
Loads encoder → encodes audio → frees encoder → loads decoder layer-by-layer
→ transcribes → checks output.

Prerequisites:
    uvx --with playwright playwright install chromium
    wasm-pack build --target web --no-default-features --features wasm

Usage:
    ./scripts/test_webgpu_e2e.py
"""

import http.server
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(port: int, directory: str) -> socketserver.TCPServer:
    os.chdir(directory)
    server = ThreadedHTTPServer(("", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    from playwright.sync_api import sync_playwright

    project_root = Path(__file__).parent.parent
    shards_dir = project_root / "models" / "shards-safetensors"

    # Check prerequisites
    checks = [
        (project_root / "pkg" / "voxtral_mini_realtime.js",
         "WASM pkg (run: wasm-pack build --target web --no-default-features --features wasm)"),
        (shards_dir / "manifest.json", "Streamed shards"),
        (shards_dir / "encoder.safetensors.gz", "Encoder shard"),
        (project_root / "models" / "voxtral" / "tekken.json", "Tokenizer"),
        (project_root / "test_data" / "mary_had_lamb.wav", "Test audio"),
    ]
    for path, label in checks:
        if not path.exists():
            print(f"Missing: {path}\n  {label}")
            sys.exit(1)

    port = get_free_port()
    server = start_server(port, str(project_root))
    base = f"http://localhost:{port}"
    print(f"Server: {base}")
    print(f"Shards: {shards_dir.relative_to(project_root)}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--headless=new",
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan",
                    "--disable-vulkan-surface",
                    "--use-angle=vulkan",
                ],
            )
            context = browser.new_context()
            page = context.new_page()

            def on_console(msg):
                text = msg.text
                if any(skip in text for skip in ["DevTools", "Autofill"]):
                    return
                print(f"  [browser] {text}")

            page.on("console", on_console)
            page.goto(f"{base}/web/index.html", timeout=60000)

            # 10 min timeout for full pipeline
            page.set_default_timeout(600000)

            print("\nStarting WebGPU phased transcription test...")
            result = page.evaluate("""async ({ base }) => {
                const log = (msg) => console.log(msg);

                try {
                    // Helper: fetch + gunzip a .safetensors.gz shard → Uint8Array
                    async function fetchShard(url) {
                        const resp = await fetch(url);
                        if (!resp.ok) throw new Error('Failed to fetch ' + url + ': ' + resp.status);
                        const ds = new DecompressionStream('gzip');
                        const decompressed = resp.body.pipeThrough(ds);
                        const reader = decompressed.getReader();
                        const chunks = [];
                        let total = 0;
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            chunks.push(value);
                            total += value.length;
                        }
                        const result = new Uint8Array(total);
                        let offset = 0;
                        for (const chunk of chunks) {
                            result.set(chunk, offset);
                            offset += chunk.length;
                        }
                        return result;
                    }

                    // Helper: decode WAV to Float32Array
                    function decodeWav(arrayBuf) {
                        const dv = new DataView(arrayBuf);
                        const numChannels = dv.getUint16(22, true);
                        const sampleRate = dv.getUint32(24, true);
                        const bitsPerSample = dv.getUint16(34, true);
                        let dataOffset = 12;
                        while (dataOffset < arrayBuf.byteLength - 8) {
                            const id = String.fromCharCode(
                                dv.getUint8(dataOffset), dv.getUint8(dataOffset+1),
                                dv.getUint8(dataOffset+2), dv.getUint8(dataOffset+3));
                            const size = dv.getUint32(dataOffset + 4, true);
                            if (id === 'data') { dataOffset += 8; break; }
                            dataOffset += 8 + size;
                        }
                        const numSamples = (arrayBuf.byteLength - dataOffset) / (bitsPerSample / 8) / numChannels;
                        const audio = new Float32Array(numSamples);
                        if (bitsPerSample === 16) {
                            for (let i = 0; i < numSamples; i++) {
                                audio[i] = dv.getInt16(dataOffset + i * 2 * numChannels, true) / 32768.0;
                            }
                        } else if (bitsPerSample === 32) {
                            for (let i = 0; i < numSamples; i++) {
                                audio[i] = dv.getFloat32(dataOffset + i * 4 * numChannels, true);
                            }
                        }
                        return { audio, sampleRate, numSamples };
                    }

                    // 1. Init WASM
                    log('Loading WASM module...');
                    const mod = await import(base + '/pkg/voxtral_mini_realtime.js');
                    await mod.default();
                    log('WASM initialized');

                    // Initialize WebGPU device asynchronously (required on WASM)
                    log('Initializing WebGPU device...');
                    await mod.initWgpuDevice();
                    log('WebGPU device initialized');

                    // Check WebGPU
                    if (!navigator.gpu) {
                        return { success: false, error: 'WebGPU not available' };
                    }
                    const adapter = await navigator.gpu.requestAdapter();
                    if (!adapter) {
                        return { success: false, error: 'No WebGPU adapter' };
                    }
                    log('WebGPU adapter found');

                    // 2. Create phased model
                    const voxtral = new mod.VoxtralPhased();

                    // 3. Load tokenizer
                    log('Loading tokenizer...');
                    const tokResp = await fetch(base + '/models/voxtral/tekken.json');
                    const tokJson = await tokResp.text();
                    voxtral.loadTokenizer(tokJson);
                    log('Tokenizer loaded');

                    // 4. Load encoder + adapter
                    const shardsBase = base + '/models/shards-safetensors';

                    log('Loading encoder shard (~712 MB compressed)...');
                    const t0 = performance.now();
                    const encoderBytes = await fetchShard(shardsBase + '/encoder.safetensors.gz');
                    log('Encoder fetched: ' + (encoderBytes.length / 1e6).toFixed(0) + ' MB decompressed');
                    voxtral.loadEncoderShard(encoderBytes);
                    log('Encoder loaded into model');

                    log('Loading adapter shard...');
                    const adapterBytes = await fetchShard(shardsBase + '/adapter.safetensors.gz');
                    voxtral.loadAdapterShard(adapterBytes);
                    log('Adapter loaded');

                    // 5. Load and encode audio
                    log('Loading test audio...');
                    const audioResp = await fetch(base + '/test_data/mary_had_lamb.wav');
                    const audioBuf = await audioResp.arrayBuffer();
                    const { audio, sampleRate, numSamples } = decodeWav(audioBuf);
                    log('Audio: ' + sampleRate + ' Hz, ' + numSamples + ' samples, ' + (numSamples / sampleRate).toFixed(1) + 's');

                    log('Encoding audio...');
                    await voxtral.encodeAudio(audio);
                    log('Audio encoded');

                    // 6. Free encoder to reclaim memory
                    voxtral.freeEncoder();
                    log('Encoder freed');

                    // 7. Load decoder layer-by-layer
                    log('Loading decoder embeddings...');
                    const embBytes = await fetchShard(shardsBase + '/decoder_embeddings.safetensors.gz');
                    log('Embeddings fetched: ' + (embBytes.length / 1e6).toFixed(0) + ' MB');
                    voxtral.loadDecoderEmbeddings(embBytes);
                    log('Embeddings loaded');

                    for (let i = 0; i < 26; i++) {
                        const padded = String(i).padStart(2, '0');
                        const layerBytes = await fetchShard(shardsBase + '/decoder_layer_' + padded + '.safetensors.gz');
                        voxtral.loadDecoderLayer(i, layerBytes);
                        if (i % 5 === 0 || i === 25) {
                            log('Decoder layer ' + i + '/25 loaded');
                        }
                    }

                    log('Loading decoder norm...');
                    const normBytes = await fetchShard(shardsBase + '/decoder_norm.safetensors.gz');
                    voxtral.loadDecoderNorm(normBytes);
                    log('Decoder assembled');

                    const loadSec = (performance.now() - t0) / 1000;

                    // 8. Transcribe
                    log('Transcribing...');
                    const t1 = performance.now();
                    const text = await voxtral.transcribe();
                    const transcribeSec = (performance.now() - t1) / 1000;
                    log('Result: ' + text);

                    // 9. Cleanup
                    voxtral.freeDecoder();

                    return {
                        success: true,
                        text,
                        loadTimeSec: loadSec,
                        transcribeTimeSec: transcribeSec,
                        sampleRate,
                        audioSamples: numSamples,
                    };
                } catch (e) {
                    return { success: false, error: e.message || String(e), stack: e.stack };
                }
            }""",
                {"base": base},
            )

            browser.close()

    finally:
        server.shutdown()

    # Report results
    print("\n" + "=" * 60)
    if result["success"]:
        print("PASS")
        print(f"  Transcription: {result['text']}")
        print(f"  Model load:    {result['loadTimeSec']:.1f}s")
        print(f"  Transcribe:    {result['transcribeTimeSec']:.1f}s")
        print(f"  Audio:         {result['audioSamples']} samples @ {result['sampleRate']} Hz")

        text_lower = result["text"].lower()
        if "mary" in text_lower and "lamb" in text_lower:
            print("\n  Content check: PASS (contains 'mary' and 'lamb')")
        else:
            print(f"\n  Content check: WARN - expected 'mary' and 'lamb' in output")
    else:
        print("FAIL")
        print(f"  Error: {result['error']}")
        if result.get("stack"):
            print(f"  Stack: {result['stack']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
