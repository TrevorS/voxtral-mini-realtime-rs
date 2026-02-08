# Voxtral Mini 4B Realtime (Rust)

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model_on_HuggingFace-yellow)](https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf)
[![Live Demo](https://img.shields.io/badge/%F0%9F%94%8A-Live_Demo-blue)](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime)

Streaming speech recognition running natively and in the browser. A pure Rust implementation of [Mistral's Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) model using the [Burn](https://burn.dev) ML framework.

The Q4 GGUF quantized path (2.5 GB) runs entirely client-side in a browser tab via WASM + WebGPU. [Try it live.](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime)

## Quick Start

### Native CLI

```bash
# Download model weights (~9 GB)
uv run --with huggingface_hub \
  hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral

# Transcribe an audio file (f32 SafeTensors path)
cargo run --release --features "wgpu,cli,hub" --bin voxtral-transcribe -- \
  --audio audio.wav --model models/voxtral

# Or use the Q4 quantized path (~2.5 GB)
cargo run --release --features "wgpu,cli,hub" --bin voxtral-transcribe -- \
  --audio audio.wav --gguf models/voxtral-q4.gguf --tokenizer models/voxtral/tekken.json
```

### Browser Demo

```bash
# Build WASM package
wasm-pack build --target web --no-default-features --features wasm

# Generate self-signed cert (WebGPU requires secure context)
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout /tmp/voxtral-key.pem -out /tmp/voxtral-cert.pem \
  -days 7 -nodes -subj "/CN=localhost"

# Start dev server
bun serve.mjs
```

Open `https://localhost:8443`, accept the certificate, and click **Load from Server** to download the model shards. Record from your microphone or upload a WAV file to transcribe.

[Hosted demo on HuggingFace Spaces](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime) if you want to skip local setup.

## Architecture

```
Audio (16kHz mono)
  -> Mel spectrogram [B, 128, T]
    -> Causal encoder (32 layers, 1280 dim, sliding window 750)
      -> Conv 4x downsample -> Reshape [B, T/16, 5120]
        -> Adapter [B, T/16, 3072]
          -> Autoregressive decoder (26 layers, 3072 dim, GQA 32Q/8KV)
            -> Token IDs -> Text
```

### Two Inference Paths

| | F32 (native) | Q4 GGUF (native + browser) |
|---|---|---|
| Weights | SafeTensors (~9 GB) | GGUF Q4_0 (~2.5 GB) |
| Linear ops | Burn tensor matmul | Custom WGSL shader (fused dequant + matmul) |
| Embeddings | f32 tensor (1.5 GiB) | Q4 on GPU (216 MB) + CPU bytes for lookups |
| Browser | No | Yes (WASM + WebGPU) |

### Q4 Padding Workaround

The upstream mistral-common library left-pads audio with 32 silence tokens (at 12.5 Hz). After the mel/conv/reshape pipeline, this covers only 16 of the 38 decoder prefix positions with silence — the remaining 22 contain actual audio. The f32 model handles this fine, but Q4_0 quantization makes the decoder sensitive to speech content in the prefix: audio that starts immediately with speech (mic recordings, clips with no leading silence) produces all-pad tokens instead of text.

The left padding is increased to 76 tokens, which maps to exactly 38 decoder tokens of silence and covers the full streaming prefix. See [`src/audio/pad.rs`](src/audio/pad.rs) for details.

### WASM Constraints Solved

Running a 4B model in a browser tab required solving five hard constraints:

1. **2 GB allocation limit** — `ShardedCursor` reads across multiple `Vec<u8>` buffers
2. **4 GB address space** — Two-phase loading: parse weights, drop reader, then finalize
3. **1.5 GiB embedding table** — Q4 embeddings on GPU + CPU-side row lookups
4. **No sync GPU readback** — All tensor reads use `into_data_async().await`
5. **256 workgroup invocation limit** — Patched cubecl-wgpu to cap reduce kernel workgroups

## Building

```bash
# Native (default features: wgpu + native-tokenizer)
cargo build --release

# With all features
cargo build --release --features "wgpu,cli,hub"

# WASM
wasm-pack build --target web --no-default-features --features wasm
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `wgpu` (default) | GPU backend via Burn/CubeCL (WebGPU, Vulkan, Metal) |
| `native-tokenizer` (default) | Tekken tokenizer (C deps, not WASM-compatible) |
| `wasm` | Browser support: wasm-bindgen, WebGPU device init, JS bindings |
| `cli` | CLI binary with clap + indicatif |
| `hub` | HuggingFace Hub model downloads |

## Testing

```bash
# Unit + integration tests (requires GPU for full suite)
cargo test --features "wgpu,cli,hub"

# Lint
cargo clippy --features "wgpu,cli,hub" -- -D warnings
cargo clippy --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

# E2E browser test (requires Playwright + model shards)
bunx playwright test tests/e2e_browser.spec.ts
```

GPU-dependent tests (model layer shapes, Q4 matmul, WGSL shader correctness) are skipped in CI since GitHub Actions runners lack a GPU adapter. These tests run locally on any machine with Vulkan, Metal, or WebGPU support.

## Model Preparation

### Q4 GGUF Sharding (for browser)

The GGUF file must be split into shards of 512 MB or less to stay under the browser's `ArrayBuffer` limit:

```bash
split -b 512m models/voxtral-q4.gguf models/voxtral-q4-shards/shard-
```

The dev server and E2E test discover shards automatically from `models/voxtral-q4-shards/`.

## Benchmarks

Coming soon: accuracy (WER) and inference speed benchmarks across native and browser targets.

## Project Structure

```
src/
  audio/          # Mel spectrogram, chunking, resampling, padding
  models/         # F32 model: encoder, decoder, adapter, attention, RoPE, KV cache
  gguf/           # Q4 GGUF: reader, loader, model, tensor, WGSL shader, tests
  web/            # WASM bindings: VoxtralQ4, initWgpuDevice, async decode loop
  tokenizer/      # Tekken tokenizer wrapper (native only)
  bin/transcribe  # CLI binary

web/              # Browser demo: index.html, worker.js, voxtral-client.js
tests/            # Integration tests + Playwright E2E spec
scripts/          # Dev scripts: reference implementations, weight inspection, E2E helpers
patches/          # cubecl-wgpu workgroup size fix for WebGPU
```

## License

Apache-2.0
