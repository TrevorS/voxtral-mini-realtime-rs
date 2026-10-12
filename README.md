# Voxtral Mini 4B Realtime (Rust)

[![HuggingFace ASR](https://img.shields.io/badge/%F0%9F%A4%97-ASR_Model-yellow)](https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf)
[![HuggingFace TTS](https://img.shields.io/badge/%F0%9F%A4%97-TTS_Model-yellow)](https://huggingface.co/TrevorJS/voxtral-tts-q4-gguf)
[![ASR Demo](https://img.shields.io/badge/%F0%9F%8E%99%EF%B8%8F-ASR_Demo-blue)](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime)
[![TTS Demo](https://img.shields.io/badge/%F0%9F%94%8A-TTS_Demo-purple)](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

Streaming speech recognition and text-to-speech running natively and in the browser. A pure Rust implementation of Mistral's [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (ASR) and [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) models using the [Burn](https://burn.dev) ML framework.

## Benchmarks

NVIDIA DGX Spark (GB10, LPDDR5x).

### ASR (Speech Recognition)

16s test audio, 3-run average:

| Path | Encode | Decode | Total | RTF | Tok/s | Memory |
|------|--------|--------|-------|-----|-------|--------|
| **Q4 GGUF native** | 1021 ms | 5578 ms | 6629 ms | **0.416** | **19.4** | 703 MB |
| BF16 native | 887 ms | 23689 ms | 24607 ms | 1.543 | 4.6 | 9.2 GB |
| Q4 GGUF WASM | — | — | ~225 s | ~14.1 | ~0.5 | (browser) |

- **8.49% WER** on FLEURS English (647 utterances), vs. Mistral's reported 4.90% at f32

### TTS (Text-to-Speech)

"The quick brown fox jumps over the lazy dog" (9 tokens), casual_female voice:

| Path | Euler Steps | Gen Time | Audio | RTF | Model Size |
|------|-------------|----------|-------|-----|------------|
| **Q4 GGUF native** | 3 | 3.7s | 3.84s | **0.97** | 2.67 GB |
| Q4 GGUF native | 4 | 5.0s | 4.96s | 1.01 | 2.67 GB |
| BF16 native | 3 | 10.4s | 2.72s | 3.82 | ~8 GB |
| BF16 native | 8 | 20.6s | 2.96s | 6.97 | ~8 GB |
| Q4 GGUF WASM | 8 | 367s | 3.52s | 104 | 2.67 GB |

- **RTF** < 1.0 means faster-than-real-time synthesis
- Q4 at 3 Euler steps achieves **real-time** with perfect Whisper large-v3 transcription
- Optimizations: batched CFG (2× → batch=2), fused QKV+gate/up projections, pre-allocated KV cache
- Q4 model load: 3.9s native, 9.2s WASM (including shard download over localhost)
- 20 preset voices across 9 languages. Use `--euler-steps` to tune speed/quality tradeoff

### Architecture Notes

- Custom WGSL compute shaders with vectorized u32 reads and vec4 dot products
- Dual-path kernel dispatch: shared-memory tiled kernel for single-token decode, naive kernel for multi-row encode/prefill
- Q4 GGUF (2.5 GB ASR, 2.67 GB TTS) runs entirely client-side in a browser tab via WASM + WebGPU

Try the demos: [ASR (speech-to-text)](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime) | [TTS (text-to-speech)](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

## Quick Start

### Native CLI

```bash
# Download ASR model weights (~9 GB BF16 or ~2.5 GB Q4)
uv run --with huggingface_hub \
  hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral
uv run --with huggingface_hub \
  hf download TrevorJS/voxtral-mini-realtime-gguf --local-dir models/

# Transcribe audio (BF16 or Q4)
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  transcribe --audio audio.wav --model models/voxtral
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  transcribe --audio audio.wav --gguf models/voxtral-q4.gguf
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

Hosted demos: [ASR on HuggingFace Spaces](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime) | [TTS on HuggingFace Spaces](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

### Text-to-Speech

```bash
# Download TTS model weights (~8 GB BF16 or ~2.67 GB Q4)
uv run --with huggingface_hub \
  hf download mistralai/Voxtral-4B-TTS-2603 --local-dir models/voxtral-tts
uv run --with huggingface_hub \
  hf download TrevorJS/voxtral-tts-q4-gguf voxtral-tts-q4.gguf --local-dir models

# Synthesize speech (BF16 or Q4)
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --voice casual_female
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --voice casual_female --gguf models/voxtral-tts-q4.gguf

# Real-time with 3 Euler steps
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --gguf models/voxtral-tts-q4.gguf --euler-steps 3

# List available voices
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- speak --list-voices
```

#### Q8_0 (optional, experimental)

The Q8_0 block layout suggests an approximately 4.5 GB GGUF on disk. This
validation does not include a generated full-model Q8 file or listening-quality
evaluation. There is no hosted Q8 GGUF; generate one locally from the BF16
weights to evaluate it in your own environment:

```bash
# Quantize BF16 -> Q8_0 GGUF (approximately 4.5 GB from the block layout)
uv run --with safetensors --with torch --with numpy scripts/quantize_tts_gguf.py \
  models/voxtral-tts/ -o models/voxtral-tts-q8.gguf --quant-type q8_0

# Synthesize with Q8
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --voice casual_female --gguf models/voxtral-tts-q8.gguf
```

20 preset voices across 9 languages. The TTS pipeline runs backbone (Ministral 3B) autoregressive decoding, flow-matching acoustic prediction, and codec synthesis to produce 24 kHz audio.

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

| | BF16 (native) | Q4 GGUF (native + browser) |
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
| `native-tokenizer` (default) | Tekken BPE encoding via tiktoken (WASM-compatible) |
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

GGUF files must be split into shards of 512 MB or less to stay under the browser's `ArrayBuffer` limit:

```bash
# ASR shards
split -b 512m models/voxtral-q4.gguf models/voxtral-q4-shards/shard-

# TTS shards (quantize first, then shard)
uv run --with safetensors --with torch --with numpy --with packaging \
  scripts/quantize_tts_gguf.py models/voxtral-tts/ -o models/voxtral-tts-q4.gguf
split -b 512m models/voxtral-tts-q4.gguf models/voxtral-tts-q4-shards/shard-
```

The dev server discovers shards from `models/voxtral-q4-shards/` (ASR) and `models/voxtral-tts-q4-shards/` (TTS).

## Project Structure

```
src/
  audio/          # Mel spectrogram, chunking, resampling, padding
  models/         # BF16 model: encoder, decoder, adapter, attention, RoPE, KV cache
  gguf/           # Q4 GGUF: reader, loader, model, tensor, WGSL shader, tests
  web/            # WASM bindings: VoxtralQ4, initWgpuDevice, async decode loop
  tts/            # TTS pipeline: backbone, flow matching, codec, voice presets
  tokenizer/      # Tekken tokenizer: decode (ASR) + encode (TTS via tiktoken)
  bin/transcribe  # ASR CLI binary
  bin/speak       # TTS CLI binary

web/              # Browser demo: index.html, worker.js, voxtral-client.js
tests/            # Integration tests + Playwright E2E spec
scripts/          # Dev scripts: reference implementations, weight inspection, E2E helpers
patches/          # cubecl-wgpu workgroup size fix for WebGPU
```

## License

Apache-2.0
