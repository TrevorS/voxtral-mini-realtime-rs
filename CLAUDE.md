# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voxtral Mini 4B Realtime — streaming ASR in Rust using Burn. Runs natively (Vulkan/Metal) and in the browser (WASM + WebGPU). The Q4 GGUF path enables a 4B-parameter model to run entirely client-side in a browser tab.

## Build & Development Commands

```bash
# Native build
cargo build --features "wgpu,cli,hub"
cargo build --release --features "wgpu,cli,hub"

# WASM build (produces pkg/ directory)
wasm-pack build --target web --no-default-features --features wasm

# Lint both targets
cargo clippy --features "wgpu,cli,hub" -- -D warnings
cargo clippy --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

# Format
cargo fmt
cargo fmt -- --check

# Tests (110 tests: 107 unit + 3 integration)
cargo test --features "wgpu,cli,hub"
cargo test test_q4_roundtrip_small           # single test
cargo test gguf::tests                        # module tests

# E2E browser test (requires model shards + Playwright)
bunx playwright test tests/e2e_browser.spec.ts

# CLI inference
cargo run --features "wgpu,cli,hub" --bin voxtral-transcribe -- \
  --audio test_data/mary_had_lamb.wav \
  --gguf models/voxtral-q4.gguf \
  --tokenizer models/voxtral/tekken.json

# Dev HTTPS server for browser testing
bun serve.mjs
```

## Feature Flags

```toml
default = ["wgpu", "native-tokenizer"]
wgpu              # Burn GPU backend (WebGPU/Vulkan/Metal) — required for gguf module
native-tokenizer  # tokenizers crate (C deps, disabled in WASM)
wasm              # WASM bindings: enables wgpu + wasm-bindgen + web-sys + dep:wgpu
cli               # clap + indicatif for voxtral-transcribe binary
hub               # hf-hub for model downloads
```

The `wasm` feature enables both `wgpu` (for burn's WebGPU backend) and `dep:wgpu` (direct wgpu crate access for custom device init). The `gguf` module is gated behind `wgpu`. The `web` module is gated behind `wasm`.

## Model Weights

```bash
# F32 SafeTensors (~9 GB)
uv run --with huggingface_hub hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral

# Q4 GGUF (~2.5 GB)
uv run --with huggingface_hub hf download TrevorJS/voxtral-mini-realtime-gguf voxtral-q4.gguf --local-dir models

# Sharded for WASM (5 × ≤512 MB each)
# Expected: models/voxtral-q4-shards/shard-{aa..ae}
```

## Architecture

### Data Flow

```
Audio (16kHz) → Peak normalize (0.95) → Mel [B, 128, T] → Encoder [B, T/4, 1280]
  → Reshape [B, T/16, 5120] → Adapter [B, T/16, 3072] → Decoder (autoregressive)
  → Token IDs → Text
```

### Audio Peak Normalization

Both the CLI (`src/bin/transcribe.rs`) and WASM (`src/web/bindings.rs`) paths call `AudioBuffer::peak_normalize(0.95)` before mel computation. This is critical for Q4 inference: quiet audio (peak < 0.02) produces mel spectrograms indistinguishable from silence after log normalization with `log_mel_max=1.5`, causing the decoder to emit only pad tokens. The f32 model tolerates this due to higher precision throughout the pipeline, but Q4 cannot resolve the subtle mel features. Normalization to 0.95 lifts quiet audio ~4.75 in log-space while barely affecting normal audio (~0.05 shift).

### Two Inference Paths

**F32 path** (`src/models/`): SafeTensors weights, full precision, native only.

**Q4 GGUF path** (`src/gguf/`): Q4_0 quantized weights, fused GPU dequant+matmul via custom WGSL shader, works in browser.

| Component | F32 | Q4 GGUF |
|-----------|-----|---------|
| Weight source | SafeTensors | GGUF v2/v3 |
| Linear layers | Burn tensor matmul | Custom `q4_matmul` (shader.wgsl) |
| Token embeddings | f32 tensor (1.5 GiB) | Q4 on GPU + CPU bytes for lookups |
| Memory (model) | ~16 GB | ~2.5 GB |

### GGUF Pipeline

```
reader.rs (GGUF parser + ShardedCursor)
  → loader.rs (Q4ModelLoader, two-phase for WASM)
    → model.rs (Q4 model components, TokEmbedStore)
      → tensor.rs + linear.rs + op.rs + shader.wgsl (GPU Q4 matmul)
        → bindings.rs (WASM JS API)
```

### Key Constants

| Parameter | Value |
|-----------|-------|
| Encoder: 32 layers, 1280 dim, MHA 32 heads, sliding window 750 |
| Decoder: 26 layers, 3072 dim, GQA 32Q/8KV, sliding window 8192 |
| Adapter: 5120 → 3072 (two linear projections) |
| Mel: 128 bins, hop 160, conv 4× downsample → 12.5 Hz frame rate |
| Vocab: 131,072 (Tekken tokenizer) |
| Streaming: 38-token prefix (BOS + 37 pad), lookahead t=6 (480ms) |
| Left pad: 76 tokens at 12.5 Hz (Q4 workaround; upstream default is 32) |

## WASM Constraints & Workarounds

The browser imposes hard limits that drive most of the complexity in `src/gguf/` and `src/web/`:

1. **2 GB per-allocation limit** (wasm32 `isize::MAX`): `ShardedCursor` in `reader.rs` implements `Read + Seek` over `Vec<Vec<u8>>`, keeping each shard as a separate allocation.

2. **4 GB total address space**: Two-phase loading (`Q4ModelParts` in `loader.rs`) loads Q4 weights first, drops the GGUF reader (~2.5 GB), then finalizes the model.

3. **1.5 GiB embedding table**: `TokEmbedStore::Q4` in `model.rs` keeps embeddings as Q4 on GPU (~216 MB) plus raw Q4 bytes on CPU for `embed_tokens_from_ids()` lookups. Avoids a 1.5 GiB f32 GPU buffer.

4. **No synchronous GPU readback**: `into_data_async().await` throughout `bindings.rs` instead of `into_scalar()` which calls `block_on()` and panics in WASM.

5. **WebGPU 256 workgroup invocation limit**: Patched cubecl-wgpu (`patches/cubecl-wgpu-0.9.0/src/runtime.rs`) to cap `max_subgroup_size` so reduce kernels stay within `max_compute_invocations_per_workgroup`. Without this, cubek-reduce creates 128×8=1024 workgroups which Chrome rejects.

## Patches

`patches/cubecl-wgpu-0.9.0/` is a vendored copy of cubecl-wgpu 0.9.0 with one fix in `src/runtime.rs`: when WebGPU reports `min_subgroup_size=0, max_subgroup_size=0`, the workaround now caps `max_subgroup_size` at `max_compute_invocations_per_workgroup / 8` instead of hardcoding 128. Applied via `[patch.crates-io]` in Cargo.toml. The patch is a no-op when the adapter reports real subgroup sizes (native GPU, Chrome 134+ with subgroups).

## Web App

`serve.mjs` — HTTPS dev server (self-signed cert for WebGPU secure context). Serves shards from `models/voxtral-q4-shards/` via `/api/shards` endpoint.

`web/worker.js` — ES module WebWorker. Handles `init` (WASM + WebGPU device), `loadFromServer` (sequential shard download → `appendModelShard` → `loadModelFromShards`), and `transcribe` (async inference).

`web/voxtral-client.js` — Main-thread client wrapping worker communication, Web Audio API for mic input, and audio resampling to 16kHz.

## E2E Testing

`tests/e2e_browser.spec.ts` runs in headless Chromium via Playwright. Config in `playwright.config.ts` sets NVIDIA Vulkan ICD, WebGPU flags, and 10-minute timeout. The test spins up an HTTP server, loads shards, transcribes `test_data/mary_had_lamb.wav`, and asserts the output contains "mary" and "lamb".
