# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voxtral Mini 4B Realtime is a streaming ASR (automatic speech recognition) implementation in Rust using the Burn ML framework. It's a port of Mistral's Voxtral Mini 4B Realtime model with WASM/browser support as a goal.

## Build & Development Commands

```bash
# Build
cargo build
cargo build --release

# Run tests (79 tests, ~3 min with model)
cargo test

# Run specific test
cargo test test_name
cargo test module::tests::test_name

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
cargo fmt -- --check

# Run inference test (requires model weights)
cargo run --bin test_inference -- audio.wav
```

## Model Weights

Download from HuggingFace (requires ~9GB):
```bash
# Using hf CLI or scripts/download_model.py
hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral
```

Expected path: `models/voxtral/consolidated.safetensors`

## Architecture

### Data Flow
```
mel [B, 128, T] → encoder [B, T/4, 1280] → reshape [B, T/16, 5120]
  → adapter [B, T/16, 3072] → decoder [B, T/16, vocab_size]
```

### Key Components

**Audio Processing** (`src/audio/`):
- `mel.rs` - Mel spectrogram extraction (128 bins, 16kHz, hop=160)
- `chunk.rs` - Audio chunking for `max_source_positions` limit (default 1500 frames ≈ 15 sec)
- `resample.rs` - Resampling to 16kHz

**Model** (`src/models/`):
- `encoder.rs` - Causal Whisper-style encoder (32 layers, 1280 dim, sliding window 750)
- `decoder.rs` - Ministral-3B based LLM (26 layers, 3072 dim, GQA 32Q/8KV)
- `adapter.rs` - Projects encoder output to LLM dimension
- `voxtral.rs` - Complete model combining all components
- `loader.rs` - Weight loading from SafeTensors
- `weights.rs` - `OwnedSafeTensors` wrapper (Arc-based, no memory leak)

**Layers** (`src/models/layers/`):
- `attention.rs` - Multi-head attention with RoPE and sliding window
- `rope.rs` - Rotary position embeddings
- `kv_cache.rs` - KV cache for streaming inference
- `rms_norm.rs` - RMS normalization with ADA conditioning

### Key Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Encoder sliding window | 750 | ~60 sec audio |
| Decoder sliding window | 8192 | tokens |
| max_source_positions | 1500 | mel frames (~15 sec) |
| Frame rate | 12.5 Hz | 1 token = 80ms audio |
| Vocab size | 131,072 | |

### Memory Considerations

- Model weights: ~8GB (BF16) → ~16GB (F32 after conversion)
- `OwnedSafeTensors` uses `Arc<Vec<u8>>` to avoid leaking memory
- Tests use `OnceLock` shared loaders to prevent OOM from parallel test execution

## Feature Flags

```toml
wgpu    # Default GPU backend (WebGPU/Vulkan/Metal)
cuda    # NVIDIA CUDA backend
cpu     # CPU backend (ndarray)
cli     # CLI tools (clap, indicatif)
hub     # HuggingFace Hub integration
```
