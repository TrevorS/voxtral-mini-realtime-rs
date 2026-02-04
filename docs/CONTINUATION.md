# Project Status

## Overview

Pure Rust implementation of Voxtral Mini 4B Realtime using the Burn ML framework. Target: streaming ASR with WASM/browser support.

## Model Summary (Verified)

| Component | Layers | Dim | Heads | Window | Special |
|-----------|--------|-----|-------|--------|---------|
| Audio Encoder | 32 | 1280 | 32 MHA | 750 | Causal, ADA RMSNorm, biases |
| Language Model | 26 | 3072 | 32Q/8KV | 8192 | GQA 4:1, no biases, tied embed |
| Adapter | 2 | 5120→3072 | - | - | GELU activation |

**Timing:** 1 text token = 80ms audio = 1280 samples @ 16kHz (12.5 Hz frame rate)

## Implementation Status

### Phase 1: Foundation ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Project setup | ✅ Complete | Cargo.toml with Burn, features for CPU/WGPU/CUDA |
| Config parsing | ✅ Complete | Parses nested `params.json`, verified against model |
| Mel spectrogram | ✅ Complete | Pure Rust FFT, 16kHz/128 bins/hop=160/win=400 |
| Audio I/O | ✅ Complete | WAV load/save with format conversion |
| Resampling | ✅ Complete | High-quality FFT resampling to 16kHz |
| Tokenizer wrapper | ✅ Complete | Tekken tokenizer integration |
| Model download | ✅ Complete | `scripts/download_model.py` |

**Test counts:** 67 unit tests passing, clippy clean
**Model downloaded:** 8.86 GB weights + config + tokenizer

### Development Tools ✅

| Script | Purpose |
|--------|---------|
| `scripts/inspect_weights.py` | Browse SafeTensors: component summaries, shapes, stats |
| `scripts/dump_weight_names.py` | Get full weight paths (not truncated) |
| `scripts/reference_forward.py` | Generate reference outputs for RMSNorm, RoPE, SwiGLU, Conv, Attention |
| `scripts/compare_tensors.py` | Compare Rust outputs vs Python reference with tolerances |

**Test data generated:** `test_data/*.npy` - reference inputs/outputs for all core components
**Rust test utilities:** `src/test_utils.rs` - load_npy, assert_tensors_close

### Phase 2: Audio Encoder ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Conv1d downsampler | ✅ Complete | 128→1280→1280, stride=2, 4x downsample, GELU |
| RMSNorm | ✅ Complete | Validated against reference (max_diff < 1e-3) |
| ADA RMSNorm | ✅ Complete | T-conditional, validated against reference |
| RoPE embeddings | ✅ Complete | theta=1M, interleaved layout, validated |
| SwiGLU MLP | ✅ Complete | gate/up/down, validated against reference |
| Causal self-attention | ✅ Complete | MHA + GQA support, sliding window, validated |
| EncoderLayer | ✅ Complete | Full layer with ADA norm, attn, MLP, residuals |
| 32-layer stack | ✅ Complete | Full AudioEncoder with configurable layers |

### Phase 3: Language Model ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Token embeddings | ✅ Complete | vocab=131072, dim=3072 |
| GQA attention | ✅ Complete | 32Q/8KV heads, head_dim=128 |
| Sliding window | ✅ Complete | 8192 tokens |
| SwiGLU MLP | ✅ Complete | hidden=9216, no biases |
| DecoderLayer | ✅ Complete | Full layer with ADA norm, GQA, MLP |
| 26-layer stack | ✅ Complete | LanguageModel with configurable layers |
| LM head | ✅ Complete | Tied with embeddings |

### Phase 4: Integration 🔄

| Component | Status | Notes |
|-----------|--------|-------|
| AudioLanguageAdapter | ✅ Complete | Linear(5120)→GELU→Linear(3072) |
| VoxtralModel | ✅ Complete | Full end-to-end model combining all components |
| Weight loading infra | ✅ Complete | SafeTensors → Burn tensors, supports F32/F16/BF16 |
| KV cache | ✅ Complete | Concatenation-based, sliding window eviction |
| Layer cache integration | ✅ Complete | forward_with_cache on all layers |
| E2E forward pass | ✅ Complete | test_e2e.rs verified with random weights |
| Full weight loading | ✅ Complete | VoxtralModelLoader loads 8GB SafeTensors into model |
| Streaming loop | 🔲 Pending | Incremental mel + causal forward |

### Phase 5: Browser/WASM 🔲

| Component | Status | Notes |
|-----------|--------|-------|
| WGPU backend | 🔲 Pending | Test with CPU fallback |
| Web Audio API | 🔲 Pending | Microphone input |
| WebWorker | 🔲 Pending | Off-main-thread inference |
| Quantization | 🔲 Pending | INT8/INT4 for model size |

## Project Structure

```
voxtral-mini-realtime-rs/
├── Cargo.toml              # Burn framework, CPU/WGPU/CUDA features
├── scripts/
│   ├── download_model.py   # HuggingFace model download
│   ├── inspect_weights.py  # SafeTensors browser
│   ├── dump_weight_names.py # Full weight paths
│   ├── reference_forward.py # Generate test data
│   └── compare_tensors.py  # Validate Rust vs Python
├── test_data/              # Reference tensors (gitignored)
├── models/
│   └── voxtral/            # Downloaded model (gitignored)
│       ├── consolidated.safetensors  # 8.86 GB
│       ├── params.json               # Architecture config
│       └── tekken.json               # Tokenizer
├── src/
│   ├── lib.rs              # Public API (VoxtralRealtime<B>)
│   ├── main.rs             # Simple placeholder
│   ├── models/
│   │   ├── mod.rs
│   │   └── config.rs       # VoxtralConfig parser (verified)
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── io.rs           # AudioBuffer, WAV I/O
│   │   ├── mel.rs          # MelSpectrogram extractor
│   │   └── resample.rs     # FFT-based resampling
│   ├── tokenizer/
│   │   └── mod.rs          # Tekken tokenizer wrapper
│   └── bin/
│       └── transcribe.rs   # CLI stub
└── docs/
    ├── VOXTRAL_ARCHITECTURE.md  # Model deep dive (verified)
    ├── CONTINUATION.md          # This file
    └── VALIDATION.md            # Validation strategy
```

## Getting Started

### Download Model Weights

```bash
# Download model files (~9GB total) - no auth required
./scripts/download_model.py

# Or specify custom output directory
./scripts/download_model.py --output-dir /path/to/models/voxtral
```

This downloads:
- `consolidated.safetensors` (8.86 GB) - Model weights in BF16
- `params.json` (1.34 KB) - Architecture configuration
- `tekken.json` (14.9 MB) - Tokenizer vocabulary

### Build & Test

```bash
# Build (CPU backend)
cargo build

# Run tests
cargo test

# Run with WGPU backend
cargo build --features wgpu --no-default-features

# Run with CUDA backend
cargo build --features cuda --no-default-features
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `burn` | ML framework with swappable backends |
| `tokenizers` | Tekken tokenizer |
| `hound` | WAV I/O |
| `rubato` | Audio resampling |
| `rustfft` | FFT for mel spectrograms |
| `safetensors` | Weight loading |
| `serde` | Configuration parsing |

## Key Discoveries from params.json

1. **Frame rate is 12.5 Hz** (not 100 Hz) after all downsampling
   - Raw mel: 100 Hz (16000/160)
   - After 4x conv downsample: 25 Hz
   - After 2x reshape: 12.5 Hz = 80ms per token

2. **ADA RMSNorm confirmed**
   - `ada_rms_norm_t_cond: true`
   - `ada_rms_norm_t_cond_dim: 32`
   - Used in audio encoder only

3. **Encoder has biases, LLM does not**
   - Encoder attention: `use_biases: true`
   - LLM attention: `use_biases: false`

4. **Config structure is nested**
   - LLM params at top level
   - Encoder at `multimodal.whisper_model_args.encoder_args`
   - Audio specs at `.encoder_args.audio_encoding_args`

## Key Discoveries from SafeTensors Weights

Weight structure (discovered via `inspect_weights.py`):

| Component | Tensors | Params | Prefix |
|-----------|---------|--------|--------|
| Audio Encoder | 421 | 970M | `mm_streams_embeddings.embedding_module.whisper_encoder.*` |
| LLM Decoder | 286 | 3.03B | `layers.{N}.*` |
| Token Embeddings | 1 | 403M | `mm_streams_embeddings.embedding_module.tok_embeddings.weight` |
| Adapter | 2 | 25M | `mm_streams_embeddings.embedding_module.audio_language_projection.*` |
| Final Norm | 1 | 3K | `norm.weight` |

Key weight patterns:
```
# LLM layer structure (26 layers)
layers.{N}.ada_rms_norm_t_cond.0.weight    # [32, 3072] - ADA RMSNorm in LLM!
layers.{N}.ada_rms_norm_t_cond.2.weight    # [3072, 32]
layers.{N}.attention_norm.weight           # [3072]
layers.{N}.attention.wq/wk/wv/wo.weight   # GQA attention
layers.{N}.ffn_norm.weight                 # [3072]
layers.{N}.feed_forward.w1/w2/w3.weight   # SwiGLU MLP

# Encoder layer structure (32 layers)
mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{N}.*
  .attention.wq/wk/wv.weight + .wq/wv.bias    # MHA with biases
  .attention.wo.weight + .wo.bias
  .feed_forward.w1/w2/w3.weight + .w2.bias

# Conv downsampler
mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.*  # [1280, 128, 3]
mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.*  # [1280, 1280, 3]

# Adapter (Sequential with indices 0, 2 - GELU at index 1)
mm_streams_embeddings.embedding_module.audio_language_projection.0.weight  # [3072, 5120]
mm_streams_embeddings.embedding_module.audio_language_projection.2.weight  # [3072, 3072]
```

**Correction:** ADA RMSNorm is ONLY in decoder (LLM) layers, NOT in encoder layers.
The encoder uses standard RMSNorm. This contradicts an earlier note in this document.

## Next Steps

1. ~~Download model weights~~ ✅
2. ~~Verify config parsing~~ ✅
3. Inspect SafeTensors weight names
4. Implement shared components (RMSNorm, RoPE, SwiGLU)
5. Build audio encoder layer by layer
6. Validate against Python reference
7. Implement LLM decoder
8. Wire up streaming pipeline
9. Test WGPU backend

## Open Questions

1. **ADA RMSNorm t_embed**: How is the temporal conditioning vector computed?
   - No learned t_embed weights in the model - it must be computed or passed in
   - Zeros works for inference but may not be optimal
   - Need to find reference implementation to understand proper values

2. **Tekken tokenizer**: ✅ Resolved
   - Custom loader implemented - HuggingFace `tokenizers` crate doesn't support Tekken format
   - Tekken uses base64-encoded token bytes with some null token_str entries

3. **Weight names**: ✅ Resolved
   - Encoder: `mm_streams_embeddings.embedding_module.whisper_encoder.*`
   - Decoder: `layers.{N}.*`

4. **WASM size**: 8.86GB model needs quantization for browser
   - INT8: ~2.2GB, INT4: ~1.1GB
   - May need dynamic quantization or progressive loading

5. **Model outputs all spaces**: Forward pass runs but outputs token 32 (space) for all positions
   - Mel spectrogram normalization has been verified against Python reference
   - Issue likely in decoder processing or t_embed handling
   - Need reference Python inference to compare intermediate outputs

## Known Issues

### Mel Spectrogram Differences
- Our Rust mel computation differs slightly from torchaudio's MelSpectrogram
- Python reference mel (generated via `scripts/reference_inference.py`) can be loaded for testing
- Root cause: filterbank normalization differences between librosa-style (Rust) and torchaudio

## Reference Materials

- [Voxtral Model Card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Burn Documentation](https://burn.dev/)
- qwen3-tts-rs patterns (../qwen3-tts-rs)
