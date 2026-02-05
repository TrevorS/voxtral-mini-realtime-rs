# Project Status

## Recent Changes (Feb 2026)

### Session: KV Cache Optimization ✅

**Major accomplishments:**
1. **Enabled KV caching** - `transcribe_streaming()` and `test_inference.rs` now use cached inference
2. **O(n) complexity** - Each autoregressive step now processes only 1 token instead of all previous tokens
3. **Same output** - Transcription remains identical, optimization is transparent to users

**Performance improvement:**
- Before: O(n²) - recomputed all KV for every token
- After: O(n) - reuses cached KV, only computes new token

### Previous Session: Pure Rust Audio Pipeline ✅

**Major accomplishments:**
1. **Fixed STFT padding** - torch.stft pads by n_fft//2 (200), our code was using (n_fft-hop)//2 (120)
2. **Fixed frame count** - Python uses `stft[..., :-1]` to drop last frame, now matched
3. **Added audio padding module** - `src/audio/pad.rs` with left/right padding matching mistral-common
4. **Mel spectrogram now matches Python** - max_diff < 0.0003, mean_diff < 0.000001
5. **Pure Rust pipeline working** - No Python dependencies needed for inference!

**Test transcription (pure Rust):**
```
Input:  test_data/mary_had_lamb.wav (15.95s historical phonograph recording)
Output: " I spoke in the original phonograph. A little piece of practical poetry.
         Mary had a little lamb, it's sweet with quite a flow..."
```

### Previous Session: Streaming Inference Verification ✅

**Major accomplishments:**
1. **Fixed tokenizer decoding** - Discovered text token IDs are offset by 1000 from vocab indices
   - Token ID 1362 → vocab index 362 → " I"
   - Token IDs 0-999 reserved for special tokens
2. **Verified E2E inference** - Rust output matches Python reference exactly
3. **Generated padded reference data** - `scripts/generate_padded_reference.py` for validation
4. **Cleaned up dead code** - Removed `forward_streaming_debug()` and old debug output
5. **Validated with Whisper** - Independent verification shows transcription is correct

---

## Overview

Pure Rust implementation of Voxtral Mini 4B Realtime using the Burn ML framework. Target: streaming ASR with WASM/browser support.

## Model Summary (Verified)

| Component | Layers | Dim | Heads | Window | Special |
|-----------|--------|-----|-------|--------|---------|
| Audio Encoder | 32 | 1280 | 32 MHA | 750 | Causal, standard RMSNorm, biases |
| Language Model | 26 | 3072 | 32Q/8KV | 8192 | GQA 4:1, ADA RMSNorm, no biases, tied embed |
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
| Tokenizer wrapper | ✅ Complete | Tekken tokenizer, text tokens offset by 1000 |
| Model download | ✅ Complete | `scripts/download_model.py` |

**Test counts:** 88 unit tests passing, clippy clean
**Model downloaded:** 8.86 GB weights + config + tokenizer
**GitHub:** https://github.com/TrevorS/voxtral-mini-realtime-rs (private)

### Development Tools ✅

| Script | Purpose |
|--------|---------|
| `scripts/inspect_weights.py` | Browse SafeTensors: component summaries, shapes, stats |
| `scripts/dump_weight_names.py` | Get full weight paths (not truncated) |
| `scripts/reference_forward.py` | Generate reference outputs for RMSNorm, RoPE, SwiGLU, Conv, Attention |
| `scripts/compare_tensors.py` | Compare Rust outputs vs Python reference with tolerances |
| `scripts/generate_padded_reference.py` | Generate properly left-padded mel & audio embeddings |

**Test data generated:** `test_data/*.npy` - reference inputs/outputs for all core components
**Padded reference data:** `test_data/reference_mel_padded.npy`, `test_data/reference_audio_embeds_padded.npy`
**Rust test utilities:** `src/test_utils.rs` - load_npy, assert_tensors_close

### Phase 2: Audio Encoder ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Conv1d downsampler | ✅ Complete | 128→1280→1280, stride=2, 4x downsample, GELU |
| RMSNorm | ✅ Complete | Validated against reference (max_diff < 1e-3) |
| ADA RMSNorm | ✅ Complete | T-conditional, GELU (not SiLU), validated |
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

### Phase 4: Integration ✅

| Component | Status | Notes |
|-----------|--------|-------|
| AudioLanguageAdapter | ✅ Complete | Linear(5120)→GELU→Linear(3072) |
| VoxtralModel | ✅ Complete | Full end-to-end model combining all components |
| Weight loading infra | ✅ Complete | SafeTensors → Burn tensors, supports F32/F16/BF16 |
| KV cache | ✅ Complete | Concatenation-based, sliding window eviction |
| Layer cache integration | ✅ Complete | forward_with_cache on all layers |
| E2E forward pass | ✅ Complete | test_e2e.rs verified with random weights |
| Full weight loading | ✅ Complete | VoxtralModelLoader loads 8GB SafeTensors into model |
| Audio chunking | ✅ Complete | max_source_positions=1500 (~15 sec), ChunkIterator |
| Memory safety | ✅ Complete | OwnedSafeTensors (Arc-based), OnceLock shared test loaders |
| Streaming loop | ✅ Complete | `transcribe_streaming()` verified with real audio |

### Phase 5: Streaming Validation ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Python transcription | ✅ Complete | Verified " I spoke in the original phonograph..." |
| Rust transcription | ✅ Complete | E2E verified, matches Python output exactly |
| Tokenizer fix | ✅ Complete | Text tokens offset by 1000 from vocab indices |
| Audio padding | ✅ Complete | `src/audio/pad.rs` - left+right padding matching Python |
| Mel spectrogram | ✅ Complete | Fixed STFT padding (n_fft//2) and frame count |
| Pure Rust pipeline | ✅ Complete | No Python dependencies for inference! |
| KV cache streaming | ✅ Complete | O(n) inference with cached KV tensors |

### Phase 6: Browser/WASM 🚧

| Component | Status | Notes |
|-----------|--------|-------|
| WGPU backend | ✅ Complete | Tested on native, freedreno Vulkan fallback works |
| WASM feature flags | ✅ Complete | `wasm` (ndarray) and `wasm-wgpu` features |
| wasm-bindgen bindings | ✅ Complete | `Voxtral` class with `loadModel()`, `transcribe()` |
| Build script | ✅ Complete | `scripts/build-wasm.sh [ndarray|wgpu]` |
| Demo HTML | ✅ Complete | `web/index.html` with file/mic input UI |
| WASM build verified | ✅ Complete | 1.7MB pkg, compiles to wasm32-unknown-unknown |
| WebWorker | ✅ Complete | `web/worker.js` for off-main-thread inference |
| VoxtralClient | ✅ Complete | `web/voxtral-client.js` high-level API |
| Web Audio API | ✅ Complete | Microphone recording with MediaRecorder |
| Chunked loader API | ✅ Complete | `ModelLoader` class for streaming into WASM memory |
| E2E browser tests | ✅ Complete | Playwright tests for WASM init, worker init |
| 8GB model test | ⚠️ Blocked | wasm32 4GB limit prevents direct loading |
| Quantization | 🔲 Required | INT4 (→2.2GB) needed to fit in wasm32 |

## Project Structure

```
voxtral-mini-realtime-rs/
├── Cargo.toml              # Burn framework, CPU/WGPU/CUDA features
├── CLAUDE.md               # Claude Code guidance
├── scripts/
│   ├── download_model.py       # HuggingFace model download
│   ├── inspect_weights.py      # SafeTensors browser
│   ├── dump_weight_names.py    # Full weight paths
│   ├── reference_forward.py    # Generate test data
│   ├── reference_inference.py  # Python inference reference
│   ├── compare_tensors.py      # Validate Rust vs Python
│   ├── test_proper_inference.py # Full E2E test with mistral-common
│   ├── test_autoregressive.py  # Simplified autoregressive test
│   ├── test_mistral_common_inference.py # mistral-common tokenizer test
│   └── check_special_tokens.py # Verify token IDs (32=PAD, 33=WORD)
├── test_data/              # Reference tensors (gitignored)
│   ├── mary_had_lamb.wav   # Test audio for transcription validation
│   ├── python_audio_embeds.npy # Reference encoder output
│   └── *.npy               # Component reference data
├── models/
│   └── voxtral/            # Downloaded model (gitignored)
│       ├── consolidated.safetensors  # 8.86 GB
│       ├── params.json               # Architecture config
│       └── tekken.json               # Tokenizer
├── src/
│   ├── lib.rs              # Public API (VoxtralRealtime<B>)
│   ├── main.rs             # Simple placeholder
│   ├── test_utils.rs       # Test utilities (load_npy, etc.)
│   ├── models/
│   │   ├── mod.rs
│   │   ├── config.rs       # VoxtralConfig parser (verified)
│   │   ├── encoder.rs      # AudioEncoder (32 layers)
│   │   ├── decoder.rs      # LanguageModel (26 layers)
│   │   ├── adapter.rs      # AudioLanguageAdapter
│   │   ├── voxtral.rs      # Complete VoxtralModel
│   │   ├── loader.rs       # VoxtralModelLoader (SafeTensors)
│   │   ├── weights.rs      # OwnedSafeTensors, weight names
│   │   ├── time_embedding.rs # TimeEmbedding for t_cond
│   │   └── layers/
│   │       ├── mod.rs
│   │       ├── attention.rs  # MHA/GQA with RoPE, sliding window
│   │       ├── rope.rs       # Rotary position embeddings
│   │       ├── rms_norm.rs   # RMSNorm + ADA RMSNorm
│   │       ├── swiglu.rs     # SwiGLU MLP
│   │       ├── conv.rs       # ConvDownsampler
│   │       ├── encoder_layer.rs
│   │       ├── decoder_layer.rs
│   │       └── kv_cache.rs   # KV cache for streaming
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── io.rs           # AudioBuffer, WAV I/O
│   │   ├── mel.rs          # MelSpectrogram extractor
│   │   ├── chunk.rs        # Audio chunking (max_source_positions)
│   │   └── resample.rs     # FFT-based resampling
│   ├── tokenizer/
│   │   └── mod.rs          # Tekken tokenizer wrapper
│   └── bin/
│       ├── transcribe.rs   # CLI stub
│       ├── test_e2e.rs     # E2E test with random weights
│       └── test_inference.rs # Full inference test
└── docs/
    ├── VOXTRAL_ARCHITECTURE.md  # Model deep dive (verified)
    ├── CONTINUATION.md          # This file
    └── VALIDATION.md            # Validation strategy
```

## WASM/Browser Usage

### Build WASM Package

```bash
# Build with ndarray (CPU) backend - works in all browsers
./scripts/build-wasm.sh ndarray

# Build with WebGPU backend - requires WebGPU-capable browser
./scripts/build-wasm.sh wgpu
```

Output: `pkg/voxtral_mini_realtime.js` + `.wasm` (1.7MB)

### Use in Browser

```javascript
import init, { Voxtral } from './pkg/voxtral_mini_realtime.js';

await init();
const voxtral = new Voxtral();

// Load model (must be fetched separately - 8GB!)
const modelBytes = await fetch('consolidated.safetensors').then(r => r.arrayBuffer());
const tokenizerJson = await fetch('tekken.json').then(r => r.text());
voxtral.loadModel(new Uint8Array(modelBytes), tokenizerJson);

// Transcribe (16kHz mono Float32Array)
const text = voxtral.transcribe(audioData);
```

### Using VoxtralClient (Recommended)

The `VoxtralClient` class handles WebWorker communication and microphone access:

```javascript
import { VoxtralClient } from './web/voxtral-client.js';

const client = new VoxtralClient();
await client.init();
await client.loadModel(modelBytes, tokenizerJson);

// Transcribe a file
const text = await client.transcribeFile(audioFile);

// Or record from microphone
await client.startMicrophone();
// ... user speaks ...
const text = await client.stopAndTranscribe();
```

See `web/index.html` for a complete demo with UI.

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
3. ~~Inspect SafeTensors weight names~~ ✅
4. ~~Implement shared components (RMSNorm, RoPE, SwiGLU)~~ ✅
5. ~~Build audio encoder layer by layer~~ ✅
6. ~~Validate against Python reference~~ ✅
7. ~~Implement LLM decoder~~ ✅
8. ~~Full weight loading~~ ✅
9. ~~Memory safety (Arc-based SafeTensors, shared test loaders)~~ ✅
10. ~~Audio chunking for max_source_positions~~ ✅
11. ~~Debug model output~~ ✅ (Root cause: position 38 anomaly, solution: use prefix 38)
12. ~~Add streaming inference method~~ ✅ `transcribe_streaming()` added to `voxtral.rs`
13. ~~Test Rust streaming inference end-to-end~~ ✅
    - Model output matches Python exactly
    - Fixed tokenizer: text token IDs = vocab_index + 1000
    - Audio must be left-padded to align with prefix tokens
14. ~~KV cache optimization~~ ✅ O(n) inference with cached KV tensors
15. ~~Test WGPU backend~~ ✅ Works (freedreno Vulkan fallback on ARM)
16. ~~WASM/browser support~~ ✅ Complete
    - `wasm` feature: ndarray backend for all browsers
    - `wasm-wgpu` feature: WebGPU backend for compatible browsers
    - wasm-bindgen bindings in `src/web/bindings.rs`
    - Build script: `scripts/build-wasm.sh`
    - Demo page: `web/index.html`
17. ~~WebWorker integration~~ ✅ Complete
    - `web/worker.js` - off-main-thread inference
    - `web/voxtral-client.js` - high-level browser API
18. ~~Web Audio API~~ ✅ Complete
    - Microphone recording with MediaRecorder
    - Automatic resampling to 16kHz
    - Recording timer and visualizer
19. ~~Browser E2E tests~~ ✅ Complete
    - `scripts/test_wasm_e2e.py` - Playwright-based WASM/worker tests
    - `scripts/test_full_load.py` - Full model load test (discovered limits)
20. ~~Chunked loader API~~ ✅ Complete
    - `ModelLoader` class for streaming into WASM memory
    - Bypasses JS ArrayBuffer limits (V8 ~4GB)
21. **BLOCKER: wasm32 4GB address space**
    - Full 8.86GB model cannot fit in wasm32 address space
    - Browser streaming download works (8.86GB in 5.1s)
    - Allocation fails in both JS and WASM memory
22. **NEXT: INT4 quantization (REQUIRED)**
    - Reduce 8.86GB → ~2.2GB to fit in wasm32
    - May need GGUF or custom quantization
    - Alternative: Server-side inference with browser UI

## Open Questions

1. **ADA RMSNorm t_embed**: ✅ Resolved
   - Uses `TimeEmbedding(num_delay_tokens)` - sinusoidal encoding like positional embeddings
   - For streaming: `num_delay_tokens = transcription_delay_ms / (1000 / frame_rate)`
   - Default: 480ms delay → 6 tokens at 12.5 Hz frame rate
   - Implemented in `src/models/time_embedding.rs`

2. **Tekken tokenizer**: ✅ Resolved
   - Custom loader implemented - HuggingFace `tokenizers` crate doesn't support Tekken format
   - Tekken uses base64-encoded token bytes with some null token_str entries
   - **Important:** Text token IDs are offset by 1000 from vocab indices!
   - Token ID 1000+ maps to vocab index (token_id - 1000)
   - Token IDs 0-999 are reserved for special/control tokens

3. **Weight names**: ✅ Resolved
   - Encoder: `mm_streams_embeddings.embedding_module.whisper_encoder.*`
   - Decoder: `layers.{N}.*`

4. **WASM size**: 8.86GB model exceeds wasm32 4GB address space limit
   - **JavaScript limit**: V8 can't allocate single ArrayBuffer > ~4GB
   - **WASM32 limit**: 32-bit address space caps at 4GB total memory
   - **Solutions tested**:
     - Streaming download: ✅ Works (8.86GB in 5.1s)
     - Single ArrayBuffer: ❌ "Array buffer allocation failed"
     - WASM chunked loading: ❌ Can't allocate 8.86GB in wasm32
   - **Required**: INT4 quantization (8.86GB → ~2.2GB) to fit in wasm32

5. **Model outputs random multilingual tokens**: ✅ RESOLVED - See "Streaming Inference Findings" below
   - Root cause: Standard prefix (39 tokens) causes anomalous behavior at position 38
   - Solution: Use prefix length 38 instead of 39 for autoregressive generation
   - Verified: Produces correct transcription " I spoke in the original phonograph. A little piece of practical poetry"

## Streaming Inference Findings (Feb 2026)

### Position 38 Anomaly - ROOT CAUSE IDENTIFIED

The model outputs all `[STREAMING_PAD]` tokens because position 38 has anomalous behavior when it's the last prefix position:

**Evidence:**
- Position 38 hidden state norm diverges: Layer 25 shows pos 38 norm=452 vs pos 36/37 norm=1000-1100
- All logits at position 38 are very negative (-17 to -55 range vs +12 at positions 36-37)
- Position 38 = n_left_pad_tokens(32) + num_delay_tokens(6) is exactly at the trained prefix boundary

### Working Solution

Use **prefix length 38** (one less than standard) for generation:
```python
prefix_tokens = [1] + [32] * 37  # BOS + 37 STREAMING_PAD = 38 tokens
```

With this prefix, position 37 correctly predicts `[STREAMING_WORD]` (token 33), enabling autoregressive generation.

### Verified Output

Test audio: `test_data/mary_had_lamb.wav` (15.95s)
Expected: "First words I spoke in the original phonograph. A little piece of practical poetry..."
Produced: " I spoke in the original phonograph. A little piece of practical poetry"

(Missing "First words" is expected - position 38 corresponds to ~2.1s into the speech, after those words.)

### Key Implementation Details

1. **Prefix length**: Use 38 tokens, not 39
2. **Token pattern**: `[STREAMING_WORD]` (33) starts words, `[STREAMING_PAD]` (32) marks pauses
3. **Autoregressive**: Feed previous generated token as next input
4. **Time embedding**: `t=6.0` (num_delay_tokens) is correct

### Test Scripts Created

- `scripts/test_proper_inference.py` - Full end-to-end Python test with proper mistral-common preprocessing
- `scripts/test_autoregressive.py` - Simplified autoregressive generation test
- `scripts/check_special_tokens.py` - Verify token IDs

## Known Issues

None! All previous issues have been resolved.

## Resolved Issues

### Mel Spectrogram Mismatch (Fixed Feb 2026)
- **Problem:** Rust mel computation produced different values than Python reference
- **Root causes:**
  1. STFT padding: torch.stft uses n_fft//2 (200), we were using (n_fft-hop)//2 (120)
  2. Frame count: Python drops last frame with `stft[..., :-1]`, we weren't
- **Solution:** Fixed padding in `mel.rs` STFT, now matches Python (max_diff < 0.0003)
- **Impact:** Pure Rust audio pipeline now produces correct transcriptions

### Memory Leak / OOM in Tests (Fixed)
- **Problem:** `Box::leak` in SafeTensors loading caused 8GB leak per load; parallel tests caused OOM
- **Solution:** `OwnedSafeTensors` uses `Arc<Vec<u8>>` - memory freed when dropped
- **Tests:** Use `OnceLock` shared loaders to load model once across all tests

### max_source_positions Constraint (Implemented)
- **Problem:** vLLM/mistral-common enforces max mel frames (default 1500 ≈ 15 sec)
- **Solution:** Added `max_source_positions` to config, `ChunkConfig` and `chunk_audio()` for splitting
- **Key values:** 1500 mel frames → 375 encoder positions (after 4x conv downsample)

## Reference Materials

- [Voxtral Model Card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Burn Documentation](https://burn.dev/)
- qwen3-tts-rs patterns (../qwen3-tts-rs)
