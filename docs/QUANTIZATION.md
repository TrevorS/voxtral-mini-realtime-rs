# Voxtral Quantization for WASM/Browser Deployment

## Overview

The Voxtral model is 8.86 GB in BF16, exceeding the 4GB WASM32 memory limit. Quantization is one strategy to reduce model size for browser deployment.

## Model Parameter Breakdown

| Component | Parameters | BF16 Size | % of Model |
|-----------|-----------|-----------|------------|
| Encoder | 970M | ~1.9 GB | ~22% |
| Decoder | 3,031M | ~6.1 GB | ~68% |
| Embeddings | 403M (131K × 3072) | ~0.8 GB | ~9% |
| Adapter | 25M | ~0.05 GB | ~1% |
| **Total** | **4,429M** | **~8.86 GB** | **100%** |

## Burn Quantization Support

Burn 0.20+ provides post-training quantization (PTQ) with the following schemes:

| Precision | Bits | Store | Size Reduction |
|-----------|------|-------|----------------|
| Q8S | 8-bit signed | Native (i8) | ~50% |
| Q4S | 4-bit signed | PackedU32 | ~75% |

### Backend Support

**NdArray Backend:**
- Q8S: Fully supported (used for all quantized models)
- Q4S: Not implemented (`"Quantization not supported for scheme..."`)
- Conv1d quantization: Fixed — uses per-tensor quantization (block_size=0) instead of per-block

**CubeC/Wgpu Backend:**
- Q8S: In progress
- Q4S: In progress (`"Can't store in u32"`)
- Better GPU-optimized quantization support expected

## Limitations Discovered

### 1. Conv1d Block Quantization (Fixed)

The audio encoder's Conv1d layers failed with per-block quantization:
```
Shape should be compatible shape=[1280, 128, 1]: ShapeError/IncompatibleShape
```
**Root cause**: 1D `BlockSize([32])` is padded to 3D as `[1, 1, 32]`, dividing weight shape `[1280, 128, 1]` produces `[1280, 128, 0]` — a zero-sized dimension.

**Fix**: Force per-tensor quantization (`block_size=0`) for the encoder via `quantize_encoder()`. This avoids the shape mismatch while still quantizing all encoder weights.

### 2. Q4 Not Yet Supported in NdArray

The 4-bit quantization scheme isn't implemented in the NdArray backend:
```
not implemented: Quantization not supported for scheme QuantScheme {
  value: Q4S, param: F32, store: PackedU32(0), level: Block(BlockSize([32])), mode: Symmetric
}
```
Even with the `export_tests` feature flag, Q4 values are stored as i8 (Native) — no actual memory savings over Q8. True packed Q4 (0.5 bytes/param) requires PackedU32 operations that NdArray doesn't implement.

## Working Configurations

### q8-v32k (WASM32 Target)

Q8 quantization for all components + vocabulary truncation from 131K to 32K tokens. **This is the only configuration that fits in wasm32 memory.**

| Component | Precision | Notes |
|-----------|-----------|-------|
| Encoder | Q8 (per-tensor) | Per-tensor avoids Conv1d shape issues |
| Decoder | Q8 (block=32) | Per-block for better accuracy |
| Adapter | Q8 (block=32) | |
| Embeddings | Q8 + truncated | 131K → 32K tokens, saves 301 MB |

**Results:**
- Parameters: 4.13B (down from 4.43B)
- On-disk: **3.93 GB** (3.7 GB compressed)
- Headroom: ~360 MB under wasm32's 4.29 GB limit
- Vocab coverage: BPE assigns lower IDs to more common tokens, so 32K covers most English ASR

```bash
cargo run --release --bin quantize_model --features "cli cpu" -- \
  quantize --preset q8-full --max-vocab 32768 --output models/quantized/voxtral-q8-v32k
```

### q8-full

Q8 quantization for all components, full vocabulary.

| Component | Precision | Notes |
|-----------|-----------|-------|
| Encoder | Q8 (per-tensor) | |
| Decoder | Q8 (block=32) | |
| Adapter | Q8 (block=32) | |

**Results:**
- Parameters: 4.43B
- On-disk: **4.24 GB**
- Does NOT fit in wasm32 (exceeds 4.29 GB limit by ~50 MB)
- Suitable for native/server deployment

### mixed-q8-decoder

Quantizes only the decoder to Q8, keeping encoder and adapter at full precision.

| Component | Precision | % of Model |
|-----------|-----------|------------|
| Encoder | Full (BF16) | ~17% |
| Decoder | Q8 | ~75% |
| Adapter | Full (BF16) | ~8% |

**Results:**
- On-disk: **5.34 GB**
- Does NOT fit in wasm32
- Best accuracy preservation for native deployment

## Size Estimates vs Reality

| Preset | Estimated | Actual | Status |
|--------|-----------|--------|--------|
| full | 8.86 GB | 8.86 GB | Baseline |
| mixed-q8-decoder | 5.54 GB | 5.34 GB | Works |
| q8-full | 4.43 GB | 4.24 GB | Works |
| **q8-v32k** | **~3.9 GB** | **3.93 GB** | **Works — fits wasm32** |
| mixed-q4-decoder | 3.88 GB | N/A | Q4 not supported |
| q4-full | 2.21 GB | N/A | Q4 not supported |

## Phased Inference (Recommended for Browser)

The q8-full model (4.24 GB) doesn't fit in wasm32 memory all at once, but it can be loaded in two sequential phases — encoder and decoder are never in memory simultaneously.

### How It Works

```
Phase 1: Load encoder (0.75 GB) + adapter (0.03 GB) → encode audio → free encoder
Phase 2: Load decoder (3.47 GB) → decode tokens → free decoder

Peak memory = max(0.78, 3.47) = 3.47 GB — fits wasm32 with ~800 MB headroom
```

The audio embeddings passed between phases are tiny: **~1.52 MB** for 15 seconds of audio (shape `[1, 124, 3072]`), so the phase boundary is essentially free.

### Shard Sizes

Generated via `quantize_model shard`:

| Shard | File | Size |
|-------|------|------|
| Encoder | `encoder.mpk.gz` | 0.75 GB |
| Adapter | `adapter.mpk.gz` | 0.03 GB |
| Decoder | `decoder.mpk.gz` | 3.47 GB |
| **Total** | | **4.24 GB** |

A `manifest.json` is also generated with exact byte sizes for each shard.

### Generating Shards

```bash
# First, generate the q8-full quantized model
cargo run --release --bin quantize_model --features "cli cpu" -- \
  quantize --preset q8-full --output models/quantized/voxtral-q8-full

# Then split it into phased shards
cargo run --release --bin quantize_model --features "cli cpu" -- \
  shard --input models/quantized/voxtral-q8-full --output-dir models/shards
```

### Accuracy

Phased inference is **token-for-token identical** to full q8-full inference. The same test audio produces the exact same token IDs and transcription text. No accuracy tradeoff.

### WASM API (Phased)

The `VoxtralPhased` WASM binding in `src/web/bindings.rs` provides:

```
init → loadTokenizer → loadEncoderShard → loadAdapterShard
     → encodeAudio → freeEncoder
     → loadDecoderShard → transcribe → freeDecoder
```

The JS client (`web/voxtral-client.js`) orchestrates this via `transcribePhased(audio, urls)`.

### Memory Budget Caveat

During shard deserialization, both the raw bytes and the deserialized model exist in memory simultaneously. For the decoder shard (3.47 GB), this means ~7 GB peak during loading — exceeding wasm32 limits.

Possible solutions (not yet implemented):
- **Per-layer streaming**: Deserialize one transformer layer at a time
- **Memory-mapped loading**: Use wasm32 linear memory more efficiently
- **Compression tradeoff**: Decompress + deserialize in smaller chunks

The phased architecture is correct and verified on native. The WASM memory-during-deserialization problem is the remaining blocker for actual browser deployment.

## Browser Deployment Options

### 1. Phased Loading with q8-full (Recommended)

Full Q8 model split into encoder/decoder shards loaded sequentially:
- No vocabulary truncation — full 131K token vocabulary
- Token-for-token identical accuracy to full model
- Peak runtime memory: ~3.5 GB (fits wasm32)
- Remaining work: streaming deserialization to handle the decoder shard's loading peak

### 2. Full WASM32 with q8-v32k

The Q8 + 32K vocab model fits within wasm32's 4 GiB address space:
- Model size: 3.93 GB
- Available for activations/runtime: ~360 MB
- Trade-off: Can only generate token IDs < 32768

This enables fully client-side ASR in the browser without a server, but truncated vocabulary degrades transcription quality.

### 3. Hybrid Client-Server

Client runs lightweight mel spectrogram extraction in WASM (~50KB code).
Server runs full model inference and returns transcription text.

```javascript
// Browser code using MelClient
const client = new MelClient();
const mel = client.extractMel(audioSamples);
// Send mel to server via WebSocket
ws.send(mel.buffer);
```

See `src/web/bindings.rs` for `MelClient` implementation.

### 4. wasm64 (Memory64) - Blocked

WebAssembly 3.0 includes Memory64 for >4GB memory.
Currently blocked: `wasm-bindgen` doesn't support the `wasm64-unknown-unknown` target.

## Vocabulary Truncation

The Tekken BPE tokenizer assigns lower IDs to more frequent tokens:
- IDs 0–999: Control/special tokens
- IDs 1000+: Text BPE tokens (most common first)

Truncating from 131K to 32K tokens:
- Saves 301 MB of embedding weight memory
- Covers the vast majority of English ASR output
- Model simply cannot emit token IDs ≥ 32768
- Tied embeddings (LM head reuses tok_embeddings via matmul) mean truncation automatically limits output vocabulary

## Usage

Generate full Q8 model:
```bash
cargo run --release --bin quantize_model --features "cli cpu" -- \
  quantize --preset q8-full --output models/quantized/voxtral-q8-full
```

Generate phased shards (recommended for browser):
```bash
cargo run --release --bin quantize_model --features "cli cpu" -- \
  shard --input models/quantized/voxtral-q8-full --output-dir models/shards
```

Generate wasm32-fit single-file model (truncated vocab):
```bash
cargo run --release --bin quantize_model --features "cli cpu" -- \
  quantize --preset q8-full --max-vocab 32768 --output models/quantized/voxtral-q8-v32k
```

List all presets:
```bash
cargo run --bin quantize_model --features "cli cpu" -- list-presets
```

Test phased inference accuracy:
```bash
cargo run --release --bin test_accuracy --features "cli cpu wgpu" -- --variant phased
```

## Current Status

- **Quantization module**: Complete (`src/quantization/mod.rs`)
- **Quantize binary**: Complete (`cargo run --bin quantize_model`)
- **Phased shard generation**: Complete (`quantize_model shard`)
- **Phased inference**: Verified token-for-token identical to full model
- **WASM phased bindings**: Complete (`VoxtralPhased` in `src/web/bindings.rs`)
- **JS client/worker**: Complete (`web/voxtral-client.js`, `web/worker.js`)
- **Conv1d fix**: Per-tensor quantization for encoder (avoids shape errors)
- **Vocab truncation**: 131K → 32K saves 301 MB
- **Generated models**:
  - `models/quantized/voxtral-q8-v32k.mpk.gz` — 3.93 GB (fits wasm32 single-load)
  - `models/quantized/voxtral-q8-full.mpk.gz` — 4.24 GB
  - `models/quantized/voxtral-mixed-q8.mpk.gz` — 5.34 GB
- **Generated shards** (from q8-full):
  - `models/shards/encoder.mpk.gz` — 0.75 GB
  - `models/shards/adapter.mpk.gz` — 0.03 GB
  - `models/shards/decoder.mpk.gz` — 3.47 GB

## Loading Quantized Models

The quantized models are saved in Burn's NamedMpkGz format (.mpk.gz), not SafeTensors.
To load them requires:

```rust
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};

// First initialize the model structure
let model: VoxtralModel<B> = /* initialize empty or from original */;

// Then load quantized weights
let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
let model = model.load_file("path/to/model", &recorder, &device)?;
```

**TODO**: Implement `VoxtralModel::load_quantized()` helper method.

## Future Work

1. **Streaming deserialization**: Per-layer decoder loading to keep peak memory within wasm32 during deserialization
2. **WASM integration testing**: End-to-end browser test with phased shards
3. **GGUF export**: Convert to GGML format for mature Q4 support (if needed)
4. **wasm64 support**: Track wasm-bindgen Memory64 progress

## Research References

- RedHat's Voxtral-Mini-3B-FP8: Only quantizes decoder Linear layers
- Whisper quantization research: INT4 can improve WER (1.59% vs 1.99% for INT8)
- WebLLM/MLC-LLM: Uses quantization + sharding for browser LLMs
