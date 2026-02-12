# Changelog

## 0.2.0

> Performance numbers measured on NVIDIA DGX Spark (GB10, LPDDR5x) for Vulkan,
> Apple M4 Max for Metal, and headless Chromium (DGX Spark) for WASM/WebGPU.

### Added

- **Long audio chunking** — audio exceeding the GPU's shared-memory limit is
  automatically split into chunks (default: 1200 mel frames) with overlap for
  continuity. Chunks are transcribed sequentially and concatenated.
  Contributed by [@sleep3r](https://github.com/sleep3r) in [#3](https://github.com/TrevorS/voxtral-mini-realtime-rs/pull/3).
- **Criterion pipeline benchmarks** (`cargo bench q4_pipeline`) — sequential
  stage-level benchmarks for model load, preprocessing, encoding, and full
  transcription with regression tracking.

### Performance

- **Q4 native: 0.416 RTF, 19.4 tok/s** (down from 0.535 RTF / 14.5 tok/s).
  Tiled WGSL shader with shared-memory tiling for single-token decode,
  vectorized u32 reads, and vec4 dot products.
- **F32 native: 1.543 RTF, 4.6 tok/s** — Q4 decode is 4.2× faster than F32.

### Fixed

- **Tiled Q4 shader corruption on Metal.** Baking dimensions as compile-time
  WGSL constants via `SourceTemplate` caused CubeCL's pipeline cache to serve
  stale pipelines when many shape variants accumulated during inference. Both
  kernels now read dimensions from a runtime info buffer, producing a single
  cached pipeline per kernel variant.

- **Q4 produces all-pad tokens on quiet audio (44.59% → 8.49% WER on FLEURS
  English).** 37% of FLEURS utterances had peak amplitude below 0.02, producing
  mel spectrograms indistinguishable from silence after log normalization. Added
  `peak_normalize(0.95)` before mel computation so Q4 can resolve subtle
  features. Normal-volume audio is barely affected (~0.05 log-space shift).
- **Q4 inference fails on audio without leading silence.** The Q4_0 quantized
  model is sensitive to speech content in the 38-token streaming prefix. Audio
  that starts immediately with speech (e.g. mic recordings with no silence)
  produced all-pad tokens and "no speech detected." Increased left padding from
  32 to 76 tokens so the full prefix sees only silence. The upstream
  mistral-common default of 32 works for f32 inference but is insufficient for
  Q4. ([details in `src/audio/pad.rs`](src/audio/pad.rs))

### Changed

- HuggingFace Space deployment (static SDK, model shards fetched from CDN)
- Browser UI redesign with weights caching via Cache API
- Audio resampling in browser uses `OfflineAudioContext` for correct 16 kHz conversion
- WASM uses naive-only Q4 kernel dispatch (tiled kernel produces incorrect
  results on WebGPU due to a CubeCL bind group layout issue with mixed binding
  counts)

## 0.1.0

Initial release of Voxtral Mini 4B Realtime in Rust.

- Native CLI for streaming ASR via Vulkan/Metal (F32 SafeTensors path)
- Q4 GGUF quantized inference path (~2.5 GB) for native and browser
- WASM + WebGPU browser demo with client-side model loading
- Custom WGSL shader for fused Q4 dequantization + matrix multiplication
- Sharded GGUF loading to stay within browser memory limits
- Causal encoder (32 layers), GQA decoder (26 layers), audio-language adapter
- Streaming mode with 38-token prefix and lookahead t=6 (480ms)
- 103 unit and integration tests
