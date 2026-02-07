# Changelog

## Unreleased

### Fixed

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
