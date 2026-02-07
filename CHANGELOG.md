# Changelog

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
