//! WASM/browser bindings for Voxtral Mini 4B Realtime (Q4 GGUF).
//!
//! This module provides JavaScript-callable APIs for GPU-accelerated Q4
//! inference in the browser via WebGPU (wgpu backend).
//!
//! ## Usage from JavaScript
//!
//! ```javascript
//! import init, { VoxtralQ4, initWgpuDevice } from './pkg/voxtral_mini_realtime.js';
//!
//! await init();
//! await initWgpuDevice();
//! const voxtral = new VoxtralQ4();
//! voxtral.loadModel(ggufBytes, tokenizerJson);
//!
//! // Transcribe audio (16kHz mono Float32Array)
//! const text = await voxtral.transcribe(audioData);
//! console.log(text);
//! ```

mod bindings;

pub use bindings::*;
