//! WASM/browser bindings for Voxtral Mini 4B Realtime.
//!
//! This module provides JavaScript-callable APIs for GPU-accelerated audio
//! transcription in the browser via WebGPU (wgpu backend).
//!
//! ## Usage from JavaScript
//!
//! ```javascript
//! import init, { Voxtral } from './pkg/voxtral_mini_realtime.js';
//!
//! await init();
//! const voxtral = new Voxtral();
//! await voxtral.loadModel(modelBytes);
//!
//! // Transcribe audio (16kHz mono Float32Array)
//! const text = await voxtral.transcribe(audioData);
//! console.log(text);
//! ```

mod bindings;

pub use bindings::*;
