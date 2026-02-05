//! WASM/browser bindings for Voxtral Mini 4B Realtime.
//!
//! This module provides JavaScript-callable APIs for audio transcription in the browser.
//! Uses wasm-bindgen for JS interop and supports both ndarray (CPU) and wgpu (WebGPU) backends.
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

#[cfg(feature = "wasm")]
mod bindings;

#[cfg(feature = "wasm")]
pub use bindings::*;

#[cfg(feature = "wasm-wgpu")]
mod bindings_wgpu;

#[cfg(feature = "wasm-wgpu")]
pub use bindings_wgpu::*;
