//! # Voxtral Mini 4B Realtime
//!
//! Streaming automatic speech recognition (ASR) in Rust using the Burn framework.
//! Port of Mistral's Voxtral Mini 4B Realtime model with WASM/browser support as a key goal.
//!
//! ## Architecture
//!
//! The model consists of two main components:
//!
//! 1. **Audio Encoder** (~0.6B params): Causal Whisper-style encoder that processes mel spectrograms
//!    with sliding window attention (750 tokens) for infinite streaming support.
//!
//! 2. **Language Model** (~3.4B params): Ministral-3B based decoder with GQA attention (32 Q / 8 KV heads)
//!    and sliding window attention (8192 tokens).
//!
//! ## Key Features
//!
//! - **Streaming-first**: Causal attention in the audio encoder enables real-time transcription
//! - **Configurable latency**: 80ms-2.4s delay (sweet spot: 480ms = 6 tokens lookahead)
//! - **Backend-agnostic**: Works with CPU, CUDA, Metal, and WebGPU via Burn

pub mod audio;
#[cfg(feature = "wgpu")]
pub mod gguf;
pub mod models;
pub mod tokenizer;

#[cfg(feature = "wasm")]
pub mod web;

#[cfg(feature = "hub")]
pub mod hub;

#[cfg(test)]
mod test_utils;

// Re-exports
pub use audio::AudioBuffer;
