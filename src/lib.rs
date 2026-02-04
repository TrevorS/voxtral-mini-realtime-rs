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
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use voxtral_mini_realtime::{VoxtralRealtime, TranscriptionOptions};
//!
//! let model = VoxtralRealtime::from_pretrained("path/to/model")?;
//! let audio = voxtral_mini_realtime::audio::load_wav("audio.wav")?;
//! let text = model.transcribe(&audio, None)?;
//! println!("{}", text);
//! ```

pub mod audio;
pub mod models;
pub mod tokenizer;

#[cfg(feature = "hub")]
pub mod hub;

#[cfg(test)]
mod test_utils;

use anyhow::Result;

// Re-exports
pub use audio::AudioBuffer;
pub use models::config::{AudioEncoderConfig, LanguageModelConfig, VoxtralConfig};

/// Main Voxtral Realtime interface for streaming ASR.
pub struct VoxtralRealtime<B: burn::tensor::backend::Backend> {
    /// Audio encoder (causal Whisper-style)
    _audio_encoder: std::marker::PhantomData<B>,
    /// Language model (Ministral-3B based)
    // language_model: LanguageModel<B>,
    /// Audio-to-LLM adapter projection
    // adapter: AudioLanguageAdapter<B>,
    /// Configuration
    config: VoxtralConfig,
}

impl<B: burn::tensor::backend::Backend> VoxtralRealtime<B> {
    /// Load model from a local directory containing weights and config.
    pub fn from_pretrained(_model_path: &str, _device: &B::Device) -> Result<Self> {
        // TODO: Implement weight loading
        todo!("Model loading not yet implemented")
    }

    /// Get the model configuration.
    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    /// Transcribe audio to text.
    pub fn transcribe(
        &self,
        _audio: &AudioBuffer,
        _options: Option<TranscriptionOptions>,
    ) -> Result<String> {
        // TODO: Implement transcription
        todo!("Transcription not yet implemented")
    }
}

/// Options for transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    /// Delay in tokens (1 token = 80ms). Range: 1-30, default: 6 (480ms).
    pub delay_tokens: usize,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            delay_tokens: 6, // 480ms latency
            max_tokens: 4096,
            temperature: 0.0, // Greedy decoding for ASR
        }
    }
}
