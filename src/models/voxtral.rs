//! Complete Voxtral Realtime model.
//!
//! Combines audio encoder, adapter, and language model for streaming ASR.

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::adapter::{reshape_encoder_output, AudioLanguageAdapter, AudioLanguageAdapterConfig};
use super::decoder::{LanguageModel, LanguageModelConfig};
use super::encoder::{AudioEncoder, AudioEncoderConfig};
use super::layers::LayerCaches;

/// Complete Voxtral model configuration.
#[derive(Config, Debug)]
pub struct VoxtralModelConfig {
    /// Audio encoder configuration.
    pub encoder: AudioEncoderConfig,
    /// Language model configuration.
    pub decoder: LanguageModelConfig,
    /// Adapter configuration.
    pub adapter: AudioLanguageAdapterConfig,
    /// Reshape factor for encoder output (typically 4).
    #[config(default = 4)]
    pub reshape_factor: usize,
}

impl VoxtralModelConfig {
    /// Create config from the Voxtral model defaults.
    pub fn voxtral() -> Self {
        Self {
            encoder: AudioEncoderConfig::voxtral(),
            decoder: LanguageModelConfig::voxtral(),
            adapter: AudioLanguageAdapterConfig::voxtral(),
            reshape_factor: 4,
        }
    }

    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VoxtralModel<B> {
        let encoder = self.encoder.init(device);
        let decoder = self.decoder.init(device);
        let adapter = self.adapter.init(device);

        VoxtralModel {
            encoder,
            decoder,
            adapter,
            reshape_factor: self.reshape_factor,
        }
    }
}

/// Complete Voxtral Realtime model.
///
/// Architecture:
/// 1. Audio encoder: mel spectrogram -> encoder hidden states
/// 2. Reshape: concatenate adjacent frames (4x downsample)
/// 3. Adapter: project to LLM dimension
/// 4. Language model: generate tokens
///
/// Forward flow:
/// ```text
/// mel [B, 128, T] -> encoder [B, T/4, 1280] -> reshape [B, T/16, 5120]
///   -> adapter [B, T/16, 3072] -> decoder [B, T/16, vocab]
/// ```
#[derive(Module, Debug)]
pub struct VoxtralModel<B: Backend> {
    /// Audio encoder.
    encoder: AudioEncoder<B>,
    /// Language model decoder.
    decoder: LanguageModel<B>,
    /// Audio-to-LLM adapter.
    adapter: AudioLanguageAdapter<B>,
    /// Reshape factor for encoder output.
    reshape_factor: usize,
}

impl<B: Backend> VoxtralModel<B> {
    /// Create model from components (for weight loading).
    pub fn new(
        encoder: AudioEncoder<B>,
        decoder: LanguageModel<B>,
        adapter: AudioLanguageAdapter<B>,
        reshape_factor: usize,
    ) -> Self {
        Self {
            encoder,
            decoder,
            adapter,
            reshape_factor,
        }
    }

    /// Encode audio to hidden states.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    ///
    /// # Returns
    /// Audio hidden states ready for LLM [batch, seq, llm_d_model]
    pub fn encode_audio(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        // Encode mel spectrogram
        let encoder_out = self.encoder.forward(mel, 0);

        // Reshape to combine adjacent frames
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);

        // Project to LLM dimension
        self.adapter.forward(reshaped)
    }

    /// Full forward pass from mel spectrogram to logits.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `t_embed_decoder` - Temporal embedding for decoder [batch, 1, decoder_d_model]
    ///
    /// # Returns
    /// Logits [batch, seq, vocab_size]
    pub fn forward(&self, mel: Tensor<B, 3>, t_embed_decoder: Tensor<B, 3>) -> Tensor<B, 3> {
        // Encode audio
        let audio_hidden = self.encode_audio(mel);

        // Decode through LLM
        let hidden = self
            .decoder
            .forward_hidden(audio_hidden, t_embed_decoder, 0);

        // Compute logits
        self.decoder.lm_head(hidden)
    }

    /// Prefill step: process initial audio and return hidden states.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `t_embed_decoder` - Temporal embedding for decoder
    ///
    /// # Returns
    /// Hidden states [batch, seq, decoder_d_model]
    pub fn prefill(&self, mel: Tensor<B, 3>, t_embed_decoder: Tensor<B, 3>) -> Tensor<B, 3> {
        let audio_hidden = self.encode_audio(mel);
        self.decoder
            .forward_hidden(audio_hidden, t_embed_decoder, 0)
    }

    /// Continue generation from text tokens.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// Logits for next token [batch, seq, vocab_size]
    pub fn generate_step(
        &self,
        token_ids: Tensor<B, 2, Int>,
        t_embed: Tensor<B, 3>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let hidden = self.decoder.forward(token_ids, t_embed, offset);
        self.decoder.lm_head(hidden)
    }

    /// Encode audio with KV cache (for streaming).
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `encoder_cache` - KV cache for encoder layers
    ///
    /// # Returns
    /// Audio hidden states ready for LLM [batch, seq, llm_d_model]
    pub fn encode_audio_with_cache(
        &self,
        mel: Tensor<B, 3>,
        encoder_cache: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let encoder_out = self.encoder.forward_with_cache(mel, encoder_cache);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Full forward pass with KV caches (for streaming).
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `t_embed_decoder` - Temporal embedding for decoder
    /// * `encoder_cache` - KV cache for encoder layers
    /// * `decoder_cache` - KV cache for decoder layers
    ///
    /// # Returns
    /// Logits [batch, seq, vocab_size]
    pub fn forward_with_cache(
        &self,
        mel: Tensor<B, 3>,
        t_embed_decoder: Tensor<B, 3>,
        encoder_cache: &mut LayerCaches<B>,
        decoder_cache: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let audio_hidden = self.encode_audio_with_cache(mel, encoder_cache);
        let hidden =
            self.decoder
                .forward_hidden_with_cache(audio_hidden, t_embed_decoder, decoder_cache);
        self.decoder.lm_head(hidden)
    }

    /// Autoregressive generation step with cache.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq] (typically seq=1 for autoregressive)
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `decoder_cache` - KV cache for decoder layers
    ///
    /// # Returns
    /// Logits for next token [batch, seq, vocab_size]
    pub fn generate_step_with_cache(
        &self,
        token_ids: Tensor<B, 2, Int>,
        t_embed: Tensor<B, 3>,
        decoder_cache: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let hidden = self
            .decoder
            .forward_with_cache(token_ids, t_embed, decoder_cache);
        self.decoder.lm_head(hidden)
    }

    /// Get encoder configuration.
    pub fn encoder(&self) -> &AudioEncoder<B> {
        &self.encoder
    }

    /// Get decoder configuration.
    pub fn decoder(&self) -> &LanguageModel<B> {
        &self.decoder
    }

    /// Create KV caches for the encoder.
    pub fn create_encoder_cache(&self) -> LayerCaches<B> {
        self.encoder.create_cache()
    }

    /// Create KV caches for the decoder.
    pub fn create_decoder_cache(&self) -> LayerCaches<B> {
        self.decoder.create_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_voxtral_model_shape() {
        let device = Default::default();

        // Small config for testing
        let config = VoxtralModelConfig {
            encoder: AudioEncoderConfig {
                n_mels: 128,
                d_model: 64,
                n_layers: 1,
                n_heads: 4,
                head_dim: 16,
                mlp_hidden_dim: 256,
                sliding_window: Some(32),
                max_seq_len: 512,
                rope_theta: 1_000_000.0,
                norm_eps: 1e-5,
            },
            decoder: LanguageModelConfig::new(1000, 32, 1, 2)
                .with_n_kv_heads(1)
                .with_head_dim(16)
                .with_mlp_hidden_dim(128)
                .with_t_cond_dim(8)
                .with_sliding_window(Some(32))
                .with_max_seq_len(512),
            adapter: AudioLanguageAdapterConfig::new(64 * 4, 32, 32), // 64 * 4 = 256 after reshape
            reshape_factor: 4,
        };

        let model = config.init::<TestBackend>(&device);

        // Input: mel spectrogram [batch=1, n_mels=128, time=64]
        let mel = Tensor::<TestBackend, 3>::zeros([1, 128, 64], &device);
        let t_embed_dec = Tensor::<TestBackend, 3>::zeros([1, 1, 32], &device);

        // After conv downsample: 64 -> 16 (4x from two stride-2 convs)
        // After reshape (factor 4): 16 -> 4
        // After adapter: -> LLM dimension
        let logits = model.forward(mel, t_embed_dec);

        // Output: [1, 4, vocab_size=1000]
        assert_eq!(logits.dims()[0], 1);
        assert_eq!(logits.dims()[1], 4);
        assert_eq!(logits.dims()[2], 1000);
    }

    #[test]
    fn test_encode_audio() {
        let device = Default::default();

        let config = VoxtralModelConfig {
            encoder: AudioEncoderConfig {
                n_mels: 128,
                d_model: 64,
                n_layers: 1,
                n_heads: 4,
                head_dim: 16,
                mlp_hidden_dim: 256,
                sliding_window: Some(32),
                max_seq_len: 512,
                rope_theta: 1_000_000.0,
                norm_eps: 1e-5,
            },
            decoder: LanguageModelConfig::new(1000, 32, 1, 2)
                .with_n_kv_heads(1)
                .with_head_dim(16)
                .with_mlp_hidden_dim(128)
                .with_t_cond_dim(8)
                .with_sliding_window(Some(32))
                .with_max_seq_len(512),
            adapter: AudioLanguageAdapterConfig::new(256, 32, 32),
            reshape_factor: 4,
        };

        let model = config.init::<TestBackend>(&device);

        let mel = Tensor::<TestBackend, 3>::zeros([1, 128, 64], &device);

        let audio_hidden = model.encode_audio(mel);

        // After conv: 64/4 = 16
        // After reshape: 16/4 = 4
        // Dimension: adapter output = 32
        assert_eq!(audio_hidden.dims(), [1, 4, 32]);
    }
}
