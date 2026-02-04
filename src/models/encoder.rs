//! Audio Encoder for Voxtral.
//!
//! Whisper-style causal encoder with sliding window attention.

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::layers::{
    ConvDownsampler, ConvDownsamplerConfig, EncoderLayer, EncoderLayerConfig, LayerCaches, RoPE,
    RoPEConfig,
};

/// Audio encoder configuration.
#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    /// Number of mel frequency bins.
    pub n_mels: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// MLP hidden dimension.
    pub mlp_hidden_dim: usize,
    /// Sliding window size for attention.
    pub sliding_window: Option<usize>,
    /// Maximum sequence length for RoPE.
    #[config(default = 4096)]
    pub max_seq_len: usize,
    /// RoPE theta.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
}

impl AudioEncoderConfig {
    /// Create a config from the Voxtral model defaults.
    pub fn voxtral() -> Self {
        Self {
            n_mels: 128,
            d_model: 1280,
            n_layers: 32,
            n_heads: 32,
            head_dim: 64,
            mlp_hidden_dim: 5120,
            sliding_window: Some(750),
            max_seq_len: 4096,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
        }
    }
}

/// Audio encoder module.
///
/// Architecture:
/// 1. Conv downsampler (128 -> 1280, 4x temporal downsample)
/// 2. 32 transformer layers with:
///    - ADA RMSNorm (t-conditioned)
///    - Causal self-attention with sliding window (750)
///    - Standard RMSNorm
///    - SwiGLU MLP
///
/// Input: Mel spectrogram [batch, n_mels, time]
/// Output: Hidden states [batch, time/4, d_model]
#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    /// Convolutional downsampler.
    conv: ConvDownsampler<B>,
    /// Rotary position embeddings.
    rope: RoPE<B>,
    /// Transformer layers.
    layers: Vec<EncoderLayer<B>>,
}

impl AudioEncoderConfig {
    /// Initialize the audio encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioEncoder<B> {
        let conv = ConvDownsamplerConfig::new(self.n_mels, self.d_model, self.d_model).init(device);

        let rope = RoPEConfig::new(self.head_dim, self.max_seq_len)
            .with_theta(self.rope_theta)
            .init(device);

        let layers = (0..self.n_layers)
            .map(|_| {
                EncoderLayerConfig::new(
                    self.d_model,
                    self.n_heads,
                    self.head_dim,
                    self.mlp_hidden_dim,
                )
                .with_sliding_window(self.sliding_window)
                .with_norm_eps(self.norm_eps)
                .init(device)
            })
            .collect();

        AudioEncoder { conv, rope, layers }
    }
}

impl<B: Backend> AudioEncoder<B> {
    /// Create encoder from components (for weight loading).
    pub fn new(conv: ConvDownsampler<B>, rope: RoPE<B>, layers: Vec<EncoderLayer<B>>) -> Self {
        Self { conv, rope, layers }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `offset` - Position offset for KV cache (streaming)
    ///
    /// # Returns
    /// Hidden states [batch, time/4, d_model]
    pub fn forward(&self, mel: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        // Conv downsampler: [batch, n_mels, time] -> [batch, d_model, time/4]
        let x = self.conv.forward(mel);

        // Transpose for transformer: [batch, d_model, time] -> [batch, time, d_model]
        let x = x.swap_dims(1, 2);

        // Transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, offset);
        }

        x
    }

    /// Forward pass with KV cache.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    /// * `caches` - KV caches for all layers
    ///
    /// # Returns
    /// Hidden states [batch, time/4, d_model]
    pub fn forward_with_cache(
        &self,
        mel: Tensor<B, 3>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        // Conv downsampler
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);

        // Transformer layers with cache
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, cache);
            }
        }

        x
    }

    /// Get the number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Create a new cache for this encoder.
    pub fn create_cache(&self) -> LayerCaches<B> {
        LayerCaches::new(self.layers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_audio_encoder_shape() {
        let device = Default::default();

        // Small config for testing (fewer layers)
        let config = AudioEncoderConfig {
            n_mels: 128,
            d_model: 64, // Smaller for testing
            n_layers: 2, // Fewer layers for speed
            n_heads: 4,
            head_dim: 16,
            mlp_hidden_dim: 256,
            sliding_window: Some(32),
            max_seq_len: 512,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
        };
        let encoder = config.init::<TestBackend>(&device);

        // Input: [batch=1, n_mels=128, time=100]
        let mel = Tensor::<TestBackend, 3>::zeros([1, 128, 100], &device);

        let out = encoder.forward(mel, 0);

        // Output should be [1, 25, 64] (100/4 = 25 after conv downsampling)
        assert_eq!(out.dims()[0], 1);
        assert_eq!(out.dims()[1], 25); // 100 -> 50 -> 25 (two stride-2 convs)
        assert_eq!(out.dims()[2], 64);
    }

    #[test]
    fn test_voxtral_config() {
        let config = AudioEncoderConfig::voxtral();

        assert_eq!(config.n_mels, 128);
        assert_eq!(config.d_model, 1280);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.mlp_hidden_dim, 5120);
        assert_eq!(config.sliding_window, Some(750));
    }
}
