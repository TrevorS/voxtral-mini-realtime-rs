//! Audio encoder transformer layer.
//!
//! Combines attention, SwiGLU MLP, and RMSNorm into a full layer.

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use burn::module::{Param, ParamId};
use burn::nn::Linear;

use super::attention::{Attention, AttentionConfig};
use super::kv_cache::KVCache;
use super::rms_norm::{RmsNorm, RmsNormConfig};
use super::rope::RoPE;
use super::swiglu::{SwiGLU, SwiGLUConfig};

/// Encoder layer configuration.
#[derive(Config, Debug)]
pub struct EncoderLayerConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// MLP hidden dimension.
    pub mlp_hidden_dim: usize,
    /// Sliding window size for attention.
    pub sliding_window: Option<usize>,
    /// RMSNorm epsilon.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
}

/// Audio encoder transformer layer.
///
/// Architecture:
/// ```text
/// x -> RMSNorm -> Attention -> + -> x'
///                              |
/// x' -> RMSNorm -> SwiGLU -> + -> out
/// ```
///
/// Note: The encoder uses standard RMSNorm (not ADA) based on actual model weights.
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    /// Pre-attention normalization.
    attention_norm: RmsNorm<B>,
    /// Self-attention with RoPE.
    attention: Attention<B>,
    /// Pre-MLP normalization.
    ffn_norm: RmsNorm<B>,
    /// SwiGLU MLP.
    ffn: SwiGLU<B>,
}

impl EncoderLayerConfig {
    /// Initialize the encoder layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderLayer<B> {
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_eps(self.norm_eps)
            .init(device);

        let attention = AttentionConfig::new(self.d_model, self.n_heads, self.head_dim)
            .with_q_bias(true)
            .with_k_bias(false) // K has no bias in encoder
            .with_v_bias(true)
            .with_o_bias(true)
            .with_sliding_window(self.sliding_window)
            .init(device);

        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_eps(self.norm_eps)
            .init(device);

        let ffn = SwiGLUConfig::new(self.d_model, self.mlp_hidden_dim)
            .with_bias(true) // Encoder MLP has bias on w2
            .init(device);

        EncoderLayer {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }
}

impl<B: Backend> EncoderLayer<B> {
    /// Create encoder layer from components (for weight loading).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attention_norm_weight: Tensor<B, 1>,
        attention: Attention<B>,
        ffn_norm_weight: Tensor<B, 1>,
        w1: Linear<B>,
        w2: Linear<B>,
        w3: Linear<B>,
        eps: f64,
    ) -> Self {
        let attention_norm = RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), attention_norm_weight),
                epsilon: eps,
            },
        };
        let ffn_norm = RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), ffn_norm_weight),
                epsilon: eps,
            },
        };
        let ffn = SwiGLU::new(w1, w2, w3);

        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RoPE<B>, offset: usize) -> Tensor<B, 3> {
        // Attention with residual
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, true); // causal=true
        let x = x + residual;

        // MLP with residual
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }

    /// Forward pass with KV cache.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `cache` - KV cache for this layer
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        rope: &RoPE<B>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 3> {
        // Attention with residual
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache, true);
        let x = x + residual;

        // MLP with residual
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layers::rope::RoPEConfig;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_encoder_layer_shape() {
        let device = Default::default();

        // Voxtral encoder config
        let config = EncoderLayerConfig::new(1280, 32, 64, 5120).with_sliding_window(Some(750));
        let layer = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(64, 1024)
            .with_theta(1_000_000.0)
            .init::<TestBackend>(&device);

        // Input: [batch=1, seq=100, d_model=1280]
        let x = Tensor::<TestBackend, 3>::zeros([1, 100, 1280], &device);

        let out = layer.forward(x, &rope, 0);

        assert_eq!(out.dims(), [1, 100, 1280]);
    }
}
