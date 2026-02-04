//! Language model decoder layer.
//!
//! GQA attention with sliding window and no biases.

use burn::config::Config;
use burn::module::{Module, Param, ParamId};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::attention::{Attention, AttentionConfig};
use super::kv_cache::KVCache;
use super::rms_norm::{AdaRmsNorm, AdaRmsNormConfig, RmsNorm, RmsNormConfig};
use super::rope::RoPE;
use super::swiglu::{SwiGLU, SwiGLUConfig};

/// Decoder layer configuration.
#[derive(Config, Debug)]
pub struct DecoderLayerConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of KV heads (for GQA).
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// MLP hidden dimension.
    pub mlp_hidden_dim: usize,
    /// Temporal conditioning dimension for ADA RMSNorm.
    pub t_cond_dim: usize,
    /// Sliding window size for attention.
    pub sliding_window: Option<usize>,
    /// RMSNorm epsilon.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
}

/// Language model decoder layer.
///
/// Architecture (Pre-LN with ADA modulation):
/// ```text
/// x -> RMSNorm -> Attention -> + -> x'
///                              |
/// x' -> RMSNorm -> ADA_mod(t_embed) -> SwiGLU -> + -> out
/// ```
///
/// The ADA modulation happens AFTER ffn_norm and BEFORE the MLP,
/// per vLLM's implementation: hidden_states * (1 + ada_rms_norm_t_cond(t_cond))
///
/// Key differences from encoder:
/// - Uses GQA (32 query heads, 8 KV heads)
/// - No biases on linear layers
/// - Larger sliding window (8192)
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    /// Pre-attention ADA normalization.
    ada_rms_norm: AdaRmsNorm<B>,
    /// Pre-attention standard normalization.
    attention_norm: RmsNorm<B>,
    /// Self-attention with GQA.
    attention: Attention<B>,
    /// Pre-MLP normalization.
    ffn_norm: RmsNorm<B>,
    /// SwiGLU MLP.
    ffn: SwiGLU<B>,
}

impl DecoderLayerConfig {
    /// Initialize the decoder layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderLayer<B> {
        let ada_rms_norm = AdaRmsNormConfig::new(self.d_model, self.t_cond_dim)
            .with_eps(self.norm_eps)
            .init(device);

        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_eps(self.norm_eps)
            .init(device);

        let attention = AttentionConfig::new(self.d_model, self.n_heads, self.head_dim)
            .with_n_kv_heads(Some(self.n_kv_heads))
            .with_q_bias(false) // No biases in LLM
            .with_k_bias(false)
            .with_v_bias(false)
            .with_o_bias(false)
            .with_sliding_window(self.sliding_window)
            .init(device);

        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_eps(self.norm_eps)
            .init(device);

        let ffn = SwiGLUConfig::new(self.d_model, self.mlp_hidden_dim)
            .with_bias(false) // No biases in LLM
            .init(device);

        DecoderLayer {
            ada_rms_norm,
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }
}

impl<B: Backend> DecoderLayer<B> {
    /// Create decoder layer from components (for weight loading).
    ///
    /// Note: The ada_norm weights are stored differently in SafeTensors:
    /// - ada_norm_down: [t_cond_dim, d_model] - projects d_model -> t_cond_dim
    /// - ada_norm_up: [d_model, t_cond_dim] - projects t_cond_dim -> d_model
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ada_norm_down: Tensor<B, 2>,
        ada_norm_up: Tensor<B, 2>,
        attention_norm_weight: Tensor<B, 1>,
        attention: Attention<B>,
        ffn_norm_weight: Tensor<B, 1>,
        w1: Linear<B>,
        w2: Linear<B>,
        w3: Linear<B>,
        _t_cond_dim: usize,
        eps: f64,
    ) -> Self {
        use crate::models::weights::linear_from_weights;

        // ADA RMSNorm uses w0 (down) and w2 (up) projections
        // ada_norm_down: [t_cond_dim, d_model] -> w0 Linear(d_model, t_cond_dim)
        // ada_norm_up: [d_model, t_cond_dim] -> w2 Linear(t_cond_dim, d_model)
        let w0 = linear_from_weights(ada_norm_down, None);
        let ada_w2 = linear_from_weights(ada_norm_up, None);
        let ada_rms_norm = AdaRmsNorm::new(w0, ada_w2, eps);

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
            ada_rms_norm,
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
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        rope: &RoPE<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        // Attention with residual
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, true);
        let x = x + residual;

        // MLP with residual
        // ADA modulation happens AFTER ffn_norm and BEFORE MLP (per vLLM)
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ada_rms_norm.forward(x, t_embed.clone());
        let x = self.ffn.forward(x);
        x + residual
    }

    /// Forward pass with KV cache.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `cache` - KV cache for this layer
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        rope: &RoPE<B>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 3> {
        // Attention with residual
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache, true);
        let x = x + residual;

        // MLP with residual
        // ADA modulation happens AFTER ffn_norm and BEFORE MLP (per vLLM)
        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ada_rms_norm.forward(x, t_embed.clone());
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
    fn test_decoder_layer_shape() {
        let device = Default::default();

        // Voxtral LLM config
        let config =
            DecoderLayerConfig::new(3072, 32, 8, 128, 9216, 32).with_sliding_window(Some(8192));
        let layer = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(128, 16384)
            .with_theta(1_000_000.0)
            .init::<TestBackend>(&device);

        // Input: [batch=1, seq=10, d_model=3072]
        let x = Tensor::<TestBackend, 3>::zeros([1, 10, 3072], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([1, 1, 3072], &device);

        let out = layer.forward(x, t_embed, &rope, 0);

        assert_eq!(out.dims(), [1, 10, 3072]);
    }

    #[test]
    fn test_decoder_layer_small() {
        let device = Default::default();

        // Small config for faster testing
        let config = DecoderLayerConfig::new(64, 4, 2, 16, 256, 8).with_sliding_window(Some(32));
        let layer = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(16, 512)
            .with_theta(1_000_000.0)
            .init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 20, 64], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([2, 1, 64], &device);

        let out = layer.forward(x, t_embed, &rope, 0);

        assert_eq!(out.dims(), [2, 20, 64]);
    }
}
