//! Language Model Decoder for Voxtral.
//!
//! Ministral-3B based decoder with GQA and sliding window attention.

use burn::config::Config;
use burn::module::{Module, Param, ParamId};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::layers::{
    DecoderLayer, DecoderLayerConfig, LayerCaches, RmsNorm, RmsNormConfig, RoPE, RoPEConfig,
};

/// Language model configuration.
#[derive(Config, Debug)]
pub struct LanguageModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of KV heads (for GQA).
    #[config(default = 8)]
    pub n_kv_heads: usize,
    /// Head dimension.
    #[config(default = 128)]
    pub head_dim: usize,
    /// MLP hidden dimension.
    #[config(default = 9216)]
    pub mlp_hidden_dim: usize,
    /// Temporal conditioning dimension for ADA RMSNorm.
    #[config(default = 32)]
    pub t_cond_dim: usize,
    /// Sliding window size for attention.
    pub sliding_window: Option<usize>,
    /// Maximum sequence length for RoPE.
    #[config(default = 16384)]
    pub max_seq_len: usize,
    /// RoPE theta.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
}

impl LanguageModelConfig {
    /// Create a config from the Voxtral model defaults.
    pub fn voxtral() -> Self {
        Self::new(131072, 3072, 26, 32).with_sliding_window(Some(8192))
    }
}

/// Language model decoder module.
///
/// Architecture:
/// 1. Token embeddings (tied with LM head)
/// 2. 26 transformer layers with:
///    - ADA RMSNorm (t-conditioned)
///    - GQA attention (32Q/8KV) with sliding window (8192)
///    - Standard RMSNorm
///    - SwiGLU MLP (no biases)
/// 3. Final RMSNorm
/// 4. LM head (tied to embeddings)
///
/// Input: Token IDs [batch, seq]
/// Output: Logits [batch, seq, vocab_size]
#[derive(Module, Debug)]
pub struct LanguageModel<B: Backend> {
    /// Token embeddings (tied with LM head).
    tok_embeddings: Embedding<B>,
    /// Rotary position embeddings.
    rope: RoPE<B>,
    /// Transformer layers.
    layers: Vec<DecoderLayer<B>>,
    /// Final normalization.
    norm: RmsNorm<B>,
    /// Model dimension (for LM head).
    d_model: usize,
}

impl LanguageModelConfig {
    /// Initialize the language model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LanguageModel<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);

        let rope = RoPEConfig::new(self.head_dim, self.max_seq_len)
            .with_theta(self.rope_theta)
            .init(device);

        let layers = (0..self.n_layers)
            .map(|_| {
                DecoderLayerConfig::new(
                    self.d_model,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                    self.mlp_hidden_dim,
                    self.t_cond_dim,
                )
                .with_sliding_window(self.sliding_window)
                .with_norm_eps(self.norm_eps)
                .init(device)
            })
            .collect();

        let norm = RmsNormConfig::new(self.d_model)
            .with_eps(self.norm_eps)
            .init(device);

        LanguageModel {
            tok_embeddings,
            rope,
            layers,
            norm,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> LanguageModel<B> {
    /// Create language model from components (for weight loading).
    pub fn new(
        tok_embeddings_weight: Tensor<B, 2>,
        rope: RoPE<B>,
        layers: Vec<DecoderLayer<B>>,
        final_norm_weight: Tensor<B, 1>,
        eps: f64,
    ) -> Self {
        let d_model = tok_embeddings_weight.dims()[1];

        let tok_embeddings = Embedding {
            weight: Param::initialized(ParamId::new(), tok_embeddings_weight),
        };

        let norm = RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), final_norm_weight),
                epsilon: eps,
            },
        };

        Self {
            tok_embeddings,
            rope,
            layers,
            norm,
            d_model,
        }
    }

    /// Forward pass returning hidden states (before LM head).
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// Hidden states [batch, seq, d_model]
    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, burn::tensor::Int>,
        t_embed: Tensor<B, 3>,
        offset: usize,
    ) -> Tensor<B, 3> {
        // Token embeddings
        let x = self.tok_embeddings.forward(token_ids);

        // Transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, t_embed.clone(), &self.rope, offset);
        }

        // Final normalization
        self.norm.forward(x)
    }

    /// Get token embeddings (for adding to audio embeddings in multimodal mode).
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq]
    ///
    /// # Returns
    /// Token embeddings [batch, seq, d_model]
    pub fn embed_tokens(&self, token_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.tok_embeddings.forward(token_ids)
    }

    /// Forward pass with hidden states input (for multimodal).
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states [batch, seq, d_model]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// Hidden states [batch, seq, d_model]
    pub fn forward_hidden(
        &self,
        hidden_states: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let mut x = hidden_states;
        for layer in &self.layers {
            x = layer.forward(x, t_embed.clone(), &self.rope, offset);
        }
        self.norm.forward(x)
    }

    /// Compute logits from hidden states (LM head with tied embeddings).
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states [batch, seq, d_model]
    ///
    /// # Returns
    /// Logits [batch, seq, vocab_size]
    pub fn lm_head(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // Tied embeddings: logits = hidden @ embeddings.T
        let [batch, seq, _d_model] = hidden_states.dims();

        // Get embedding weights [vocab_size, d_model]
        let embed_weights = self.tok_embeddings.weight.val();
        let vocab_size = embed_weights.dims()[0];

        // Compute logits: [batch, seq, d_model] @ [d_model, vocab_size] -> [batch, seq, vocab_size]
        let embed_t = embed_weights.transpose().unsqueeze::<3>(); // [1, d_model, vocab_size]
        let logits = hidden_states.matmul(embed_t);

        // Result shape should be [batch, seq, vocab_size]
        logits.reshape([batch, seq, vocab_size])
    }

    /// Forward pass with KV cache (for autoregressive generation).
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `caches` - KV caches for all layers
    ///
    /// # Returns
    /// Hidden states [batch, seq, d_model]
    pub fn forward_with_cache(
        &self,
        token_ids: Tensor<B, 2, burn::tensor::Int>,
        t_embed: Tensor<B, 3>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let x = self.tok_embeddings.forward(token_ids);

        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, t_embed.clone(), &self.rope, cache);
            }
        }

        self.norm.forward(x)
    }

    /// Forward pass with hidden states input and KV cache.
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states [batch, seq, d_model]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    /// * `caches` - KV caches for all layers
    ///
    /// # Returns
    /// Hidden states [batch, seq, d_model]
    pub fn forward_hidden_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let mut x = hidden_states;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, t_embed.clone(), &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// Get the number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Create a new cache for this decoder.
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
    fn test_language_model_shape() {
        let device = Default::default();

        // Small config for testing
        let config = LanguageModelConfig::new(1000, 64, 2, 4)
            .with_n_kv_heads(2)
            .with_head_dim(16)
            .with_mlp_hidden_dim(256)
            .with_t_cond_dim(8)
            .with_sliding_window(Some(32))
            .with_max_seq_len(512);
        let model = config.init::<TestBackend>(&device);

        // Input: [batch=1, seq=10]
        let token_ids = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([1, 10], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([1, 1, 64], &device);

        let hidden = model.forward(token_ids, t_embed, 0);

        assert_eq!(hidden.dims(), [1, 10, 64]);

        // Test LM head
        let logits = model.lm_head(hidden);
        assert_eq!(logits.dims(), [1, 10, 1000]);
    }

    #[test]
    fn test_forward_hidden() {
        let device = Default::default();

        let config = LanguageModelConfig::new(1000, 64, 2, 4)
            .with_n_kv_heads(2)
            .with_head_dim(16)
            .with_mlp_hidden_dim(256)
            .with_t_cond_dim(8)
            .with_sliding_window(Some(32))
            .with_max_seq_len(512);
        let model = config.init::<TestBackend>(&device);

        // Input: hidden states from encoder
        let hidden = Tensor::<TestBackend, 3>::zeros([1, 20, 64], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([1, 1, 64], &device);

        let out = model.forward_hidden(hidden, t_embed, 0);

        assert_eq!(out.dims(), [1, 20, 64]);
    }

    #[test]
    fn test_voxtral_config() {
        let config = LanguageModelConfig::voxtral();

        assert_eq!(config.vocab_size, 131072);
        assert_eq!(config.d_model, 3072);
        assert_eq!(config.n_layers, 26);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.mlp_hidden_dim, 9216);
        assert_eq!(config.t_cond_dim, 32);
        assert_eq!(config.sliding_window, Some(8192));
    }
}
