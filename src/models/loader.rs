//! Model weight loading from SafeTensors into Burn models.
//!
//! This module provides the VoxtralModelLoader for loading pretrained weights
//! from SafeTensors format into the VoxtralModel.

use anyhow::Result;
use burn::module::{Param, ParamId};
use burn::nn::conv::Conv1d;
use burn::nn::Linear;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use std::path::Path;
use tracing::info;

use super::adapter::AudioLanguageAdapter;
use super::decoder::{LanguageModel, LanguageModelConfig};
use super::encoder::{AudioEncoder, AudioEncoderConfig};
use super::layers::{Attention, ConvDownsampler, DecoderLayer, EncoderLayer, RoPEConfig};
use super::voxtral::VoxtralModel;
use super::weights::{
    adapter_weight_names, conv_weight_names, decoder_layer_weight_names,
    encoder_layer_weight_names, linear_from_weights, load_safetensors, load_tensor, prefixes,
    OwnedSafeTensors,
};

/// Loads pretrained Voxtral model from SafeTensors.
///
/// Owns the loaded weights and provides methods to construct model components.
pub struct VoxtralModelLoader {
    safetensors: OwnedSafeTensors,
}

impl VoxtralModelLoader {
    /// Create a new loader from a SafeTensors file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let safetensors = load_safetensors(path)?;
        Ok(Self { safetensors })
    }

    /// Create a new loader from raw SafeTensors bytes.
    ///
    /// This is useful for WASM where file access is not available.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let safetensors = OwnedSafeTensors::from_bytes(bytes)?;
        Ok(Self { safetensors })
    }

    /// Load the complete VoxtralModel with pretrained weights.
    pub fn load<B: Backend>(&self, device: &B::Device) -> Result<VoxtralModel<B>> {
        self.load_with_options(device, None)
    }

    /// Load with options for memory optimization.
    ///
    /// `max_vocab_size` truncates the embedding table to save memory.
    /// For wasm32 deployment, use 32768 to save ~302 MB.
    /// The model can only generate token IDs < max_vocab_size.
    pub fn load_with_options<B: Backend>(
        &self,
        device: &B::Device,
        max_vocab_size: Option<usize>,
    ) -> Result<VoxtralModel<B>> {
        info!("Loading Voxtral model");

        info!(layers = 32, "Loading audio encoder");
        let encoder = self.load_encoder(device)?;

        info!("Loading audio-language adapter");
        let adapter = self.load_adapter(device)?;

        info!(layers = 26, "Loading language model");
        let decoder = self.load_decoder_with_vocab(device, max_vocab_size)?;

        info!("Model loaded");

        Ok(VoxtralModel::new(encoder, decoder, adapter, 4))
    }

    /// Load the audio encoder with pretrained weights.
    pub fn load_encoder<B: Backend>(&self, device: &B::Device) -> Result<AudioEncoder<B>> {
        use super::layers::RmsNorm;
        use super::weights::prefixes;

        let config = AudioEncoderConfig::voxtral();

        // Load conv downsampler
        let conv = self.load_conv_downsampler(device)?;

        // Load RoPE (computed, not from weights)
        let rope = RoPEConfig::new(config.head_dim, config.max_seq_len)
            .with_theta(config.rope_theta)
            .init(device);

        // Load encoder layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = self.load_encoder_layer(i, &config, device)?;
            layers.push(layer);
        }

        // Load final layer norm
        let norm_weight_name = format!("{}.transformer.norm.weight", prefixes::ENCODER);
        let norm_weight: Tensor<B, 1> = load_tensor(&self.safetensors, &norm_weight_name, device)?;
        let norm = RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), norm_weight),
                epsilon: config.norm_eps,
            },
        };

        Ok(AudioEncoder::new(conv, rope, layers, norm))
    }

    /// Load the conv downsampler with pretrained weights.
    fn load_conv_downsampler<B: Backend>(&self, device: &B::Device) -> Result<ConvDownsampler<B>> {
        let names = conv_weight_names();

        let conv1_weight: Tensor<B, 3> =
            load_tensor(&self.safetensors, &names.conv1_weight, device)?;
        let conv1_bias: Tensor<B, 1> = load_tensor(&self.safetensors, &names.conv1_bias, device)?;
        let conv2_weight: Tensor<B, 3> =
            load_tensor(&self.safetensors, &names.conv2_weight, device)?;
        let conv2_bias: Tensor<B, 1> = load_tensor(&self.safetensors, &names.conv2_bias, device)?;

        let conv1 = conv1d_from_weights(conv1_weight, Some(conv1_bias));
        let conv2 = conv1d_from_weights(conv2_weight, Some(conv2_bias));

        Ok(ConvDownsampler::new(conv1, conv2))
    }

    /// Load a single encoder layer with pretrained weights.
    fn load_encoder_layer<B: Backend>(
        &self,
        layer_idx: usize,
        config: &AudioEncoderConfig,
        device: &B::Device,
    ) -> Result<EncoderLayer<B>> {
        let names = encoder_layer_weight_names(layer_idx);

        // Load attention norm
        let attention_norm_weight: Tensor<B, 1> =
            load_tensor(&self.safetensors, &names.attention_norm, device)?;

        // Load attention weights
        let wq =
            self.load_linear_with_optional_bias(&names.wq_weight, Some(&names.wq_bias), device)?;
        let wk = self.load_linear_without_bias(&names.wk_weight, device)?;
        let wv =
            self.load_linear_with_optional_bias(&names.wv_weight, Some(&names.wv_bias), device)?;
        let wo =
            self.load_linear_with_optional_bias(&names.wo_weight, Some(&names.wo_bias), device)?;

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.n_heads,
            config.n_heads, // MHA: n_kv_heads = n_heads
            config.head_dim,
            config.sliding_window,
        );

        // Load FFN norm
        let ffn_norm_weight: Tensor<B, 1> =
            load_tensor(&self.safetensors, &names.ffn_norm, device)?;

        // Load SwiGLU weights
        let w1 = self.load_linear_without_bias(&names.w1_weight, device)?;
        let w2 =
            self.load_linear_with_optional_bias(&names.w2_weight, Some(&names.w2_bias), device)?;
        let w3 = self.load_linear_without_bias(&names.w3_weight, device)?;

        Ok(EncoderLayer::new(
            attention_norm_weight,
            attention,
            ffn_norm_weight,
            w1,
            w2,
            w3,
            config.norm_eps,
        ))
    }

    /// Load the audio-language adapter with pretrained weights.
    pub fn load_adapter<B: Backend>(&self, device: &B::Device) -> Result<AudioLanguageAdapter<B>> {
        let names = adapter_weight_names();

        let linear1 = self.load_linear_without_bias(&names.linear1_weight, device)?;
        let linear2 = self.load_linear_without_bias(&names.linear2_weight, device)?;

        Ok(AudioLanguageAdapter::new(linear1, linear2))
    }
    /// Load decoder with optional vocabulary truncation.
    pub fn load_decoder_with_vocab<B: Backend>(
        &self,
        device: &B::Device,
        max_vocab_size: Option<usize>,
    ) -> Result<LanguageModel<B>> {
        let config = LanguageModelConfig::voxtral();

        // Load token embeddings
        let mut tok_embeddings: Tensor<B, 2> =
            load_tensor(&self.safetensors, prefixes::TOK_EMBEDDINGS, device)?;

        // Truncate vocabulary if requested (saves memory for wasm32)
        if let Some(max_vocab) = max_vocab_size {
            let [full_vocab, d_model] = tok_embeddings.dims();
            if max_vocab < full_vocab {
                info!(
                    from = full_vocab,
                    to = max_vocab,
                    saved_mb = (full_vocab - max_vocab) * d_model / 1_000_000,
                    "Truncating vocabulary"
                );
                tok_embeddings = tok_embeddings.slice([0..max_vocab, 0..d_model]);
            }
        }

        // Load RoPE (computed, not from weights)
        let rope = RoPEConfig::new(config.head_dim, config.max_seq_len)
            .with_theta(config.rope_theta)
            .init(device);

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = self.load_decoder_layer(i, &config, device)?;
            layers.push(layer);
        }

        // Load final norm
        let final_norm_weight: Tensor<B, 1> =
            load_tensor(&self.safetensors, prefixes::FINAL_NORM, device)?;

        Ok(LanguageModel::new(
            tok_embeddings,
            rope,
            layers,
            final_norm_weight,
            config.norm_eps,
        ))
    }

    /// Load a single decoder layer with pretrained weights.
    pub fn load_decoder_layer<B: Backend>(
        &self,
        layer_idx: usize,
        config: &LanguageModelConfig,
        device: &B::Device,
    ) -> Result<DecoderLayer<B>> {
        let names = decoder_layer_weight_names(layer_idx);

        // Load ADA RMSNorm conditioning weights
        let ada_norm_down: Tensor<B, 2> =
            load_tensor(&self.safetensors, &names.ada_norm_down, device)?;
        let ada_norm_up: Tensor<B, 2> = load_tensor(&self.safetensors, &names.ada_norm_up, device)?;

        // Load attention norm
        let attention_norm_weight: Tensor<B, 1> =
            load_tensor(&self.safetensors, &names.attention_norm, device)?;

        // Load attention weights (no biases in decoder)
        let wq = self.load_linear_without_bias(&names.wq_weight, device)?;
        let wk = self.load_linear_without_bias(&names.wk_weight, device)?;
        let wv = self.load_linear_without_bias(&names.wv_weight, device)?;
        let wo = self.load_linear_without_bias(&names.wo_weight, device)?;

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.sliding_window,
        );

        // Load FFN norm
        let ffn_norm_weight: Tensor<B, 1> =
            load_tensor(&self.safetensors, &names.ffn_norm, device)?;

        // Load SwiGLU weights (no biases in decoder)
        let w1 = self.load_linear_without_bias(&names.w1_weight, device)?;
        let w2 = self.load_linear_without_bias(&names.w2_weight, device)?;
        let w3 = self.load_linear_without_bias(&names.w3_weight, device)?;

        Ok(DecoderLayer::new(
            ada_norm_down,
            ada_norm_up,
            attention_norm_weight,
            attention,
            ffn_norm_weight,
            w1,
            w2,
            w3,
            config.t_cond_dim,
            config.norm_eps,
        ))
    }

    /// Load just the token embeddings from safetensors.
    ///
    /// Returns a Burn `Embedding` module with the pretrained weight.
    pub fn load_tok_embeddings<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<burn::nn::Embedding<B>> {
        use super::weights::prefixes;

        let weight: Tensor<B, 2> =
            load_tensor(&self.safetensors, prefixes::TOK_EMBEDDINGS, device)?;
        Ok(burn::nn::Embedding {
            weight: Param::initialized(ParamId::new(), weight),
        })
    }

    /// Load just the final RMS normalization from safetensors.
    ///
    /// Returns the custom `RmsNorm` wrapper with the pretrained weight.
    pub fn load_final_norm<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<super::layers::RmsNorm<B>> {
        use super::layers::RmsNorm;
        use super::weights::prefixes;

        let config = LanguageModelConfig::voxtral();
        let weight: Tensor<B, 1> = load_tensor(&self.safetensors, prefixes::FINAL_NORM, device)?;
        Ok(RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), weight),
                epsilon: config.norm_eps,
            },
        })
    }

    /// Load a Linear layer without bias.
    fn load_linear_without_bias<B: Backend>(
        &self,
        weight_name: &str,
        device: &B::Device,
    ) -> Result<Linear<B>> {
        let weight: Tensor<B, 2> = load_tensor(&self.safetensors, weight_name, device)?;
        Ok(linear_from_weights(weight, None))
    }

    // --- Static methods that accept borrowed SafeTensors (no Vec copy) ---

    /// Load token embeddings directly from raw safetensors bytes.
    ///
    /// Uses `load_tensor_raw` to bypass `SafeTensors::deserialize`, which
    /// overflows on wasm32 for the 402M-element embedding tensor.
    pub fn tok_embeddings_from_raw<B: Backend>(
        bytes: &[u8],
        device: &B::Device,
    ) -> Result<burn::nn::Embedding<B>> {
        use super::weights::{load_tensor_raw, prefixes};

        let weight: Tensor<B, 2> = load_tensor_raw(bytes, prefixes::TOK_EMBEDDINGS, device)?;
        Ok(burn::nn::Embedding {
            weight: Param::initialized(ParamId::new(), weight),
        })
    }

    /// Load final norm from borrowed SafeTensors bytes.
    pub fn final_norm_from_st<B: Backend>(
        st: &safetensors::SafeTensors,
        device: &B::Device,
    ) -> Result<super::layers::RmsNorm<B>> {
        use super::layers::RmsNorm;
        use super::weights::prefixes;

        let config = LanguageModelConfig::voxtral();
        let weight: Tensor<B, 1> = load_tensor(st, prefixes::FINAL_NORM, device)?;
        Ok(RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), weight),
                epsilon: config.norm_eps,
            },
        })
    }

    /// Load a single decoder layer from borrowed SafeTensors bytes.
    pub fn decoder_layer_from_st<B: Backend>(
        st: &safetensors::SafeTensors,
        layer_idx: usize,
        config: &LanguageModelConfig,
        device: &B::Device,
    ) -> Result<DecoderLayer<B>> {
        let names = decoder_layer_weight_names(layer_idx);

        let ada_norm_down: Tensor<B, 2> = load_tensor(st, &names.ada_norm_down, device)?;
        let ada_norm_up: Tensor<B, 2> = load_tensor(st, &names.ada_norm_up, device)?;
        let attention_norm_weight: Tensor<B, 1> = load_tensor(st, &names.attention_norm, device)?;

        let wq = linear_from_weights(load_tensor(st, &names.wq_weight, device)?, None);
        let wk = linear_from_weights(load_tensor(st, &names.wk_weight, device)?, None);
        let wv = linear_from_weights(load_tensor(st, &names.wv_weight, device)?, None);
        let wo = linear_from_weights(load_tensor(st, &names.wo_weight, device)?, None);

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.sliding_window,
        );

        let ffn_norm_weight: Tensor<B, 1> = load_tensor(st, &names.ffn_norm, device)?;

        let w1 = linear_from_weights(load_tensor(st, &names.w1_weight, device)?, None);
        let w2 = linear_from_weights(load_tensor(st, &names.w2_weight, device)?, None);
        let w3 = linear_from_weights(load_tensor(st, &names.w3_weight, device)?, None);

        Ok(DecoderLayer::new(
            ada_norm_down,
            ada_norm_up,
            attention_norm_weight,
            attention,
            ffn_norm_weight,
            w1,
            w2,
            w3,
            config.t_cond_dim,
            config.norm_eps,
        ))
    }

    /// Load a Linear layer with optional bias.
    fn load_linear_with_optional_bias<B: Backend>(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        device: &B::Device,
    ) -> Result<Linear<B>> {
        let weight: Tensor<B, 2> = load_tensor(&self.safetensors, weight_name, device)?;
        let bias = if let Some(name) = bias_name {
            if self.safetensors.tensor(name).is_ok() {
                Some(load_tensor::<B, 1>(&self.safetensors, name, device)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(linear_from_weights(weight, bias))
    }
}

/// Create a Conv1d layer from weight tensors.
fn conv1d_from_weights<B: Backend>(weight: Tensor<B, 3>, bias: Option<Tensor<B, 1>>) -> Conv1d<B> {
    use burn::module::Ignored;

    // Conv1d weight shape: [out_channels, in_channels/groups, kernel_size]
    // Same in both PyTorch and Burn, no transpose needed
    Conv1d {
        weight: Param::initialized(ParamId::new(), weight),
        bias: bias.map(|b| Param::initialized(ParamId::new(), b)),
        stride: 2,
        kernel_size: 3,
        dilation: 1,
        groups: 1,
        padding: Ignored(burn::nn::PaddingConfig1d::Explicit(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    type TestBackend = Wgpu;

    fn model_path() -> PathBuf {
        PathBuf::from("models/voxtral/consolidated.safetensors")
    }

    /// Shared model loader for tests - loads safetensors once and reuses.
    /// This prevents OOM from parallel tests each loading ~8GB.
    static SHARED_LOADER: OnceLock<VoxtralModelLoader> = OnceLock::new();

    fn get_shared_loader() -> Option<&'static VoxtralModelLoader> {
        let path = model_path();
        if !path.exists() {
            return None;
        }
        Some(
            SHARED_LOADER.get_or_init(|| {
                VoxtralModelLoader::from_file(&path).expect("Failed to load model")
            }),
        )
    }

    #[test]
    fn test_load_full_model() {
        let Some(loader) = get_shared_loader() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        println!("Loading full Voxtral model...");
        let model = loader.load::<TestBackend>(&device).unwrap();

        // Test with small mel input
        let mel = Tensor::<TestBackend, 3>::zeros([1, 128, 320], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([1, 1, 3072], &device);

        println!("Running forward pass...");
        let logits = model.forward(mel, t_embed);

        println!("Logits shape: {:?}", logits.dims());
        // 320 mel frames -> 80 after conv -> 20 after reshape
        assert_eq!(logits.dims()[0], 1);
        assert_eq!(logits.dims()[1], 20);
        assert_eq!(logits.dims()[2], 131072); // vocab size

        println!("Full model load and forward pass successful!");
    }

    #[test]
    fn test_load_conv_downsampler() {
        let Some(loader) = get_shared_loader() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        let conv = loader
            .load_conv_downsampler::<TestBackend>(&device)
            .unwrap();

        // Test with dummy input
        let x = Tensor::<TestBackend, 3>::zeros([1, 128, 100], &device);
        let out = conv.forward(x);

        // Should be [1, 1280, 25] after 4x downsample
        assert_eq!(out.dims(), [1, 1280, 25]);
    }

    #[test]
    fn test_load_adapter() {
        let Some(loader) = get_shared_loader() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        let adapter = loader.load_adapter::<TestBackend>(&device).unwrap();

        // Test with dummy input [batch, seq, 5120] (reshaped encoder output)
        let x = Tensor::<TestBackend, 3>::zeros([1, 10, 5120], &device);
        let out = adapter.forward(x);

        // Should be [1, 10, 3072] (LLM dimension)
        assert_eq!(out.dims(), [1, 10, 3072]);
    }
}
