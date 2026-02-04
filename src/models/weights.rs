//! Weight loading from SafeTensors files.
//!
//! Loads pre-trained Voxtral weights into Burn model modules.

use anyhow::{Context, Result};
use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Load a tensor from SafeTensors by name.
pub fn load_tensor<B: Backend, const D: usize>(
    safetensors: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>> {
    let tensor_view = safetensors
        .tensor(name)
        .with_context(|| format!("Tensor '{}' not found", name))?;

    let shape: Vec<usize> = tensor_view.shape().to_vec();
    let dtype = tensor_view.dtype();

    // Convert to f32
    let data: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => {
            let bytes = tensor_view.data();
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            let bytes = tensor_view.data();
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            let bytes = tensor_view.data();
            bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()
        }
        _ => anyhow::bail!("Unsupported dtype: {:?}", dtype),
    };

    // Create tensor data with shape
    let tensor_data = TensorData::new(data, shape);

    // Create Burn tensor
    let tensor: Tensor<B, D> = Tensor::from_data(tensor_data, device);
    Ok(tensor)
}

/// Owning wrapper for SafeTensors that keeps bytes alive without leaking.
///
/// This struct owns the raw bytes and provides safe access to the SafeTensors
/// view. The bytes are freed when this struct is dropped.
pub struct OwnedSafeTensors {
    // Arc allows sharing the bytes across clones without copying
    _bytes: Arc<Vec<u8>>,
    // SAFETY: safetensors borrows from _bytes which we keep alive
    // We use 'static here but the actual lifetime is tied to _bytes
    safetensors: SafeTensors<'static>,
}

impl OwnedSafeTensors {
    /// Load SafeTensors from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = fs::read(path.as_ref())
            .with_context(|| format!("Failed to read file: {}", path.as_ref().display()))?;

        Self::from_bytes(bytes)
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let bytes = Arc::new(bytes);

        // SAFETY: We're creating a SafeTensors that borrows from `bytes`.
        // We store both in the same struct, and _bytes is never moved or dropped
        // while safetensors exists. The Arc ensures the bytes live long enough.
        let safetensors = unsafe {
            let bytes_ref: &[u8] = &bytes;
            // Transmute the lifetime to 'static - safe because we control the lifetime
            // via the Arc in _bytes
            let static_ref: &'static [u8] = std::mem::transmute(bytes_ref);
            SafeTensors::deserialize(static_ref).context("Failed to deserialize SafeTensors")?
        };

        Ok(Self {
            _bytes: bytes,
            safetensors,
        })
    }

    /// Get a reference to the SafeTensors.
    pub fn tensors(&self) -> &SafeTensors<'_> {
        &self.safetensors
    }
}

// Implement Deref for convenient access
impl std::ops::Deref for OwnedSafeTensors {
    type Target = SafeTensors<'static>;

    fn deref(&self) -> &Self::Target {
        &self.safetensors
    }
}

/// Load SafeTensors file from disk (returns owning wrapper).
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<OwnedSafeTensors> {
    OwnedSafeTensors::from_file(path)
}

/// Weight name prefixes for different model components.
pub mod prefixes {
    /// Audio encoder prefix.
    pub const ENCODER: &str = "mm_streams_embeddings.embedding_module.whisper_encoder";
    /// LLM decoder prefix.
    pub const DECODER: &str = "layers";
    /// Token embeddings prefix.
    pub const TOK_EMBEDDINGS: &str = "mm_streams_embeddings.embedding_module.tok_embeddings.weight";
    /// Adapter prefix.
    pub const ADAPTER: &str = "mm_streams_embeddings.embedding_module.audio_language_projection";
    /// Final norm prefix.
    pub const FINAL_NORM: &str = "norm.weight";
}

/// List all tensor names in a SafeTensors file.
pub fn list_tensors<'a>(safetensors: &'a SafeTensors<'a>) -> Vec<&'a str> {
    safetensors.names().into_iter().collect()
}

/// Filter tensor names by prefix.
pub fn filter_tensors<'a>(safetensors: &'a SafeTensors<'a>, prefix: &str) -> Vec<&'a str> {
    safetensors
        .names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .collect()
}

/// Create a Linear layer from weight tensors.
///
/// Note: PyTorch stores Linear weights as [out_features, in_features],
/// but Burn expects [in_features, out_features]. This function handles
/// the transpose automatically.
pub fn linear_from_weights<B: Backend>(
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Linear<B> {
    // PyTorch Linear: weight [out, in], Burn Linear: weight [in, out]
    // Need to transpose
    let weight = weight.transpose();

    Linear {
        weight: Param::initialized(ParamId::new(), weight),
        bias: bias.map(|b| Param::initialized(ParamId::new(), b)),
    }
}

/// Load a Linear layer from SafeTensors.
///
/// # Arguments
/// * `safetensors` - SafeTensors file
/// * `weight_name` - Name of the weight tensor
/// * `bias_name` - Optional name of the bias tensor
/// * `device` - Device to load tensors on
pub fn load_linear<B: Backend>(
    safetensors: &SafeTensors,
    weight_name: &str,
    bias_name: Option<&str>,
    device: &B::Device,
) -> Result<Linear<B>> {
    let weight: Tensor<B, 2> = load_tensor(safetensors, weight_name, device)?;
    let bias = if let Some(name) = bias_name {
        // Check if bias exists
        if safetensors.tensor(name).is_ok() {
            Some(load_tensor::<B, 1>(safetensors, name, device)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(linear_from_weights(weight, bias))
}

/// Encoder layer weight names.
pub fn encoder_layer_weight_names(layer_idx: usize) -> EncoderLayerWeightNames {
    let prefix = format!("{}.transformer.layers.{}", prefixes::ENCODER, layer_idx);

    EncoderLayerWeightNames {
        attention_norm: format!("{}.attention_norm.weight", prefix),
        wq_weight: format!("{}.attention.wq.weight", prefix),
        wq_bias: format!("{}.attention.wq.bias", prefix),
        wk_weight: format!("{}.attention.wk.weight", prefix),
        wv_weight: format!("{}.attention.wv.weight", prefix),
        wv_bias: format!("{}.attention.wv.bias", prefix),
        wo_weight: format!("{}.attention.wo.weight", prefix),
        wo_bias: format!("{}.attention.wo.bias", prefix),
        ffn_norm: format!("{}.ffn_norm.weight", prefix),
        w1_weight: format!("{}.feed_forward.w1.weight", prefix),
        w2_weight: format!("{}.feed_forward.w2.weight", prefix),
        w2_bias: format!("{}.feed_forward.w2.bias", prefix),
        w3_weight: format!("{}.feed_forward.w3.weight", prefix),
    }
}

/// Weight names for an encoder layer.
pub struct EncoderLayerWeightNames {
    pub attention_norm: String,
    pub wq_weight: String,
    pub wq_bias: String,
    pub wk_weight: String,
    pub wv_weight: String,
    pub wv_bias: String,
    pub wo_weight: String,
    pub wo_bias: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String,
    pub w2_bias: String,
    pub w3_weight: String,
}

/// Decoder layer weight names.
pub fn decoder_layer_weight_names(layer_idx: usize) -> DecoderLayerWeightNames {
    let prefix = format!("{}.{}", prefixes::DECODER, layer_idx);

    DecoderLayerWeightNames {
        // ADA RMSNorm conditioning (t-embed projection)
        ada_norm_down: format!("{}.ada_rms_norm_t_cond.0.weight", prefix),
        ada_norm_up: format!("{}.ada_rms_norm_t_cond.2.weight", prefix),
        attention_norm: format!("{}.attention_norm.weight", prefix),
        wq_weight: format!("{}.attention.wq.weight", prefix),
        wk_weight: format!("{}.attention.wk.weight", prefix),
        wv_weight: format!("{}.attention.wv.weight", prefix),
        wo_weight: format!("{}.attention.wo.weight", prefix),
        ffn_norm: format!("{}.ffn_norm.weight", prefix),
        w1_weight: format!("{}.feed_forward.w1.weight", prefix),
        w2_weight: format!("{}.feed_forward.w2.weight", prefix),
        w3_weight: format!("{}.feed_forward.w3.weight", prefix),
    }
}

/// Weight names for a decoder layer.
pub struct DecoderLayerWeightNames {
    pub ada_norm_down: String,
    pub ada_norm_up: String,
    pub attention_norm: String,
    pub wq_weight: String,
    pub wk_weight: String,
    pub wv_weight: String,
    pub wo_weight: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String,
    pub w3_weight: String,
}

/// Conv downsampler weight names.
pub fn conv_weight_names() -> ConvWeightNames {
    ConvWeightNames {
        conv1_weight: format!("{}.conv_layers.0.conv.weight", prefixes::ENCODER),
        conv1_bias: format!("{}.conv_layers.0.conv.bias", prefixes::ENCODER),
        conv2_weight: format!("{}.conv_layers.1.conv.weight", prefixes::ENCODER),
        conv2_bias: format!("{}.conv_layers.1.conv.bias", prefixes::ENCODER),
    }
}

/// Weight names for the conv downsampler.
pub struct ConvWeightNames {
    pub conv1_weight: String,
    pub conv1_bias: String,
    pub conv2_weight: String,
    pub conv2_bias: String,
}

/// Adapter weight names.
pub fn adapter_weight_names() -> AdapterWeightNames {
    AdapterWeightNames {
        linear1_weight: format!("{}.0.weight", prefixes::ADAPTER),
        linear2_weight: format!("{}.2.weight", prefixes::ADAPTER),
    }
}

/// Weight names for the audio-language adapter.
pub struct AdapterWeightNames {
    pub linear1_weight: String,
    pub linear2_weight: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    fn model_path() -> PathBuf {
        PathBuf::from("models/voxtral/consolidated.safetensors")
    }

    /// Shared SafeTensors loader for tests - loads model once and reuses.
    /// This prevents OOM from parallel tests each loading ~8GB.
    static SHARED_SAFETENSORS: OnceLock<OwnedSafeTensors> = OnceLock::new();

    fn get_shared_safetensors() -> Option<&'static OwnedSafeTensors> {
        let path = model_path();
        if !path.exists() {
            return None;
        }
        Some(
            SHARED_SAFETENSORS.get_or_init(|| {
                load_safetensors(&path).expect("Failed to load shared safetensors")
            }),
        )
    }

    #[test]
    fn test_load_safetensors_exists() {
        let Some(safetensors) = get_shared_safetensors() else {
            println!("Skipping: model not downloaded. Run: ./scripts/download_model.py");
            return;
        };

        let names = list_tensors(safetensors);

        println!("Found {} tensors", names.len());
        assert!(!names.is_empty(), "Should have tensors");

        // Check expected tensor names
        assert!(names.contains(&"norm.weight"), "Should have final norm");
    }

    #[test]
    fn test_filter_encoder_tensors() {
        let Some(safetensors) = get_shared_safetensors() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let encoder_tensors = filter_tensors(safetensors, prefixes::ENCODER);

        println!("Found {} encoder tensors", encoder_tensors.len());
        assert!(!encoder_tensors.is_empty(), "Should have encoder tensors");
    }

    #[test]
    fn test_load_tensor() {
        use burn::backend::Wgpu;

        let Some(safetensors) = get_shared_safetensors() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        // Load final norm weight
        let norm_weight: Tensor<Wgpu, 1> =
            load_tensor(safetensors, prefixes::FINAL_NORM, &device).unwrap();

        println!("Norm weight shape: {:?}", norm_weight.dims());
        assert_eq!(norm_weight.dims(), [3072], "Should be [3072]");
    }

    #[test]
    fn test_load_encoder_attention_weight() {
        use burn::backend::Wgpu;

        let Some(safetensors) = get_shared_safetensors() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        // Load encoder layer 0 attention wq weight
        let names = encoder_layer_weight_names(0);
        let wq: Tensor<Wgpu, 2> = load_tensor(safetensors, &names.wq_weight, &device).unwrap();

        println!("Encoder wq shape: {:?}", wq.dims());
        // Should be [n_heads * head_dim, d_model] = [32 * 64, 1280] = [2048, 1280]
        assert_eq!(wq.dims(), [2048, 1280]);
    }

    #[test]
    fn test_load_linear() {
        use burn::backend::Wgpu;

        let Some(safetensors) = get_shared_safetensors() else {
            println!("Skipping: model not downloaded");
            return;
        };

        let device = Default::default();

        // Load adapter linear1
        let names = adapter_weight_names();
        let linear: Linear<Wgpu> =
            load_linear(safetensors, &names.linear1_weight, None, &device).unwrap();

        // Adapter linear1: [5120, 3072] in PyTorch -> [3072, 5120] after transpose
        // Actually, looking at the docs, it's [out, in] -> [in, out]
        // So if PyTorch has [3072, 5120], Burn needs [5120, 3072]
        let dims = linear.weight.dims();
        println!("Adapter linear1 weight shape: {:?}", dims);
        // PyTorch [3072, 5120] -> Burn [5120, 3072]
        assert_eq!(dims[0], 5120, "d_input");
        assert_eq!(dims[1], 3072, "d_output");
    }

    #[test]
    fn test_encoder_layer_weight_names() {
        let names = encoder_layer_weight_names(5);
        assert!(names.wq_weight.contains(".5."));
        assert!(names.wq_weight.ends_with("attention.wq.weight"));
    }

    #[test]
    fn test_decoder_layer_weight_names() {
        let names = decoder_layer_weight_names(10);
        assert!(names.wq_weight.contains(".10."));
        assert!(names.wq_weight.ends_with("attention.wq.weight"));
    }
}
