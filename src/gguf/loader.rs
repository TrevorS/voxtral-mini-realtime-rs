//! GGUF model loader for Q4-quantized Voxtral.
//!
//! Reads a GGUF file containing Q4_0 quantized weights and builds a
//! [`Q4VoxtralModel`]. Handles both native file I/O and in-memory bytes
//! for WASM deployment.

use anyhow::{bail, Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::module::{Param, ParamId};
use burn::nn::conv::Conv1d;
use burn::tensor::{Tensor, TensorData};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;
use tracing::info;

use crate::models::config;
use crate::models::layers::{ConvDownsampler, RmsNorm, RoPE, RoPEConfig};
use crate::models::weights::{
    adapter_weight_names, conv_weight_names, decoder_layer_weight_names,
    encoder_layer_weight_names, prefixes,
};

use super::linear::Q4Linear;
use super::model::{
    Q4AdaRmsNorm, Q4Adapter, Q4Attention, Q4AudioEncoder, Q4DecoderLayer, Q4EncoderLayer,
    Q4FeedForward, Q4LanguageModel, Q4VoxtralModel,
};
use super::reader::{GgmlDtype, GgufReader, ShardedCursor};
use super::tensor::Q4Tensor;

/// All Q4 model components with token embeddings still in raw Q4 form.
///
/// Used by [`Q4ModelLoader::load_deferred`] to allow freeing the GGUF
/// reader's memory (potentially >2 GB of shard data) before dequantizing
/// the 131K-vocab embedding table (~1.5 GiB as f32).
pub struct Q4ModelParts {
    pub encoder: Q4AudioEncoder,
    pub adapter: Q4Adapter,
    pub decoder_layers: Vec<Q4DecoderLayer>,
    pub decoder_rope: RoPE<Wgpu>,
    pub decoder_norm: RmsNorm<Wgpu>,
    pub tok_embed_q4_bytes: Vec<u8>,
    pub tok_embed_shape: [usize; 2],
}

impl Q4ModelParts {
    /// Assemble the final model with Q4 token embeddings.
    ///
    /// Keeps embeddings as Q4 on GPU (~216 MB) for the lm_head, with a CPU
    /// copy for embed_tokens row lookups. This avoids a 1.5 GiB f32 GPU
    /// buffer that would exceed WebGPU's `maxBufferSize`.
    pub fn finalize(self, device: &WgpuDevice) -> Result<Q4VoxtralModel> {
        let [vocab, d_model] = self.tok_embed_shape;

        // Create Q4Tensor on GPU for the lm_head matmul
        let tok_embed_q4 =
            Q4Tensor::from_q4_bytes(&self.tok_embed_q4_bytes, [vocab, d_model], device)?;

        let decoder = Q4LanguageModel::new_q4_embeddings(
            tok_embed_q4,
            self.tok_embed_q4_bytes,
            d_model,
            device.clone(),
            self.decoder_rope,
            self.decoder_layers,
            self.decoder_norm,
        );

        Ok(Q4VoxtralModel::new(self.encoder, decoder, self.adapter, 4))
    }
}

/// Loads a Q4-quantized Voxtral model from a GGUF file.
pub struct Q4ModelLoader<R: Read + Seek> {
    reader: GgufReader<R>,
}

impl Q4ModelLoader<BufReader<File>> {
    /// Open a GGUF file from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let reader = GgufReader::open(BufReader::new(file))?;
        Ok(Self { reader })
    }
}

impl<'a> Q4ModelLoader<Cursor<&'a [u8]>> {
    /// Open a GGUF file from in-memory bytes (for WASM).
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let reader = GgufReader::from_bytes(bytes)?;
        Ok(Self { reader })
    }
}

impl Q4ModelLoader<ShardedCursor> {
    /// Open a GGUF from multiple shards (for WASM where a single >2 GB
    /// allocation is impossible due to the 32-bit address space).
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Result<Self> {
        let reader = GgufReader::open(ShardedCursor::new(shards))?;
        Ok(Self { reader })
    }
}

impl<R: Read + Seek> Q4ModelLoader<R> {
    /// Load the complete Q4 Voxtral model.
    pub fn load(&mut self, device: &WgpuDevice) -> Result<Q4VoxtralModel> {
        info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            "Loading Q4 Voxtral model from GGUF"
        );

        info!(layers = 32, "Loading audio encoder");
        let encoder = self.load_encoder(device)?;

        info!("Loading audio-language adapter");
        let adapter = self.load_adapter(device)?;

        info!(layers = 26, "Loading language model");
        let decoder = self.load_decoder(device)?;

        info!("Q4 model loaded");

        Ok(Q4VoxtralModel::new(encoder, decoder, adapter, 4))
    }

    /// Load model components without dequantizing token embeddings.
    ///
    /// Returns [`Q4ModelParts`] containing all components plus raw Q4 bytes
    /// for the token embeddings. The caller should drop the loader (freeing
    /// GGUF shard memory) before calling [`Q4ModelParts::finalize`] to
    /// dequantize the embeddings.
    ///
    /// This two-phase approach keeps peak WASM memory under 4 GB:
    /// - Phase 1 (loader alive): shards ~2.5 GB + Q4 embed bytes ~216 MB
    /// - Phase 2 (loader dropped): Q4 embed bytes ~216 MB + f32 embed ~1.5 GiB
    pub fn load_deferred(&mut self, device: &WgpuDevice) -> Result<Q4ModelParts> {
        info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            "Loading Q4 Voxtral model from GGUF (deferred embedding)"
        );

        info!(layers = 32, "Loading audio encoder");
        let encoder = self.load_encoder(device)?;

        info!("Loading audio-language adapter");
        let adapter = self.load_adapter(device)?;

        // Extract raw Q4 bytes for token embeddings (don't dequantize yet)
        let tok_name = prefixes::TOK_EMBEDDINGS;
        let tok_info = self
            .reader
            .tensor_info(tok_name)
            .with_context(|| format!("Tensor '{tok_name}' not found"))?
            .clone();
        let tok_shape = reverse_gguf_dims(tok_info.shape());
        let tok_embed_q4_bytes = self.reader.tensor_data(tok_name)?;

        info!(layers = 26, "Loading decoder layers");
        let dec_config = config::LanguageModelConfig::default();
        let decoder_rope = RoPEConfig::new(dec_config.head_dim, 16384)
            .with_theta(dec_config.rope_theta)
            .init(device);
        let mut decoder_layers = Vec::with_capacity(dec_config.n_layers);
        for i in 0..dec_config.n_layers {
            let layer = self
                .load_decoder_layer(i, &dec_config, device)
                .with_context(|| format!("Failed to load decoder layer {i}"))?;
            decoder_layers.push(layer);
        }
        let decoder_norm = self.load_rms_norm(prefixes::FINAL_NORM, dec_config.norm_eps, device)?;

        info!("Q4 model loaded (token embeddings deferred)");

        Ok(Q4ModelParts {
            encoder,
            adapter,
            decoder_layers,
            decoder_rope,
            decoder_norm,
            tok_embed_q4_bytes,
            tok_embed_shape: [tok_shape[0], tok_shape[1]],
        })
    }

    /// Load the audio encoder.
    fn load_encoder(&mut self, device: &WgpuDevice) -> Result<Q4AudioEncoder> {
        let enc_config = config::AudioEncoderConfig::default();

        let conv = self.load_conv_downsampler(device)?;

        let rope = RoPEConfig::new(enc_config.head_dim, 4096)
            .with_theta(enc_config.rope_theta)
            .init(device);

        let mut layers = Vec::with_capacity(enc_config.n_layers);
        for i in 0..enc_config.n_layers {
            let layer = self
                .load_encoder_layer(i, &enc_config, device)
                .with_context(|| format!("Failed to load encoder layer {i}"))?;
            layers.push(layer);
        }

        let norm_name = format!("{}.transformer.norm.weight", prefixes::ENCODER);
        let norm = self.load_rms_norm(&norm_name, enc_config.norm_eps, device)?;

        Ok(Q4AudioEncoder::new(conv, rope, layers, norm))
    }

    /// Load a single encoder layer.
    fn load_encoder_layer(
        &mut self,
        layer_idx: usize,
        enc_config: &config::AudioEncoderConfig,
        device: &WgpuDevice,
    ) -> Result<Q4EncoderLayer> {
        let names = encoder_layer_weight_names(layer_idx);

        let attention_norm =
            self.load_rms_norm(&names.attention_norm, enc_config.norm_eps, device)?;

        let wq =
            self.load_q4_linear_with_optional_bias(&names.wq_weight, Some(&names.wq_bias), device)?;
        let wk = self.load_q4_linear(&names.wk_weight, device)?;
        let wv =
            self.load_q4_linear_with_optional_bias(&names.wv_weight, Some(&names.wv_bias), device)?;
        let wo =
            self.load_q4_linear_with_optional_bias(&names.wo_weight, Some(&names.wo_bias), device)?;

        let attention = Q4Attention::new(
            wq,
            wk,
            wv,
            wo,
            enc_config.n_heads,
            enc_config.n_heads, // MHA
            enc_config.head_dim,
            Some(enc_config.sliding_window),
        );

        let ffn_norm = self.load_rms_norm(&names.ffn_norm, enc_config.norm_eps, device)?;

        let w1 = self.load_q4_linear(&names.w1_weight, device)?;
        let w2 =
            self.load_q4_linear_with_optional_bias(&names.w2_weight, Some(&names.w2_bias), device)?;
        let w3 = self.load_q4_linear(&names.w3_weight, device)?;

        let ffn = Q4FeedForward::new(w1, w2, w3);

        Ok(Q4EncoderLayer::new(
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        ))
    }

    /// Load the conv downsampler (stays f32).
    fn load_conv_downsampler(&mut self, device: &WgpuDevice) -> Result<ConvDownsampler<Wgpu>> {
        let names = conv_weight_names();

        let conv1_weight: Tensor<Wgpu, 3> = self.load_f32_tensor(&names.conv1_weight, device)?;
        let conv1_bias: Tensor<Wgpu, 1> = self.load_f32_tensor(&names.conv1_bias, device)?;
        let conv2_weight: Tensor<Wgpu, 3> = self.load_f32_tensor(&names.conv2_weight, device)?;
        let conv2_bias: Tensor<Wgpu, 1> = self.load_f32_tensor(&names.conv2_bias, device)?;

        let conv1 = conv1d_from_weights(conv1_weight, Some(conv1_bias));
        let conv2 = conv1d_from_weights(conv2_weight, Some(conv2_bias));

        Ok(ConvDownsampler::new(conv1, conv2))
    }

    /// Load the language model decoder.
    fn load_decoder(&mut self, device: &WgpuDevice) -> Result<Q4LanguageModel> {
        let dec_config = config::LanguageModelConfig::default();

        // Token embeddings — Q4_0 in the GGUF, dequantize to f32 for tied lm_head
        let tok_embeddings = self.load_tok_embeddings(device)?;

        let rope = RoPEConfig::new(dec_config.head_dim, 16384)
            .with_theta(dec_config.rope_theta)
            .init(device);

        let mut layers = Vec::with_capacity(dec_config.n_layers);
        for i in 0..dec_config.n_layers {
            let layer = self
                .load_decoder_layer(i, &dec_config, device)
                .with_context(|| format!("Failed to load decoder layer {i}"))?;
            layers.push(layer);
        }

        let norm = self.load_rms_norm(prefixes::FINAL_NORM, dec_config.norm_eps, device)?;

        Ok(Q4LanguageModel::new(tok_embeddings, rope, layers, norm))
    }

    /// Load token embeddings (Q4_0 → dequantized f32).
    ///
    /// Dequantizes on CPU to avoid a synchronous GPU readback, which panics
    /// on WASM where `block_on()` is unavailable.
    fn load_tok_embeddings(&mut self, device: &WgpuDevice) -> Result<Tensor<Wgpu, 2>> {
        let name = prefixes::TOK_EMBEDDINGS;
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape = reverse_gguf_dims(info.shape());

        match info.dtype() {
            GgmlDtype::Q4_0 => {
                let bytes = self.reader.tensor_data(name)?;
                let f32_data = dequantize_q4_0_cpu(&bytes, shape[0] * shape[1]);
                let tensor_data = TensorData::new(f32_data, [shape[0], shape[1]]);
                Ok(Tensor::from_data(tensor_data, device))
            }
            GgmlDtype::F32 | GgmlDtype::F16 => self.load_f32_tensor(name, device),
            #[allow(unreachable_patterns)]
            other => bail!("Unsupported dtype {other:?} for tok_embeddings"),
        }
    }

    /// Load a single decoder layer.
    fn load_decoder_layer(
        &mut self,
        layer_idx: usize,
        dec_config: &config::LanguageModelConfig,
        device: &WgpuDevice,
    ) -> Result<Q4DecoderLayer> {
        let names = decoder_layer_weight_names(layer_idx);

        // ADA RMSNorm conditioning — Q4_0 in GGUF
        let ada_w0 = self.load_q4_linear(&names.ada_norm_down, device)?;
        let ada_w2 = self.load_q4_linear(&names.ada_norm_up, device)?;
        let ada_rms_norm = Q4AdaRmsNorm::new(ada_w0, ada_w2);

        let attention_norm =
            self.load_rms_norm(&names.attention_norm, dec_config.norm_eps, device)?;

        let wq = self.load_q4_linear(&names.wq_weight, device)?;
        let wk = self.load_q4_linear(&names.wk_weight, device)?;
        let wv = self.load_q4_linear(&names.wv_weight, device)?;
        let wo = self.load_q4_linear(&names.wo_weight, device)?;

        let attention = Q4Attention::new(
            wq,
            wk,
            wv,
            wo,
            dec_config.n_heads,
            dec_config.n_kv_heads,
            dec_config.head_dim,
            Some(dec_config.sliding_window),
        );

        let ffn_norm = self.load_rms_norm(&names.ffn_norm, dec_config.norm_eps, device)?;

        let w1 = self.load_q4_linear(&names.w1_weight, device)?;
        let w2 = self.load_q4_linear(&names.w2_weight, device)?;
        let w3 = self.load_q4_linear(&names.w3_weight, device)?;
        let ffn = Q4FeedForward::new(w1, w2, w3);

        Ok(Q4DecoderLayer::new(
            ada_rms_norm,
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        ))
    }

    /// Load the audio-language adapter.
    fn load_adapter(&mut self, device: &WgpuDevice) -> Result<Q4Adapter> {
        let names = adapter_weight_names();
        let linear1 = self.load_q4_linear(&names.linear1_weight, device)?;
        let linear2 = self.load_q4_linear(&names.linear2_weight, device)?;
        Ok(Q4Adapter::new(linear1, linear2))
    }

    // -----------------------------------------------------------------------
    // Primitive loading helpers
    // -----------------------------------------------------------------------

    /// Load a Q4_0 tensor as a [`Q4Linear`] (no bias).
    fn load_q4_linear(&mut self, name: &str, device: &WgpuDevice) -> Result<Q4Linear> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(Q4Linear::new(q4, None))
    }

    /// Load a Q4_0 tensor with an optional F32 bias as a [`Q4Linear`].
    fn load_q4_linear_with_optional_bias(
        &mut self,
        weight_name: &str,
        bias_name: Option<&str>,
        device: &WgpuDevice,
    ) -> Result<Q4Linear> {
        let info = self
            .reader
            .tensor_info(weight_name)
            .with_context(|| format!("Tensor '{weight_name}' not found"))?
            .clone();

        if info.dtype() != GgmlDtype::Q4_0 {
            bail!("Expected Q4_0 for '{weight_name}', got {:?}", info.dtype());
        }

        let shape = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(weight_name)?;
        let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;

        let bias = if let Some(bias_name) = bias_name {
            if self.reader.tensor_info(bias_name).is_some() {
                let bias_tensor: Tensor<Wgpu, 1> = self.load_f32_tensor(bias_name, device)?;
                Some(bias_tensor)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Q4Linear::new(q4, bias))
    }

    /// Load an F32/F16 tensor from GGUF.
    fn load_f32_tensor<const D: usize>(
        &mut self,
        name: &str,
        device: &WgpuDevice,
    ) -> Result<Tensor<Wgpu, D>> {
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape: Vec<usize> = reverse_gguf_dims(info.shape());
        let bytes = self.reader.tensor_data(name)?;

        let data: Vec<f32> = match info.dtype() {
            GgmlDtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            GgmlDtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgmlDtype::Q4_0 => bail!("Cannot load Q4_0 tensor '{name}' as f32; use load_q4_linear"),
        };

        let tensor_data = TensorData::new(data, shape);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Load an RmsNorm layer from GGUF.
    fn load_rms_norm(
        &mut self,
        name: &str,
        eps: f64,
        device: &WgpuDevice,
    ) -> Result<RmsNorm<Wgpu>> {
        let weight: Tensor<Wgpu, 1> = self.load_f32_tensor(name, device)?;
        Ok(RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), weight),
                epsilon: eps,
            },
        })
    }
}

/// Reverse GGUF dimension order to get PyTorch convention.
///
/// GGUF stores dimensions in reversed order (row-major innermost first),
/// while PyTorch uses `[out_features, in_features]` convention.
fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

/// Dequantize Q4_0 blocks on CPU, returning `num_elements` f32 values.
///
/// Same logic as [`Q4Tensor::dequantize`] but operates on raw bytes without
/// a GPU round-trip, making it safe on WASM.
fn dequantize_q4_0_cpu(raw: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut output = vec![0.0f32; num_elements];
    for block_idx in 0..num_blocks {
        let offset = block_idx * 18;
        let d = half::f16::from_bits(u16::from_le_bytes([raw[offset], raw[offset + 1]])).to_f32();
        let base = block_idx * 32;
        for i in 0..16 {
            let byte = raw[offset + 2 + i];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output[base + i] = lo * d;
            output[base + i + 16] = hi * d;
        }
    }
    output
}

/// Create a `Conv1d` from weight tensors (matches existing `loader.rs` helper).
fn conv1d_from_weights(weight: Tensor<Wgpu, 3>, bias: Option<Tensor<Wgpu, 1>>) -> Conv1d<Wgpu> {
    use burn::module::Ignored;

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
    use std::path::PathBuf;

    fn gguf_path() -> PathBuf {
        PathBuf::from("models/voxtral-q4.gguf")
    }

    #[test]
    fn test_load_q4_model() {
        let path = gguf_path();
        if !path.exists() {
            println!("Skipping: GGUF model not found at {}", path.display());
            return;
        }

        let device = WgpuDevice::default();
        let mut loader = Q4ModelLoader::from_file(&path).unwrap();
        let model = loader.load(&device).unwrap();

        // Verify layer counts
        assert_eq!(model.encoder().n_layers(), 32);
        assert_eq!(model.decoder().n_layers(), 26);
        assert_eq!(model.decoder().d_model(), 3072);

        println!("Q4 model loaded successfully from GGUF!");
    }

    #[test]
    fn test_q4_forward_shape() {
        let path = gguf_path();
        if !path.exists() {
            println!("Skipping: GGUF model not found at {}", path.display());
            return;
        }

        let device = WgpuDevice::default();
        let mut loader = Q4ModelLoader::from_file(&path).unwrap();
        let model = loader.load(&device).unwrap();

        // Small mel input: [1, 128, 320]
        let mel = Tensor::<Wgpu, 3>::zeros([1, 128, 320], &device);
        let t_embed = Tensor::<Wgpu, 3>::zeros([1, 1, 3072], &device);

        let logits = model.forward(mel, t_embed);

        // 320 mel frames → 80 after conv → 20 after reshape(4)
        assert_eq!(logits.dims()[0], 1);
        assert_eq!(logits.dims()[1], 20);
        assert_eq!(logits.dims()[2], 131072);

        println!("Q4 forward pass shape: {:?}", logits.dims());
    }
}
