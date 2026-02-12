//! Q4_0 quantized model structs for Voxtral.
//!
//! Mirrors the f32 model in `src/models/` but uses [`Q4Linear`] for all
//! weight-heavy layers (attention projections, FFN, adapter). Non-linear
//! ops (RMSNorm, RoPE, softmax, GELU, convolution, attention masking)
//! stay as regular Burn f32 tensors/ops.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::{Int, Tensor, TensorData};

use crate::models::adapter::reshape_encoder_output;
use crate::models::layers::masking::{
    apply_causal_mask, apply_causal_mask_with_offset, apply_sliding_window_mask,
    apply_sliding_window_mask_with_offset,
};
use crate::models::layers::{ConvDownsampler, KVCache, LayerCaches, RmsNorm, RoPE};

use super::linear::Q4Linear;

// ---------------------------------------------------------------------------
// Q4Attention
// ---------------------------------------------------------------------------

/// Multi-head attention with Q4-quantized weight projections.
///
/// Supports both MHA (encoder) and GQA (decoder) configurations.
/// Q/K/V/O projections use [`Q4Linear`]; attention score computation
/// uses regular Burn matmuls (activation × activation).
pub struct Q4Attention {
    wq: Q4Linear,
    wk: Q4Linear,
    wv: Q4Linear,
    wo: Q4Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
}

impl Q4Attention {
    /// Create a new Q4 attention layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Q4Linear,
        wk: Q4Linear,
        wv: Q4Linear,
        wo: Q4Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window,
        }
    }

    /// Forward pass with RoPE.
    ///
    /// # Arguments
    /// * `x` - Input tensor `[batch, seq, d_model]`
    /// * `rope` - Rotary position embeddings
    /// * `offset` - Position offset for KV cache
    /// * `causal` - Whether to apply causal masking
    pub fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        offset: usize,
        causal: bool,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = self.expand_kv(k, v);

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let scores = if causal {
            apply_causal_mask(scores, seq_len)
        } else {
            scores
        };
        let scores = if let Some(window) = self.sliding_window {
            apply_sliding_window_mask(scores, seq_len, window)
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.wo.forward(out)
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        cache: &mut KVCache<Wgpu>,
        causal: bool,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.seq_len();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        let (k, v) = self.expand_kv(k, v);

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let scores = if causal {
            apply_causal_mask_with_offset(scores, seq_len, total_seq_len, offset)
        } else {
            scores
        };
        let scores = if let Some(window) = self.sliding_window {
            apply_sliding_window_mask_with_offset(scores, seq_len, total_seq_len, window, offset)
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.wo.forward(out)
    }

    /// Expand K, V heads for GQA.
    fn expand_kv(
        &self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        if self.n_heads == self.n_kv_heads {
            return (k, v);
        }
        let repeat_factor = self.n_heads / self.n_kv_heads;
        let [batch, n_kv_heads, seq, head_dim] = k.dims();

        let k = k
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        (k, v)
    }
}

// ---------------------------------------------------------------------------
// Q4FeedForward (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU MLP with Q4-quantized weights.
///
/// Computes `w2(silu(w1(x)) * w3(x))`.
pub struct Q4FeedForward {
    w1: Q4Linear,
    w2: Q4Linear,
    w3: Q4Linear,
}

impl Q4FeedForward {
    /// Create a new Q4 feed-forward layer.
    pub fn new(w1: Q4Linear, w2: Q4Linear, w3: Q4Linear) -> Self {
        Self { w1, w2, w3 }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = silu(self.w1.forward(x.clone()));
        let up = self.w3.forward(x);
        self.w2.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// Q4AdaRmsNorm
// ---------------------------------------------------------------------------

/// Adaptive modulation with Q4-quantized projections.
///
/// Computes `x * (1 + w2(gelu(w0(t_embed))))`.
pub struct Q4AdaRmsNorm {
    w0: Q4Linear,
    w2: Q4Linear,
}

impl Q4AdaRmsNorm {
    /// Create a new Q4 ADA RMSNorm layer.
    pub fn new(w0: Q4Linear, w2: Q4Linear) -> Self {
        Self { w0, w2 }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor `[batch, seq, d_model]`
    /// * `t_embed` - Temporal embedding `[batch, 1, d_model]`
    pub fn forward(&self, x: Tensor<Wgpu, 3>, t_embed: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let scale = self.w0.forward(t_embed);
        let scale = gelu(scale);
        let scale = self.w2.forward(scale);
        x * (scale + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Q4EncoderLayer
// ---------------------------------------------------------------------------

/// Audio encoder transformer layer with Q4-quantized weights.
pub struct Q4EncoderLayer {
    attention_norm: RmsNorm<Wgpu>,
    attention: Q4Attention,
    ffn_norm: RmsNorm<Wgpu>,
    ffn: Q4FeedForward,
}

impl Q4EncoderLayer {
    /// Create a new Q4 encoder layer.
    pub fn new(
        attention_norm: RmsNorm<Wgpu>,
        attention: Q4Attention,
        ffn_norm: RmsNorm<Wgpu>,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<Wgpu, 3>, rope: &RoPE<Wgpu>, offset: usize) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        cache: &mut KVCache<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Q4DecoderLayer
// ---------------------------------------------------------------------------

/// Decoder transformer layer with Q4-quantized weights and ADA modulation.
pub struct Q4DecoderLayer {
    ada_rms_norm: Q4AdaRmsNorm,
    attention_norm: RmsNorm<Wgpu>,
    attention: Q4Attention,
    ffn_norm: RmsNorm<Wgpu>,
    ffn: Q4FeedForward,
}

impl Q4DecoderLayer {
    /// Create a new Q4 decoder layer.
    pub fn new(
        ada_rms_norm: Q4AdaRmsNorm,
        attention_norm: RmsNorm<Wgpu>,
        attention: Q4Attention,
        ffn_norm: RmsNorm<Wgpu>,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            ada_rms_norm,
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    /// Forward pass.
    pub fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        t_embed: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ada_rms_norm.forward(x, t_embed);
        let x = self.ffn.forward(x);
        x + residual
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        t_embed: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        cache: &mut KVCache<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ada_rms_norm.forward(x, t_embed);
        let x = self.ffn.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Q4AudioEncoder
// ---------------------------------------------------------------------------

/// Audio encoder with Q4-quantized transformer layers.
///
/// Conv downsampler stays f32 (small: ~1 MB).
pub struct Q4AudioEncoder {
    conv: ConvDownsampler<Wgpu>,
    rope: RoPE<Wgpu>,
    layers: Vec<Q4EncoderLayer>,
    norm: RmsNorm<Wgpu>,
}

impl Q4AudioEncoder {
    /// Create a new Q4 audio encoder.
    pub fn new(
        conv: ConvDownsampler<Wgpu>,
        rope: RoPE<Wgpu>,
        layers: Vec<Q4EncoderLayer>,
        norm: RmsNorm<Wgpu>,
    ) -> Self {
        Self {
            conv,
            rope,
            layers,
            norm,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram `[batch, n_mels, time]`
    /// * `offset` - Position offset for KV cache
    pub fn forward(&self, mel: Tensor<Wgpu, 3>, offset: usize) -> Tensor<Wgpu, 3> {
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, offset);
        }
        self.norm.forward(x)
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        mel: Tensor<Wgpu, 3>,
        caches: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);

        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// Get the number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Create a new KV cache for this encoder.
    pub fn create_cache(&self) -> LayerCaches<Wgpu> {
        LayerCaches::new(self.layers.len())
    }
}

// ---------------------------------------------------------------------------
// Q4LanguageModel
// ---------------------------------------------------------------------------

/// How the token embedding table is stored.
///
/// - **F32**: dequantized at load time. Used on native where a 1.5 GiB GPU
///   buffer is fine.
/// - **Q4**: kept as Q4_0 on GPU (lm_head via Q4 matmul) with a CPU byte
///   copy for embed_tokens row lookups. Used on WASM where a single GPU
///   buffer > ~256 MB is rejected by WebGPU.
enum TokEmbedStore {
    F32(Tensor<Wgpu, 2>),
    Q4 {
        lm_head: Q4Linear,
        cpu_bytes: Vec<u8>,
    },
}

/// Language model decoder with Q4-quantized transformer layers.
pub struct Q4LanguageModel {
    tok_embeddings: TokEmbedStore,
    rope: RoPE<Wgpu>,
    layers: Vec<Q4DecoderLayer>,
    norm: RmsNorm<Wgpu>,
    d_model: usize,
    device: WgpuDevice,
}

impl Q4LanguageModel {
    /// Create a new Q4 language model with f32 token embeddings.
    pub fn new(
        tok_embeddings: Tensor<Wgpu, 2>,
        rope: RoPE<Wgpu>,
        layers: Vec<Q4DecoderLayer>,
        norm: RmsNorm<Wgpu>,
    ) -> Self {
        let d_model = tok_embeddings.dims()[1];
        let device = tok_embeddings.device();
        Self {
            tok_embeddings: TokEmbedStore::F32(tok_embeddings),
            rope,
            layers,
            norm,
            d_model,
            device,
        }
    }

    /// Create a new Q4 language model with Q4 token embeddings.
    ///
    /// Keeps a CPU copy of the Q4 bytes for embed_tokens (small row lookups)
    /// and a Q4Linear on GPU for the lm_head (full vocab matmul).
    #[allow(clippy::too_many_arguments)]
    pub fn new_q4_embeddings(
        tok_embed_q4: super::tensor::Q4Tensor,
        tok_embed_bytes: Vec<u8>,
        d_model: usize,
        device: WgpuDevice,
        rope: RoPE<Wgpu>,
        layers: Vec<Q4DecoderLayer>,
        norm: RmsNorm<Wgpu>,
    ) -> Self {
        Self {
            tok_embeddings: TokEmbedStore::Q4 {
                lm_head: Q4Linear::new(tok_embed_q4, None),
                cpu_bytes: tok_embed_bytes,
            },
            rope,
            layers,
            norm,
            d_model,
            device,
        }
    }

    /// Embed token IDs to dense vectors.
    ///
    /// On the Q4 path, this reads token IDs back from the GPU synchronously,
    /// which panics on WASM. Use [`embed_tokens_from_ids`](Self::embed_tokens_from_ids)
    /// when the IDs are known on the CPU.
    pub fn embed_tokens(&self, token_ids: Tensor<Wgpu, 2, Int>) -> Tensor<Wgpu, 3> {
        match &self.tok_embeddings {
            TokEmbedStore::F32(embed) => {
                let [batch, seq] = token_ids.dims();
                let flat_ids = token_ids.reshape([batch * seq]);
                let selected = embed.clone().select(0, flat_ids);
                selected.reshape([batch, seq, self.d_model])
            }
            TokEmbedStore::Q4 { cpu_bytes, .. } => {
                let [batch, seq] = token_ids.dims();
                let id_data = token_ids.into_data();
                let ids: Vec<i32> = id_data
                    .to_vec()
                    .expect("tensor data extraction for token IDs");
                self.embed_from_q4_bytes(cpu_bytes, &ids, batch, seq)
            }
        }
    }

    /// Embed token IDs from a CPU slice — avoids GPU readback (safe on WASM).
    pub fn embed_tokens_from_ids(&self, ids: &[i32], batch: usize, seq: usize) -> Tensor<Wgpu, 3> {
        match &self.tok_embeddings {
            TokEmbedStore::F32(embed) => {
                let id_tensor = Tensor::<Wgpu, 2, Int>::from_data(
                    TensorData::new(ids.to_vec(), [batch, seq]),
                    &self.device,
                );
                let flat_ids = id_tensor.reshape([batch * seq]);
                let selected = embed.clone().select(0, flat_ids);
                selected.reshape([batch, seq, self.d_model])
            }
            TokEmbedStore::Q4 { cpu_bytes, .. } => {
                self.embed_from_q4_bytes(cpu_bytes, ids, batch, seq)
            }
        }
    }

    /// Dequantize specific rows from CPU Q4 bytes.
    fn embed_from_q4_bytes(
        &self,
        cpu_bytes: &[u8],
        ids: &[i32],
        batch: usize,
        seq: usize,
    ) -> Tensor<Wgpu, 3> {
        let blocks_per_row = self.d_model / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut output = vec![0.0f32; ids.len() * self.d_model];

        for (i, &id) in ids.iter().enumerate() {
            let row_offset = (id as usize) * bytes_per_row;
            let row_bytes = &cpu_bytes[row_offset..row_offset + bytes_per_row];
            let out_slice = &mut output[i * self.d_model..(i + 1) * self.d_model];

            for block in 0..blocks_per_row {
                let bo = block * 18;
                let d =
                    half::f16::from_bits(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]))
                        .to_f32();
                let base = block * 32;
                for j in 0..16 {
                    let byte = row_bytes[bo + 2 + j];
                    out_slice[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
                    out_slice[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
                }
            }
        }

        Tensor::from_data(
            TensorData::new(output, [batch, seq, self.d_model]),
            &self.device,
        )
    }

    /// Forward pass returning hidden states (before LM head).
    pub fn forward(
        &self,
        token_ids: Tensor<Wgpu, 2, Int>,
        t_embed: Tensor<Wgpu, 3>,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let x = self.embed_tokens(token_ids);
        self.forward_hidden_inner(x, t_embed, offset)
    }

    /// Forward pass with hidden states input (for multimodal).
    pub fn forward_hidden(
        &self,
        hidden_states: Tensor<Wgpu, 3>,
        t_embed: Tensor<Wgpu, 3>,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        self.forward_hidden_inner(hidden_states, t_embed, offset)
    }

    fn forward_hidden_inner(
        &self,
        mut x: Tensor<Wgpu, 3>,
        t_embed: Tensor<Wgpu, 3>,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        for layer in &self.layers {
            x = layer.forward(x, t_embed.clone(), &self.rope, offset);
        }
        self.norm.forward(x)
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        token_ids: Tensor<Wgpu, 2, Int>,
        t_embed: Tensor<Wgpu, 3>,
        caches: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let x = self.embed_tokens(token_ids);
        self.forward_hidden_with_cache(x, t_embed, caches)
    }

    /// Forward pass with hidden states input and KV cache.
    pub fn forward_hidden_with_cache(
        &self,
        mut x: Tensor<Wgpu, 3>,
        t_embed: Tensor<Wgpu, 3>,
        caches: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, t_embed.clone(), &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// Compute logits from hidden states (LM head with tied embeddings).
    pub fn lm_head(&self, hidden_states: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        match &self.tok_embeddings {
            TokEmbedStore::F32(embed) => {
                let [batch, seq, _] = hidden_states.dims();
                let vocab_size = embed.dims()[0];
                let embed_t = embed.clone().transpose().unsqueeze::<3>();
                let logits = hidden_states.matmul(embed_t);
                logits.reshape([batch, seq, vocab_size])
            }
            TokEmbedStore::Q4 { lm_head, .. } => lm_head.forward(hidden_states),
        }
    }

    /// Get the number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Create a new KV cache for this decoder.
    pub fn create_cache(&self) -> LayerCaches<Wgpu> {
        LayerCaches::new(self.layers.len())
    }

    /// Create a pre-allocated KV cache sized for the given max sequence length.
    ///
    /// Avoids per-step GPU allocations by writing into fixed buffers.
    pub fn create_cache_preallocated(&self, max_seq: usize) -> LayerCaches<Wgpu> {
        // Decoder uses GQA: 8 KV heads, head_dim = d_model / n_heads = 3072 / 32 = 96
        let n_kv_heads = self.layers.first().map_or(8, |l| l.attention.n_kv_heads);
        let head_dim = self.layers.first().map_or(96, |l| l.attention.head_dim);
        LayerCaches::new_preallocated(
            self.layers.len(),
            1, // batch = 1 for streaming
            n_kv_heads,
            max_seq,
            head_dim,
            &self.device,
        )
    }
}

// ---------------------------------------------------------------------------
// Q4Adapter
// ---------------------------------------------------------------------------

/// Audio-language adapter with Q4-quantized projections.
///
/// Two-layer MLP: `Linear(5120→3072) → GELU → Linear(3072→3072)`.
pub struct Q4Adapter {
    linear1: Q4Linear,
    linear2: Q4Linear,
}

impl Q4Adapter {
    /// Create a new Q4 adapter.
    pub fn new(linear1: Q4Linear, linear2: Q4Linear) -> Self {
        Self { linear1, linear2 }
    }

    /// Forward pass.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let x = self.linear1.forward(x);
        let x = gelu(x);
        self.linear2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Q4VoxtralModel
// ---------------------------------------------------------------------------

/// Complete Voxtral model with Q4-quantized weights.
///
/// Combines Q4 audio encoder, adapter, and language model for streaming ASR.
pub struct Q4VoxtralModel {
    encoder: Q4AudioEncoder,
    decoder: Q4LanguageModel,
    adapter: Q4Adapter,
    reshape_factor: usize,
}

impl Q4VoxtralModel {
    /// Create a new Q4 Voxtral model.
    pub fn new(
        encoder: Q4AudioEncoder,
        decoder: Q4LanguageModel,
        adapter: Q4Adapter,
        reshape_factor: usize,
    ) -> Self {
        Self {
            encoder,
            decoder,
            adapter,
            reshape_factor,
        }
    }

    /// Encode audio to hidden states ready for the LLM.
    pub fn encode_audio(&self, mel: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let _span = tracing::info_span!("encode_audio").entered();
        let encoder_out = self.encoder.forward(mel, 0);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Encode audio with KV cache.
    pub fn encode_audio_with_cache(
        &self,
        mel: Tensor<Wgpu, 3>,
        encoder_cache: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let encoder_out = self.encoder.forward_with_cache(mel, encoder_cache);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Full forward pass from mel to logits (streaming transcription mode).
    pub fn forward_streaming(
        &self,
        mel: Tensor<Wgpu, 3>,
        token_ids: Tensor<Wgpu, 2, Int>,
        t_embed_decoder: Tensor<Wgpu, 3>,
    ) -> Tensor<Wgpu, 3> {
        let audio_embeds = self.encode_audio(mel);
        let text_embeds = self.decoder.embed_tokens(token_ids);
        let inputs_embeds = audio_embeds + text_embeds;
        let hidden = self
            .decoder
            .forward_hidden(inputs_embeds, t_embed_decoder, 0);
        self.decoder.lm_head(hidden)
    }

    /// Full forward pass from mel to logits (without text tokens).
    pub fn forward(
        &self,
        mel: Tensor<Wgpu, 3>,
        t_embed_decoder: Tensor<Wgpu, 3>,
    ) -> Tensor<Wgpu, 3> {
        let audio_hidden = self.encode_audio(mel);
        let hidden = self
            .decoder
            .forward_hidden(audio_hidden, t_embed_decoder, 0);
        self.decoder.lm_head(hidden)
    }

    /// Full forward pass with KV caches.
    pub fn forward_with_cache(
        &self,
        mel: Tensor<Wgpu, 3>,
        t_embed_decoder: Tensor<Wgpu, 3>,
        encoder_cache: &mut LayerCaches<Wgpu>,
        decoder_cache: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let audio_hidden = self.encode_audio_with_cache(mel, encoder_cache);
        let hidden =
            self.decoder
                .forward_hidden_with_cache(audio_hidden, t_embed_decoder, decoder_cache);
        self.decoder.lm_head(hidden)
    }

    /// Continue generation from text tokens (no cache).
    pub fn generate_step(
        &self,
        token_ids: Tensor<Wgpu, 2, Int>,
        t_embed: Tensor<Wgpu, 3>,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let hidden = self.decoder.forward(token_ids, t_embed, offset);
        self.decoder.lm_head(hidden)
    }

    /// Autoregressive generation step with KV cache.
    pub fn generate_step_with_cache(
        &self,
        token_ids: Tensor<Wgpu, 2, Int>,
        t_embed: Tensor<Wgpu, 3>,
        decoder_cache: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let hidden = self
            .decoder
            .forward_with_cache(token_ids, t_embed, decoder_cache);
        self.decoder.lm_head(hidden)
    }

    /// Streaming transcription with KV cache.
    ///
    /// See [`VoxtralModel::transcribe_streaming`](crate::models::voxtral::VoxtralModel::transcribe_streaming)
    /// for details on the position-38 anomaly and token meanings.
    pub fn transcribe_streaming(
        &self,
        mel: Tensor<Wgpu, 3>,
        t_embed_decoder: Tensor<Wgpu, 3>,
    ) -> Vec<i32> {
        let _span = tracing::info_span!("transcribe_streaming").entered();

        let audio_embeds = self.encode_audio(mel);
        let [_, seq_len, d_model] = audio_embeds.dims();

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Vec::new();
        }

        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        // Use embed_tokens_from_ids for the prefix to skip unnecessary
        // GPU round-trip (the prefix tokens are known CPU-side).
        let prefix_text_embeds = self.decoder.embed_tokens_from_ids(&prefix, 1, PREFIX_LEN);

        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

        let prefix_inputs = prefix_audio + prefix_text_embeds;

        // Pre-allocate KV cache to the known sequence length to avoid
        // 52 growing Tensor::cat allocations per decode step (26 layers × K + V).
        let mut decoder_cache = self.decoder.create_cache_preallocated(seq_len);

        let hidden = {
            let _prefill = tracing::info_span!("prefill").entered();
            self.decoder.forward_hidden_with_cache(
                prefix_inputs,
                t_embed_decoder.clone(),
                &mut decoder_cache,
            )
        };
        let logits = self.decoder.lm_head(hidden);

        let last_logits =
            logits
                .clone()
                .slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..logits.dims()[2]]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = first_pred.into_scalar().elem();

        let mut generated = prefix;
        generated.push(first_token);

        // Pre-slice all audio positions to avoid cloning the full audio_embeds
        // tensor every decode step.
        let audio_slices: Vec<Tensor<Wgpu, 3>> = (PREFIX_LEN..seq_len)
            .map(|pos| audio_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]))
            .collect();
        // audio_embeds no longer needed — drop to free GPU memory
        drop(audio_embeds);

        let _decode_span =
            tracing::info_span!("decode", tokens = seq_len - PREFIX_LEN - 1).entered();
        for pos in (PREFIX_LEN + 1)..seq_len {
            let new_token = generated[pos - 1];
            // Use embed_tokens_from_ids to avoid GPU→CPU sync that embed_tokens
            // would trigger (it calls into_data() to read the token ID back).
            let text_embed = self.decoder.embed_tokens_from_ids(&[new_token], 1, 1);

            // Use the pre-sliced audio for position pos-1: positions PREFIX_LEN..seq_len
            // map to audio_slices indices 0.., so pos-1 maps to index pos-1-PREFIX_LEN.
            let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();

            let input = audio_pos + text_embed;

            let hidden = self.decoder.forward_hidden_with_cache(
                input,
                t_embed_decoder.clone(),
                &mut decoder_cache,
            );
            let logits = self.decoder.lm_head(hidden);

            let pred = logits.argmax(2);
            let next_token: i32 = pred.into_scalar().elem();
            generated.push(next_token);
        }

        generated.into_iter().skip(PREFIX_LEN).collect()
    }

    /// Get a reference to the encoder.
    pub fn encoder(&self) -> &Q4AudioEncoder {
        &self.encoder
    }

    /// Get a reference to the decoder.
    pub fn decoder(&self) -> &Q4LanguageModel {
        &self.decoder
    }

    /// Create KV caches for the encoder.
    pub fn create_encoder_cache(&self) -> LayerCaches<Wgpu> {
        self.encoder.create_cache()
    }

    /// Create KV caches for the decoder.
    pub fn create_decoder_cache(&self) -> LayerCaches<Wgpu> {
        self.decoder.create_cache()
    }

    /// Create pre-allocated KV caches for the decoder.
    pub fn create_decoder_cache_preallocated(&self, max_seq: usize) -> LayerCaches<Wgpu> {
        self.decoder.create_cache_preallocated(max_seq)
    }
}
