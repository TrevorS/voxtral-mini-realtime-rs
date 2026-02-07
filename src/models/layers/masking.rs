//! Attention masking functions shared between F32 and Q4 model paths.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Apply causal mask to attention scores (same-length Q and K).
///
/// Masks positions where `j > i` with `-inf`.
pub fn apply_causal_mask<B: Backend>(scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<B, 2> = mask.reshape([seq_len, seq_len]);
    let mask: Tensor<B, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

/// Apply sliding window mask to attention scores (same-length Q and K).
///
/// Masks positions where `|i - j| > window` with `-inf`.
pub fn apply_sliding_window_mask<B: Backend>(
    scores: Tensor<B, 4>,
    seq_len: usize,
    window: usize,
) -> Tensor<B, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i.abs_diff(j) > window {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<B, 2> = mask.reshape([seq_len, seq_len]);
    let mask: Tensor<B, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

/// Apply causal mask with different Q/K lengths (for KV cache).
///
/// For each query position `i` (0..q_len), masks key positions `j` where
/// `j > offset + i`.
pub fn apply_causal_mask_with_offset<B: Backend>(
    scores: Tensor<B, 4>,
    q_len: usize,
    kv_len: usize,
    offset: usize,
) -> Tensor<B, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if j > actual_pos {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<B, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<B, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

/// Apply sliding window mask with different Q/K lengths (for KV cache).
///
/// For each query position `i`, masks key positions `j` where
/// `|offset + i - j| > window`.
pub fn apply_sliding_window_mask_with_offset<B: Backend>(
    scores: Tensor<B, 4>,
    q_len: usize,
    kv_len: usize,
    window: usize,
    offset: usize,
) -> Tensor<B, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if actual_pos.abs_diff(j) > window {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<B, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<B, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}
