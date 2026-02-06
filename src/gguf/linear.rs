//! Q4_0 quantized linear layer.
//!
//! [`Q4Linear`] wraps a [`Q4Tensor`] weight matrix and optional f32 bias,
//! providing a `forward` method that delegates to [`q4_matmul`].

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use super::op::q4_matmul;
use super::tensor::Q4Tensor;

/// A linear layer with Q4_0 quantized weights.
///
/// Stores weights as `[out_features, in_features]` in Q4_0 format and an
/// optional f32 bias vector. The forward pass computes
/// `x @ weights^T + bias` via the fused dequant+matmul GPU kernel.
pub struct Q4Linear {
    weights: Q4Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4Linear {
    /// Create a new Q4 linear layer.
    ///
    /// `weights` shape must be `[out_features, in_features]`.
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q4_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}
