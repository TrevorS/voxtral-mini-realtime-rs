//! Quantized linear layer (Q4_0 and Q8_0).
//!
//! [`Q4Linear`] wraps a [`Q4Tensor`] weight matrix and optional f32 bias,
//! providing a `forward` method that delegates to [`quantized_matmul`].
//! Despite the "Q4" name (kept for backwards compatibility), it supports
//! both Q4_0 and Q8_0 quantized weights transparently.

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use super::op::quantized_matmul;
use super::tensor::Q4Tensor;

/// A linear layer with quantized weights (Q4_0 or Q8_0).
///
/// Stores weights as `[out_features, in_features]` in quantized format and an
/// optional f32 bias vector. The forward pass computes
/// `x @ weights^T + bias` via the fused dequant+matmul GPU kernel.
pub struct Q4Linear {
    weights: Q4Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4Linear {
    /// Create a new quantized linear layer.
    ///
    /// `weights` shape must be `[out_features, in_features]`.
    /// Works with both Q4_0 and Q8_0 quantized tensors.
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Access the underlying quantized weight tensor.
    pub fn weights(&self) -> &Q4Tensor {
        &self.weights
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = quantized_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

/// Fused Q/K/V projection: stores concatenated quantized weights and splits output.
///
/// Instead of 3 separate matmul launches for wq, wk, wv, uses a single
/// concatenated weight matrix `[q_out + k_out + v_out, in_features]`.
/// Reduces kernel launches from 3 to 1 per layer.
pub struct Q4FusedQKV {
    weights: Q4Tensor,
    q_out: usize,
    k_out: usize,
    v_out: usize,
}

/// Fused gate+up projection for SwiGLU: stores concatenated w1||w3 quantized weights.
///
/// Reduces 2 matmul launches to 1 per FFN layer.
pub struct Q4FusedGateUp {
    weights: Q4Tensor,
    gate_out: usize,
    up_out: usize,
}

impl Q4FusedGateUp {
    /// Create from a pre-built concatenated quantized tensor.
    pub fn new(weights: Q4Tensor, gate_out: usize, up_out: usize) -> Self {
        Self {
            weights,
            gate_out,
            up_out,
        }
    }

    /// Forward: single quantized matmul -> split into (gate, up).
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
        let fused = quantized_matmul(x, &self.weights);

        let gate = fused.clone().narrow(2, 0, self.gate_out);
        let up = fused.narrow(2, self.gate_out, self.up_out);

        (gate, up)
    }
}

impl Q4FusedQKV {
    /// Create from a pre-built concatenated quantized tensor.
    pub fn new(weights: Q4Tensor, q_out: usize, k_out: usize, v_out: usize) -> Self {
        Self {
            weights,
            q_out,
            k_out,
            v_out,
        }
    }

    /// Forward: single quantized matmul -> split into (q, k, v).
    ///
    /// `x` shape: `[B, M, K]`.
    /// Returns: `(q [B, M, q_out], k [B, M, k_out], v [B, M, v_out])`.
    pub fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
    ) -> (Tensor<Wgpu, 3>, Tensor<Wgpu, 3>, Tensor<Wgpu, 3>) {
        let fused = quantized_matmul(x, &self.weights);

        let q = fused.clone().narrow(2, 0, self.q_out);
        let k = fused.clone().narrow(2, self.q_out, self.k_out);
        let v = fused.narrow(2, self.q_out + self.k_out, self.v_out);

        (q, k, v)
    }
}
