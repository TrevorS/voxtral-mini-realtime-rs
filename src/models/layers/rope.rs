//! Rotary Position Embeddings (RoPE).
//!
//! Standard RoPE with configurable theta (1M for Voxtral).

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// RoPE configuration.
#[derive(Config, Debug)]
pub struct RoPEConfig {
    /// Head dimension.
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Base frequency (theta).
    #[config(default = 1_000_000.0)]
    pub theta: f64,
}

/// Rotary Position Embeddings.
///
/// Pre-computes cos/sin tables for efficient application during forward pass.
#[derive(Module, Debug)]
pub struct RoPE<B: Backend> {
    /// Cosine frequencies [max_seq_len, head_dim/2]
    cos: Tensor<B, 2>,
    /// Sine frequencies [max_seq_len, head_dim/2]
    sin: Tensor<B, 2>,
}

impl RoPEConfig {
    /// Initialize RoPE with pre-computed frequencies.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RoPE<B> {
        let half_dim = self.head_dim / 2;

        // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in 0..half_dim
        // Using the formula from reference: 1 / (theta ** (arange(0, dim, 2) / dim))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (self.theta as f32).powf((2 * i) as f32 / self.head_dim as f32))
            .collect();

        // Position indices
        let positions: Vec<f32> = (0..self.max_seq_len).map(|i| i as f32).collect();

        // Compute outer product: freqs[i, j] = positions[i] * inv_freq[j]
        let mut freqs = vec![0.0f32; self.max_seq_len * half_dim];
        for i in 0..self.max_seq_len {
            for j in 0..half_dim {
                freqs[i * half_dim + j] = positions[i] * inv_freq[j];
            }
        }

        // Create tensors - first create 1D then reshape
        let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
            .reshape([self.max_seq_len, half_dim]);

        // Compute cos/sin
        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        RoPE { cos, sin }
    }
}

impl<B: Backend> RoPE<B> {
    /// Apply rotary embeddings to query and key tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq, heads, head_dim]
    /// * `k` - Key tensor [batch, seq, heads, head_dim]
    /// * `offset` - Position offset for KV cache
    ///
    /// # Returns
    /// (q_rotated, k_rotated) with same shapes as input
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq_len = q.dims()[1];

        // Slice cos/sin for current sequence
        let [_max_len, half_dim] = self.cos.dims();
        let cos = self
            .cos
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);
        let sin = self
            .sin
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);

        let q_rot = self.apply_rotation(q, cos.clone(), sin.clone());
        let k_rot = self.apply_rotation(k, cos, sin);

        (q_rot, k_rot)
    }

    /// Apply rotation to a single tensor.
    fn apply_rotation(
        &self,
        x: Tensor<B, 4>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let [batch, seq, heads, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Reshape to separate interleaved pairs: [batch, seq, heads, half_dim, 2]
        let x_pairs = x.reshape([batch, seq, heads, half_dim, 2]);

        // Extract real (even indices) and imaginary (odd indices) parts
        // Use reshape instead of squeeze to avoid removing batch dim when batch=1
        let x_r: Tensor<B, 4> = x_pairs
            .clone()
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 0..1])
            .reshape([batch, seq, heads, half_dim]);
        let x_i: Tensor<B, 4> = x_pairs
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 1..2])
            .reshape([batch, seq, heads, half_dim]);

        // Broadcast cos/sin: [seq, half_dim] -> [1, seq, 1, half_dim]
        // First unsqueeze to [seq, half_dim, 1] then permute/reshape
        let cos: Tensor<B, 4> = cos
            .unsqueeze_dim::<3>(0) // [1, seq, half_dim]
            .unsqueeze_dim(2); // [1, seq, 1, half_dim]
        let sin: Tensor<B, 4> = sin.unsqueeze_dim::<3>(0).unsqueeze_dim(2);

        // Apply rotation: [x_r * cos - x_i * sin, x_r * sin + x_i * cos]
        let out_r = x_r.clone() * cos.clone() - x_i.clone() * sin.clone();
        let out_i = x_r * sin + x_i * cos;

        // Interleave back: stack on last dim then reshape
        let out_r: Tensor<B, 5> = out_r.unsqueeze_dim(4); // [batch, seq, heads, half_dim, 1]
        let out_i: Tensor<B, 5> = out_i.unsqueeze_dim(4);
        let out = Tensor::cat(vec![out_r, out_i], 4); // [batch, seq, heads, half_dim, 2]
        out.reshape([batch, seq, heads, head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::TensorData;

    type TestBackend = Wgpu;

    #[test]
    fn test_rope_shape() {
        let device = Default::default();
        let config = RoPEConfig::new(64, 512);
        let rope = config.init::<TestBackend>(&device);

        let q = Tensor::<TestBackend, 4>::zeros([2, 10, 8, 64], &device);
        let k = Tensor::<TestBackend, 4>::zeros([2, 10, 8, 64], &device);

        let (q_rot, k_rot) = rope.apply(q, k, 0);

        assert_eq!(q_rot.dims(), [2, 10, 8, 64]);
        assert_eq!(k_rot.dims(), [2, 10, 8, 64]);
    }

    #[test]
    fn test_rope_vs_reference() {
        use crate::test_utils::{load_test_data, test_data_exists};

        if !test_data_exists("rope_input") {
            println!("Skipping: test_data not generated. Run: ./scripts/reference_forward.py rope");
            return;
        }

        let device = Default::default();

        // Load reference data
        let input_arr = load_test_data("rope_input").unwrap();
        let cos_arr = load_test_data("rope_cos").unwrap();
        let sin_arr = load_test_data("rope_sin").unwrap();
        let expected_arr = load_test_data("rope_output").unwrap();

        // Get shapes from numpy arrays
        let input_shape = input_arr.shape().to_vec();
        let cos_shape = cos_arr.shape().to_vec();

        // Convert to Burn tensors
        let input_data: Vec<f32> = input_arr.iter().cloned().collect();
        let cos_data: Vec<f32> = cos_arr.iter().cloned().collect();
        let sin_data: Vec<f32> = sin_arr.iter().cloned().collect();
        let expected_data: Vec<f32> = expected_arr.iter().cloned().collect();

        // input: [1, 100, 32, 64] (batch, seq, heads, head_dim)
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                input_data,
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                ],
            ),
            &device,
        );
        // cos/sin: [100, 32] (seq, half_dim)
        let cos = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(cos_data, [cos_shape[0], cos_shape[1]]),
            &device,
        );
        let sin = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(sin_data, [cos_shape[0], cos_shape[1]]),
            &device,
        );
        let expected = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                expected_data,
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                ],
            ),
            &device,
        );

        // Create RoPE and manually set cos/sin
        let rope = RoPE { cos, sin };

        // Apply (using input as both q and k, we only check q output)
        let (output, _) = rope.apply(input.clone(), input, 0);

        // Compare
        let output_data = output.to_data();
        let expected_data = expected.to_data();

        let output_slice = output_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("RoPE max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "RoPE max diff {:.2e} exceeds tolerance",
            max_diff
        );
    }
}
