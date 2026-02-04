//! SwiGLU MLP layer.
//!
//! Used in both audio encoder and language model.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// SwiGLU configuration.
#[derive(Config, Debug)]
pub struct SwiGLUConfig {
    /// Input/output dimension.
    pub d_model: usize,
    /// Hidden dimension (typically 4x d_model for transformers).
    pub hidden_dim: usize,
    /// Whether to use bias (encoder=false, LLM=false).
    #[config(default = false)]
    pub bias: bool,
}

/// SwiGLU MLP layer.
///
/// Computes: `w2(silu(w1(x)) * w3(x))`
///
/// Named w1/w2/w3 to match Voxtral weight names:
/// - w1: gate projection
/// - w2: down projection
/// - w3: up projection
#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    /// Gate projection: d_model -> hidden_dim
    w1: Linear<B>,
    /// Down projection: hidden_dim -> d_model
    w2: Linear<B>,
    /// Up projection: d_model -> hidden_dim
    w3: Linear<B>,
}

impl SwiGLUConfig {
    /// Initialize the SwiGLU layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGLU<B> {
        let w1 = LinearConfig::new(self.d_model, self.hidden_dim)
            .with_bias(self.bias)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_dim, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let w3 = LinearConfig::new(self.d_model, self.hidden_dim)
            .with_bias(self.bias)
            .init(device);

        SwiGLU { w1, w2, w3 }
    }
}

impl<B: Backend> SwiGLU<B> {
    /// Create SwiGLU from linear layers (for weight loading).
    pub fn new(w1: Linear<B>, w2: Linear<B>, w3: Linear<B>) -> Self {
        Self { w1, w2, w3 }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.w1.forward(x.clone());
        let gate = silu(gate);
        let up = self.w3.forward(x);
        self.w2.forward(gate * up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::TensorData;

    type TestBackend = Wgpu;

    #[test]
    fn test_swiglu_shape() {
        let device = Default::default();
        let config = SwiGLUConfig::new(64, 256);
        let mlp = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let out = mlp.forward(x);

        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_swiglu_vs_reference() {
        use crate::test_utils::{load_test_data, test_data_exists};

        if !test_data_exists("swiglu_input") {
            println!(
                "Skipping: test_data not generated. Run: ./scripts/reference_forward.py swiglu"
            );
            return;
        }

        let device = Default::default();

        // Load reference data
        let input_arr = load_test_data("swiglu_input").unwrap();
        let w1_arr = load_test_data("swiglu_w1").unwrap();
        let w2_arr = load_test_data("swiglu_w2").unwrap();
        let w3_arr = load_test_data("swiglu_w3").unwrap();
        let expected_arr = load_test_data("swiglu_output").unwrap();

        // Convert to Burn tensors
        let input_data: Vec<f32> = input_arr.iter().cloned().collect();
        let w1_data: Vec<f32> = w1_arr.iter().cloned().collect();
        let w2_data: Vec<f32> = w2_arr.iter().cloned().collect();
        let w3_data: Vec<f32> = w3_arr.iter().cloned().collect();
        let expected_data: Vec<f32> = expected_arr.iter().cloned().collect();

        // Get shapes
        let w1_shape = w1_arr.shape().to_vec();
        let w2_shape = w2_arr.shape().to_vec();
        let w3_shape = w3_arr.shape().to_vec();

        // input: [1, 10, 1280]
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data, [1, 10, 1280]),
            &device,
        );
        let expected = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(expected_data, [1, 10, 1280]),
            &device,
        );

        // Weights: PyTorch Linear stores [out_features, in_features]
        // Burn Linear expects [out_features, in_features] too
        let w1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(w1_data, [w1_shape[0], w1_shape[1]]),
            &device,
        );
        let w2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(w2_data, [w2_shape[0], w2_shape[1]]),
            &device,
        );
        let w3 = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(w3_data, [w3_shape[0], w3_shape[1]]),
            &device,
        );

        // Manual SwiGLU: w2(silu(w1(x)) * w3(x))
        // Linear: x @ W.T
        // x: [1, 10, 1280], w1: [5120, 1280] -> need [1, 1280, 5120] for matmul
        let w1_3d = w1.transpose().unsqueeze::<3>(); // [1, 1280, 5120]
        let w2_3d = w2.transpose().unsqueeze::<3>(); // [1, 5120, 1280]
        let w3_3d = w3.transpose().unsqueeze::<3>(); // [1, 1280, 5120]

        let gate = input.clone().matmul(w1_3d);
        let gate = silu(gate);
        let up = input.matmul(w3_3d);
        let output = (gate * up).matmul(w2_3d);

        // Compare
        let output_data = output.to_data();
        let expected_data = expected.to_data();

        let output_slice = output_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("SwiGLU max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "SwiGLU max diff {:.2e} exceeds tolerance",
            max_diff
        );
    }
}
