//! RMSNorm and ADA RMSNorm layers.
//!
//! Standard RMSNorm for the LLM, and adaptive RMSNorm (t-conditioned) for
//! both encoder and LLM layers in Voxtral.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Standard RMSNorm configuration.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-5)]
    pub eps: f64,
}

/// Standard RMSNorm layer.
///
/// Applies: `x * weight / sqrt(mean(x^2) + eps)`
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter.
    pub weight: burn::nn::RmsNorm<B>,
}

impl RmsNormConfig {
    /// Initialize the RmsNorm layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let weight = burn::nn::RmsNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        RmsNorm { weight }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.weight.forward(x)
    }
}

/// ADA RMSNorm configuration (t-conditioned normalization).
#[derive(Config, Debug)]
pub struct AdaRmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Temporal conditioning dimension.
    pub t_cond_dim: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-5)]
    pub eps: f64,
}

/// Adaptive RMSNorm with temporal conditioning.
///
/// Architecture: Linear(d_model -> t_cond_dim) -> SiLU -> Linear(t_cond_dim -> d_model)
/// Then applies: `rms_norm(x) * (1 + scale)`
#[derive(Module, Debug)]
pub struct AdaRmsNorm<B: Backend> {
    /// First projection: d_model -> t_cond_dim
    w0: Linear<B>,
    /// Second projection: t_cond_dim -> d_model
    w2: Linear<B>,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl AdaRmsNormConfig {
    /// Initialize the ADA RMSNorm layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AdaRmsNorm<B> {
        let w0 = LinearConfig::new(self.d_model, self.t_cond_dim)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.t_cond_dim, self.d_model)
            .with_bias(false)
            .init(device);
        AdaRmsNorm {
            w0,
            w2,
            eps: self.eps,
        }
    }
}

impl<B: Backend> AdaRmsNorm<B> {
    /// Create ADA RMSNorm from linear layers (for weight loading).
    pub fn new(w0: Linear<B>, w2: Linear<B>, eps: f64) -> Self {
        Self { w0, w2, eps }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    ///
    /// # Returns
    /// Normalized tensor [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, t_embed: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute adaptive scale: Linear -> SiLU -> Linear
        let scale = self.w0.forward(t_embed);
        let scale = silu(scale);
        let scale = self.w2.forward(scale); // [batch, 1, d_model]

        // Apply RMSNorm: x / sqrt(mean(x^2) + eps)
        let variance = x.clone().powf_scalar(2.0).mean_dim(2); // [batch, seq, 1]
        let x_norm = x / (variance + self.eps).sqrt();

        // Apply adaptive modulation: (1 + scale) * x_norm
        x_norm * (scale + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::TensorData;

    type TestBackend = Wgpu;

    #[test]
    fn test_rms_norm_shape() {
        let device = Default::default();
        let config = RmsNormConfig::new(64);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let out = norm.forward(x);

        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_ada_rms_norm_shape() {
        let device = Default::default();
        let config = AdaRmsNormConfig::new(64, 8);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([2, 1, 64], &device);
        let out = norm.forward(x, t_embed);

        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_rms_norm_vs_reference() {
        use crate::test_utils::{load_test_data, test_data_exists};

        if !test_data_exists("rms_norm_input") {
            println!("Skipping: test_data not generated. Run: ./scripts/reference_forward.py");
            return;
        }

        let device = Default::default();

        // Load reference data
        let input_arr = load_test_data("rms_norm_input").unwrap();
        let weight_arr = load_test_data("rms_norm_weight").unwrap();
        let expected_arr = load_test_data("rms_norm_output").unwrap();

        // Convert to Burn tensors
        let input_data: Vec<f32> = input_arr.iter().cloned().collect();
        let weight_data: Vec<f32> = weight_arr.iter().cloned().collect();
        let expected_data: Vec<f32> = expected_arr.iter().cloned().collect();

        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data, [1, 10, 1280]),
            &device,
        );
        let weight =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(weight_data, [1280]), &device);
        let expected = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(expected_data, [1, 10, 1280]),
            &device,
        );

        // Manual RMSNorm (to test with loaded weights)
        let eps = 1e-5f32;
        let variance = input.clone().powf_scalar(2.0).mean_dim(2);
        let x_norm = input / (variance + eps).sqrt();
        let output = x_norm * weight.unsqueeze::<3>().unsqueeze();

        // Compare
        let output_data = output.to_data();
        let expected_data = expected.to_data();

        let output_slice = output_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("RMSNorm max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "RMSNorm max diff {:.2e} exceeds tolerance",
            max_diff
        );
    }

    #[test]
    fn test_ada_rms_norm_vs_reference() {
        use crate::test_utils::{load_test_data, test_data_exists};

        if !test_data_exists("ada_rms_norm_input") {
            println!(
                "Skipping: test_data not generated. Run: ./scripts/reference_forward.py ada_rms_norm"
            );
            return;
        }

        let device = Default::default();

        // Load reference data
        let input_arr = load_test_data("ada_rms_norm_input").unwrap();
        let t_embed_arr = load_test_data("ada_rms_norm_t_embed").unwrap();
        let w0_arr = load_test_data("ada_rms_norm_w0").unwrap();
        let w2_arr = load_test_data("ada_rms_norm_w2").unwrap();
        let expected_arr = load_test_data("ada_rms_norm_output").unwrap();

        // Convert to Burn tensors
        let input_data: Vec<f32> = input_arr.iter().cloned().collect();
        let t_embed_data: Vec<f32> = t_embed_arr.iter().cloned().collect();
        let w0_data: Vec<f32> = w0_arr.iter().cloned().collect();
        let w2_data: Vec<f32> = w2_arr.iter().cloned().collect();
        let expected_data: Vec<f32> = expected_arr.iter().cloned().collect();

        // Input shape: [1, 10, 3072]
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data, [1, 10, 3072]),
            &device,
        );
        let t_embed = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(t_embed_data, [1, 1, 3072]),
            &device,
        );
        let expected = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(expected_data, [1, 10, 3072]),
            &device,
        );

        // w0: [32, 3072] -> transpose for Burn's Linear (weight shape is [out, in])
        let w0 = Tensor::<TestBackend, 2>::from_data(TensorData::new(w0_data, [32, 3072]), &device);
        // w2: [3072, 32] -> transpose
        let w2 = Tensor::<TestBackend, 2>::from_data(TensorData::new(w2_data, [3072, 32]), &device);

        // Manual ADA RMSNorm
        let eps = 1e-5f32;

        // Compute scale: t_embed @ w0.T -> silu -> @ w2.T
        // t_embed: [1, 1, 3072], w0: [32, 3072]
        // For matmul: [1, 1, 3072] @ [1, 3072, 32] -> [1, 1, 32]
        let w0_3d = w0.transpose().unsqueeze::<3>(); // [1, 3072, 32]
        let w2_3d = w2.transpose().unsqueeze::<3>(); // [1, 32, 3072]
        let scale = t_embed.matmul(w0_3d);
        let scale = silu(scale);
        let scale = scale.matmul(w2_3d); // [1, 1, 3072]

        // RMSNorm
        let variance = input.clone().powf_scalar(2.0).mean_dim(2);
        let x_norm = input / (variance + eps).sqrt();

        // Apply modulation
        let output = x_norm * (scale + 1.0);

        // Compare
        let output_data = output.to_data();
        let expected_data = expected.to_data();

        let output_slice = output_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("ADA RMSNorm max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "ADA RMSNorm max diff {:.2e} exceeds tolerance",
            max_diff
        );
    }
}
