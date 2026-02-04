//! Audio-Language Adapter for Voxtral.
//!
//! Projects encoder output to LLM input dimension.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Adapter configuration.
#[derive(Config, Debug)]
pub struct AudioLanguageAdapterConfig {
    /// Input dimension (encoder d_model × 2 due to reshape).
    pub in_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension (LLM d_model).
    pub out_dim: usize,
}

impl AudioLanguageAdapterConfig {
    /// Create a config from the Voxtral model defaults.
    ///
    /// Encoder output: 1280
    /// After 2x reshape: 2560
    /// Hidden: 5120 (not actually used in Sequential)
    /// LLM input: 3072
    ///
    /// Architecture: Linear(2560, 3072) -> GELU -> Linear(3072, 3072)
    /// But looking at weights, it's:
    /// - projection.0: [3072, 5120] (Linear 5120 -> 3072)
    /// - projection.2: [3072, 3072] (Linear 3072 -> 3072)
    ///
    /// So the actual flow is: Linear(5120, 3072) -> GELU -> Linear(3072, 3072)
    /// Where 5120 = 1280 * 4 / 2 = 2560 after reshape? Or maybe the reshape
    /// is handled differently. Let's use the actual weight shapes.
    pub fn voxtral() -> Self {
        Self {
            in_dim: 5120, // Actual input dim from weights
            hidden_dim: 3072,
            out_dim: 3072,
        }
    }
}

/// Audio-Language adapter projection.
///
/// Two-layer MLP with GELU activation that projects audio encoder output
/// to the LLM's input dimension.
///
/// Architecture: Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)
#[derive(Module, Debug)]
pub struct AudioLanguageAdapter<B: Backend> {
    /// First projection.
    linear1: Linear<B>,
    /// Second projection.
    linear2: Linear<B>,
}

impl AudioLanguageAdapterConfig {
    /// Initialize the adapter.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioLanguageAdapter<B> {
        let linear1 = LinearConfig::new(self.in_dim, self.hidden_dim)
            .with_bias(false)
            .init(device);
        let linear2 = LinearConfig::new(self.hidden_dim, self.out_dim)
            .with_bias(false)
            .init(device);

        AudioLanguageAdapter { linear1, linear2 }
    }
}

impl<B: Backend> AudioLanguageAdapter<B> {
    /// Create adapter from linear layers (for weight loading).
    pub fn new(linear1: Linear<B>, linear2: Linear<B>) -> Self {
        Self { linear1, linear2 }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Audio encoder output [batch, seq, in_dim]
    ///
    /// # Returns
    /// Projected features [batch, seq, out_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = gelu(x);
        self.linear2.forward(x)
    }
}

/// Reshape encoder output for adapter input.
///
/// The encoder outputs [batch, seq, d_model] where each position represents
/// 80ms of audio. To match the 12.5 Hz frame rate, we reshape adjacent
/// frames together.
///
/// # Arguments
/// * `encoder_output` - [batch, seq, encoder_d_model]
/// * `reshape_factor` - Number of frames to concatenate (typically 2)
///
/// # Returns
/// Reshaped tensor [batch, seq/reshape_factor, encoder_d_model * reshape_factor]
pub fn reshape_encoder_output<B: Backend>(
    encoder_output: Tensor<B, 3>,
    reshape_factor: usize,
) -> Tensor<B, 3> {
    let [batch, seq, d_model] = encoder_output.dims();

    // Truncate to even number of frames
    let new_seq = seq / reshape_factor;
    let truncated_seq = new_seq * reshape_factor;

    let x = encoder_output.slice([0..batch, 0..truncated_seq, 0..d_model]);

    // Reshape: [batch, seq, d_model] -> [batch, new_seq, d_model * factor]
    x.reshape([batch, new_seq, d_model * reshape_factor])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_adapter_shape() {
        let device = Default::default();

        let config = AudioLanguageAdapterConfig::new(128, 64, 64);
        let adapter = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 10, 128], &device);
        let out = adapter.forward(x);

        assert_eq!(out.dims(), [1, 10, 64]);
    }

    #[test]
    fn test_voxtral_adapter() {
        let device = Default::default();

        let config = AudioLanguageAdapterConfig::voxtral();
        let adapter = config.init::<TestBackend>(&device);

        // Input after reshape: 1280 * 4 = 5120
        let x = Tensor::<TestBackend, 3>::zeros([1, 10, 5120], &device);
        let out = adapter.forward(x);

        // Output: 3072 (LLM dimension)
        assert_eq!(out.dims(), [1, 10, 3072]);
    }

    #[test]
    fn test_reshape_encoder_output() {
        let device = Default::default();

        // Encoder output: [batch=1, seq=20, d_model=64]
        let encoder_output = Tensor::<TestBackend, 3>::zeros([1, 20, 64], &device);

        // Reshape with factor 2
        let reshaped = reshape_encoder_output(encoder_output, 2);

        // Should be [1, 10, 128]
        assert_eq!(reshaped.dims(), [1, 10, 128]);
    }

    #[test]
    fn test_reshape_odd_sequence() {
        let device = Default::default();

        // Encoder output with odd length
        let encoder_output = Tensor::<TestBackend, 3>::zeros([1, 21, 64], &device);

        // Reshape with factor 2 (should truncate to 20)
        let reshaped = reshape_encoder_output(encoder_output, 2);

        // Should be [1, 10, 128]
        assert_eq!(reshaped.dims(), [1, 10, 128]);
    }
}
