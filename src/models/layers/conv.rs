//! Convolutional downsampler for the audio encoder.
//!
//! Two Conv1d layers with GELU activation that downsample mel spectrograms
//! from 128 channels to 1280 channels with 4x temporal downsampling.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Conv downsampler configuration.
#[derive(Config, Debug)]
pub struct ConvDownsamplerConfig {
    /// Input channels (mel bins).
    pub in_channels: usize,
    /// Hidden channels after first conv.
    pub hidden_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size for both convolutions.
    #[config(default = 3)]
    pub kernel_size: usize,
    /// Stride for both convolutions (total downsample = stride^2).
    #[config(default = 2)]
    pub stride: usize,
}

/// Convolutional downsampler.
///
/// Architecture:
/// - Conv1d(in_channels -> hidden_channels, kernel=3, stride=2, pad=1) + GELU
/// - Conv1d(hidden_channels -> out_channels, kernel=3, stride=2, pad=1) + GELU
///
/// Total temporal downsampling: 4x (stride 2 × stride 2)
#[derive(Module, Debug)]
pub struct ConvDownsampler<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
}

impl ConvDownsamplerConfig {
    /// Initialize the ConvDownsampler.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvDownsampler<B> {
        // Padding of 1 with kernel 3 and stride 2 gives: (L + 2*1 - 3) / 2 + 1 = (L - 1) / 2 + 1 = (L + 1) / 2
        // For L=100: (100 + 1) / 2 = 50
        let conv1 = Conv1dConfig::new(self.in_channels, self.hidden_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true)
            .init(device);

        let conv2 = Conv1dConfig::new(self.hidden_channels, self.out_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true)
            .init(device);

        ConvDownsampler { conv1, conv2 }
    }
}

impl<B: Backend> ConvDownsampler<B> {
    /// Create downsampler from conv layers (for weight loading).
    pub fn new(conv1: Conv1d<B>, conv2: Conv1d<B>) -> Self {
        Self { conv1, conv2 }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Mel spectrogram [batch, mel_bins, time]
    ///
    /// # Returns
    /// Downsampled features [batch, out_channels, time / 4]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv1.forward(x);
        let x = gelu(x);
        let x = self.conv2.forward(x);
        gelu(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::TensorData;

    type TestBackend = Wgpu;

    #[test]
    fn test_conv_downsampler_shape() {
        let device = Default::default();
        let config = ConvDownsamplerConfig::new(128, 1280, 1280);
        let conv = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=128, time=100]
        let x = Tensor::<TestBackend, 3>::zeros([1, 128, 100], &device);
        let out = conv.forward(x);

        // Output should be [1, 1280, 25] (4x downsample)
        assert_eq!(out.dims()[0], 1);
        assert_eq!(out.dims()[1], 1280);
        // With padding=1, kernel=3, stride=2: (100 + 2 - 3) / 2 + 1 = 50
        // Then again: (50 + 2 - 3) / 2 + 1 = 25
        assert_eq!(out.dims()[2], 25);
    }

    #[test]
    fn test_conv_vs_reference() {
        use crate::test_utils::{load_test_data, test_data_exists};

        if !test_data_exists("conv_input") {
            println!("Skipping: test_data not generated. Run: ./scripts/reference_forward.py conv");
            return;
        }

        let device = Default::default();

        // Load reference data
        let input_arr = load_test_data("conv_input").unwrap();
        let conv1_w_arr = load_test_data("conv1_weight").unwrap();
        let conv1_b_arr = load_test_data("conv1_bias").unwrap();
        let conv2_w_arr = load_test_data("conv2_weight").unwrap();
        let conv2_b_arr = load_test_data("conv2_bias").unwrap();
        let expected_arr = load_test_data("conv_output").unwrap();

        // Get shapes
        let conv1_w_shape = conv1_w_arr.shape().to_vec();
        let conv2_w_shape = conv2_w_arr.shape().to_vec();

        // Convert to Burn tensors
        let input_data: Vec<f32> = input_arr.iter().cloned().collect();
        let conv1_w_data: Vec<f32> = conv1_w_arr.iter().cloned().collect();
        let conv1_b_data: Vec<f32> = conv1_b_arr.iter().cloned().collect();
        let conv2_w_data: Vec<f32> = conv2_w_arr.iter().cloned().collect();
        let conv2_b_data: Vec<f32> = conv2_b_arr.iter().cloned().collect();
        let expected_data: Vec<f32> = expected_arr.iter().cloned().collect();

        // Input: [1, 128, 100]
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(input_data, [1, 128, 100]),
            &device,
        );

        // conv1 weight: [out, in, kernel] = [1280, 128, 3]
        let conv1_w = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                conv1_w_data,
                [conv1_w_shape[0], conv1_w_shape[1], conv1_w_shape[2]],
            ),
            &device,
        );
        let conv1_b = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(conv1_b_data, [conv1_w_shape[0]]),
            &device,
        );

        // conv2 weight: [out, in, kernel] = [1280, 1280, 3]
        let conv2_w = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                conv2_w_data,
                [conv2_w_shape[0], conv2_w_shape[1], conv2_w_shape[2]],
            ),
            &device,
        );
        let conv2_b = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(conv2_b_data, [conv2_w_shape[0]]),
            &device,
        );

        // Expected output shape
        let expected_shape = expected_arr.shape().to_vec();
        let expected = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                expected_data,
                [expected_shape[0], expected_shape[1], expected_shape[2]],
            ),
            &device,
        );

        // Use burn's tensor conv1d function directly
        use burn::tensor::module::conv1d;
        use burn::tensor::ops::ConvOptions;

        let options1 = ConvOptions::new([2], [1], [1], 1);
        let x1 = conv1d(input, conv1_w, Some(conv1_b), options1);
        let x1 = gelu(x1);

        let options2 = ConvOptions::new([2], [1], [1], 1);
        let output = conv1d(x1, conv2_w, Some(conv2_b), options2);
        let output = gelu(output);

        // Compare
        let output_data = output.to_data();
        let expected_data = expected.to_data();

        let output_slice = output_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in output_slice.iter().zip(expected_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("Conv max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-3,
            "Conv max diff {:.2e} exceeds tolerance",
            max_diff
        );
    }
}
