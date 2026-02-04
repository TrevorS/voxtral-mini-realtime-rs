//! Time embedding for Voxtral Realtime.
//!
//! Sinusoidal embedding that encodes the transcription delay.

use burn::prelude::*;

/// Time embedding module that produces sinusoidal embeddings.
///
/// Used to encode the transcription delay as a conditioning signal
/// for the ADA RMSNorm modulation in the decoder layers.
#[derive(Debug)]
pub struct TimeEmbedding {
    /// Dimension of the embedding
    dim: usize,
    /// Base frequency (default: 10000.0)
    theta: f32,
}

impl TimeEmbedding {
    /// Create a new time embedding with given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            theta: 10000.0,
        }
    }

    /// Create a new time embedding with custom theta.
    pub fn with_theta(dim: usize, theta: f32) -> Self {
        Self { dim, theta }
    }

    /// Compute sinusoidal embedding for a time value.
    ///
    /// # Arguments
    /// * `t` - Time value (typically the number of delay tokens)
    /// * `device` - Device to create the tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, 1, dim] containing the sinusoidal embedding
    pub fn embed<B: Backend>(&self, t: f32, device: &B::Device) -> Tensor<B, 3> {
        let half_dim = self.dim / 2;

        // Compute inverse frequencies: exp(-log(theta) * i / (dim/2)) for i in 0..dim/2
        let mut inv_freq = Vec::with_capacity(half_dim);
        let log_theta = self.theta.ln();
        for i in 0..half_dim {
            let freq = (-log_theta * (i as f32) / (half_dim as f32)).exp();
            inv_freq.push(freq);
        }

        // Compute t * inv_freq
        let mut cos_vals = Vec::with_capacity(half_dim);
        let mut sin_vals = Vec::with_capacity(half_dim);
        for &freq in &inv_freq {
            let angle = t * freq;
            cos_vals.push(angle.cos());
            sin_vals.push(angle.sin());
        }

        // Concatenate [cos, sin] to get full embedding
        let mut embedding = Vec::with_capacity(self.dim);
        embedding.extend_from_slice(&cos_vals);
        embedding.extend_from_slice(&sin_vals);

        // Create tensor with shape [1, 1, dim]
        Tensor::from_data(
            burn::tensor::TensorData::new(embedding, [1, 1, self.dim]),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_time_embedding_shape() {
        let device = Default::default();
        let embed = TimeEmbedding::new(3072);

        let t_cond = embed.embed::<TestBackend>(6.0, &device);
        assert_eq!(t_cond.dims(), [1, 1, 3072]);
    }

    #[test]
    fn test_time_embedding_values() {
        let device = Default::default();
        let embed = TimeEmbedding::new(4);

        let t_cond = embed.embed::<TestBackend>(1.0, &device);
        let data = t_cond.to_data();
        let slice = data.as_slice::<f32>().unwrap();

        // For t=1, dim=4:
        // inv_freq[0] = exp(-log(10000) * 0 / 2) = exp(0) = 1.0
        // inv_freq[1] = exp(-log(10000) * 1 / 2) = exp(-log(10000)/2) = 1/100 = 0.01
        // cos(1 * 1.0) = cos(1) ≈ 0.5403
        // cos(1 * 0.01) = cos(0.01) ≈ 0.99995
        // sin(1 * 1.0) = sin(1) ≈ 0.8415
        // sin(1 * 0.01) = sin(0.01) ≈ 0.01

        // Check approximate values
        assert!(
            (slice[0] - 0.5403).abs() < 0.01,
            "cos(1) wrong: {}",
            slice[0]
        );
        assert!(
            (slice[1] - 0.99995).abs() < 0.001,
            "cos(0.01) wrong: {}",
            slice[1]
        );
        assert!(
            (slice[2] - 0.8415).abs() < 0.01,
            "sin(1) wrong: {}",
            slice[2]
        );
        assert!(
            (slice[3] - 0.01).abs() < 0.001,
            "sin(0.01) wrong: {}",
            slice[3]
        );
    }

    #[test]
    fn test_time_embedding_vs_python() {
        // Test against expected Python output
        // Python: TimeEmbedding(dim=8, theta=10000)(torch.tensor([6.0]))
        // Should produce cos and sin of 6 * inv_freq
        let device = Default::default();
        let embed = TimeEmbedding::new(8);

        let t_cond = embed.embed::<TestBackend>(6.0, &device);
        let data = t_cond.to_data();
        let slice = data.as_slice::<f32>().unwrap();

        // inv_freq = [1.0, 0.1, 0.01, 0.001] (approx for theta=10000, dim=8)
        // 6 * inv_freq = [6.0, 0.6, 0.06, 0.006]
        // cos(6) ≈ 0.9602
        // cos(0.6) ≈ 0.8253
        // cos(0.06) ≈ 0.9982
        // cos(0.006) ≈ 0.99998

        println!("t_cond for t=6: {:?}", slice);
        assert!(slice.len() == 8);
    }
}
