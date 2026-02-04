//! Audio padding for streaming inference.
//!
//! Voxtral streaming mode requires left-padding audio with silence to align
//! with the prefix tokens. This module implements the padding logic.

use super::AudioBuffer;

/// Padding configuration for Voxtral streaming.
#[derive(Debug, Clone)]
pub struct PadConfig {
    /// Sample rate (must be 16kHz for Voxtral).
    pub sample_rate: u32,
    /// Number of left padding tokens (default: 32).
    pub n_left_pad_tokens: usize,
    /// Frame rate in Hz (default: 12.5).
    pub frame_rate: f32,
    /// Extra right padding tokens for encoder alignment (default: 17).
    /// mistral-common adds 17 extra tokens to ensure proper conv/reshape alignment.
    pub extra_right_pad_tokens: usize,
}

impl Default for PadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_left_pad_tokens: 32,
            frame_rate: 12.5,
            extra_right_pad_tokens: 17,
        }
    }
}

impl PadConfig {
    /// Create a new padding configuration with Voxtral defaults.
    pub fn voxtral() -> Self {
        Self::default()
    }

    /// Samples per audio token.
    pub fn samples_per_token(&self) -> usize {
        (self.sample_rate as f32 / self.frame_rate) as usize
    }

    /// Number of samples to left-pad.
    pub fn left_pad_samples(&self) -> usize {
        self.n_left_pad_tokens * self.samples_per_token()
    }

    /// Calculate right padding needed to align to token boundary plus extra padding.
    ///
    /// mistral-common pads to token boundary + 17 extra tokens for proper
    /// convolution and reshape alignment in the encoder.
    pub fn right_pad_samples(&self, total_samples: usize) -> usize {
        let spt = self.samples_per_token();
        let remainder = total_samples % spt;
        let alignment_pad = if remainder == 0 { 0 } else { spt - remainder };
        let extra_pad = self.extra_right_pad_tokens * spt;
        alignment_pad + extra_pad
    }
}

/// Left-pad audio with silence for streaming inference.
///
/// This adds zeros at the beginning of the audio to align with the
/// `n_left_pad_tokens` prefix, and optionally right-pads to align
/// to a token boundary.
///
/// # Arguments
/// * `audio` - Input audio buffer (should be 16kHz)
/// * `config` - Padding configuration
///
/// # Returns
/// New audio buffer with padding applied
pub fn pad_audio(audio: &AudioBuffer, config: &PadConfig) -> AudioBuffer {
    let left_pad = config.left_pad_samples();
    let right_pad = config.right_pad_samples(audio.samples.len() + left_pad);

    let total_len = left_pad + audio.samples.len() + right_pad;
    let mut padded = vec![0.0f32; total_len];

    // Copy original audio after left padding
    padded[left_pad..left_pad + audio.samples.len()].copy_from_slice(&audio.samples);

    AudioBuffer {
        samples: padded,
        sample_rate: audio.sample_rate,
    }
}

/// Calculate the number of audio tokens for a given number of samples.
pub fn num_audio_tokens(samples: usize, config: &PadConfig) -> usize {
    samples / config.samples_per_token()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_config_defaults() {
        let config = PadConfig::voxtral();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_left_pad_tokens, 32);
        assert_eq!(config.frame_rate, 12.5);
        assert_eq!(config.samples_per_token(), 1280);
        assert_eq!(config.left_pad_samples(), 40960);
    }

    #[test]
    fn test_right_pad_alignment() {
        let config = PadConfig::voxtral();

        // Extra padding = 17 * 1280 = 21760 samples
        let extra = 17 * 1280;

        // Already aligned to token boundary - just add extra
        assert_eq!(config.right_pad_samples(1280 * 10), extra);

        // Needs alignment + extra
        assert_eq!(config.right_pad_samples(1280 * 10 + 100), 1180 + extra);
        assert_eq!(config.right_pad_samples(1280 * 10 + 1), 1279 + extra);
    }

    #[test]
    fn test_pad_audio() {
        let config = PadConfig::voxtral();

        // Create test audio (not aligned to token boundary)
        let audio = AudioBuffer {
            samples: vec![1.0; 255168], // Same as test_data/mary_had_lamb.wav
            sample_rate: 16000,
        };

        let padded = pad_audio(&audio, &config);

        // Check left padding (zeros)
        let left_pad = config.left_pad_samples();
        for &s in &padded.samples[..left_pad] {
            assert_eq!(s, 0.0, "Left padding should be zeros");
        }

        // Check original audio preserved
        for (i, &s) in audio.samples.iter().enumerate() {
            assert_eq!(
                padded.samples[left_pad + i],
                s,
                "Original audio should be preserved"
            );
        }

        // Check total is aligned
        assert_eq!(
            padded.samples.len() % config.samples_per_token(),
            0,
            "Total should be aligned to token boundary"
        );

        // Check against known values from Python mistral-common:
        // Original: 255168 samples
        // Left pad: 40960 samples (32 tokens)
        // Total after left: 296128 samples
        // Alignment pad: 832 samples (to reach 296960 = 232 tokens)
        // Extra pad: 21760 samples (17 tokens for encoder alignment)
        // Total: 318720 samples = 249 tokens
        let expected_total = 40960 + 255168 + 832 + 21760;
        assert_eq!(padded.samples.len(), expected_total);
        assert_eq!(padded.samples.len(), 318720); // Match Python exactly

        let num_tokens = num_audio_tokens(padded.samples.len(), &config);
        assert_eq!(num_tokens, 249);
    }

    #[test]
    fn test_pad_matches_python() {
        // Verify padding matches mistral-common's behavior exactly.
        // From Python reference:
        // - Original: 255168 samples (15.948s)
        // - Left pad: 40960 samples (32 tokens)
        // - Right pad: 22592 samples (alignment + 17 extra tokens)
        // - Total: 318720 samples (249 tokens)

        let config = PadConfig::voxtral();
        let audio = AudioBuffer {
            samples: vec![0.5; 255168],
            sample_rate: 16000,
        };

        let padded = pad_audio(&audio, &config);

        // Left padding: 32 tokens
        assert_eq!(config.left_pad_samples(), 40960);

        // Total samples must match Python exactly
        assert_eq!(padded.samples.len(), 318720);

        // Total tokens: 249
        let tokens = num_audio_tokens(padded.samples.len(), &config);
        assert_eq!(tokens, 249);
        println!(
            "Padded to {} samples = {} tokens",
            padded.samples.len(),
            tokens
        );

        // Verify the audio starts at the right position
        assert_eq!(padded.samples[40960], 0.5); // First sample of original
        assert_eq!(padded.samples[40959], 0.0); // Last sample of left pad

        // Verify right padding is zeros
        let right_pad_start = 40960 + 255168;
        assert_eq!(padded.samples[right_pad_start], 0.0);
    }
}
