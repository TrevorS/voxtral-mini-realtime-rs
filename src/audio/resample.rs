//! Audio resampling utilities.
//!
//! Uses rubato for high-quality resampling to match Voxtral's expected 16kHz input.

use crate::audio::AudioBuffer;
use anyhow::{Context, Result};
use audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};

/// Resample audio to 16kHz (Voxtral's expected sample rate).
pub fn resample_to_16k(audio: &AudioBuffer) -> Result<AudioBuffer> {
    resample(audio, 16000)
}

/// Resample audio to a target sample rate.
pub fn resample(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if audio.sample_rate == target_rate {
        return Ok(audio.clone());
    }

    // Use FFT-based resampling for quality
    let mut resampler = Fft::<f32>::new(
        audio.sample_rate as usize,
        target_rate as usize,
        1024, // chunk size
        2,    // sub-chunks
        1,    // channels (mono)
        FixedSync::Input,
    )
    .context("Failed to create resampler")?;

    // Calculate required output buffer size
    let output_len = resampler.process_all_needed_output_len(audio.samples.len());

    // Create input buffer (interleaved format for 1 channel is just the samples)
    let input_buf = InterleavedOwned::new_from(audio.samples.clone(), 1, audio.samples.len())
        .context("Failed to create input buffer")?;

    // Create output buffer
    let mut output_buf = InterleavedOwned::new(0.0f32, 1, output_len);

    // Process all samples
    let (_, actual_output_len) = resampler
        .process_all_into_buffer(&input_buf, &mut output_buf, audio.samples.len(), None)
        .context("Failed to resample audio")?;

    // Extract the output samples
    let output = output_buf.take_data();
    let output = output[..actual_output_len].to_vec();

    Ok(AudioBuffer::new(output, target_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let audio = AudioBuffer::new(vec![0.5f32; 16000], 16000);
        let resampled = resample_to_16k(&audio).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        assert_eq!(resampled.len(), audio.len());
    }

    #[test]
    fn test_resample_downsample() {
        // 48kHz to 16kHz (3:1 ratio)
        let audio = AudioBuffer::new(vec![0.5f32; 48000], 48000);
        let resampled = resample_to_16k(&audio).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        // Should have approximately 1/3 the samples
        let expected_len = 16000;
        assert!(
            (resampled.len() as i64 - expected_len as i64).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_len,
            resampled.len()
        );
    }

    #[test]
    fn test_resample_upsample() {
        // 8kHz to 16kHz (1:2 ratio)
        let audio = AudioBuffer::new(vec![0.5f32; 8000], 8000);
        let resampled = resample_to_16k(&audio).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        // Should have approximately 2x the samples
        let expected_len = 16000;
        assert!(
            (resampled.len() as i64 - expected_len as i64).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_len,
            resampled.len()
        );
    }

    #[test]
    fn test_resample_preserves_duration() {
        let audio = AudioBuffer::new(vec![0.5f32; 24000], 24000); // 1 second
        let resampled = resample_to_16k(&audio).unwrap();
        // Duration should be approximately preserved
        assert!(
            (resampled.duration_secs() - 1.0).abs() < 0.02,
            "Duration changed significantly: {} -> {}",
            audio.duration_secs(),
            resampled.duration_secs()
        );
    }
}
