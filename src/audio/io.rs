//! Audio I/O utilities.
//!
//! Handles loading and saving WAV files with automatic format conversion.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Audio buffer with samples and metadata.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples (mono, normalized to [-1.0, 1.0])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create an empty buffer at the given sample rate.
    pub fn empty(sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f32 {
        self.duration_secs() * 1000.0
    }

    /// Append samples from another buffer (must have same sample rate).
    pub fn append(&mut self, other: &AudioBuffer) -> Result<()> {
        if self.sample_rate != other.sample_rate {
            anyhow::bail!(
                "Sample rate mismatch: {} vs {}",
                self.sample_rate,
                other.sample_rate
            );
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    /// Save to WAV file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_wav(self, path)
    }
}

/// Load a WAV file and return as mono f32 samples.
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;

            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
                .chunks(channels)
                .map(|chunk| {
                    // Mix to mono by averaging channels
                    let sum: i32 = chunk.iter().sum();
                    (sum as f32 / channels as f32) / max_val
                })
                .collect()
        }
        SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
                .chunks(channels)
                .map(|chunk| {
                    // Mix to mono by averaging channels
                    chunk.iter().sum::<f32>() / channels as f32
                })
                .collect()
        }
    };

    Ok(AudioBuffer::new(samples, sample_rate))
}

/// Save audio buffer to WAV file (16-bit PCM).
pub fn save_wav<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
    let path = path.as_ref();
    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file: {}", path.display()))?;

    for &sample in &audio.samples {
        // Clamp and convert to i16
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer.write_sample(i16_sample)?;
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_audio_buffer_new() {
        let samples = vec![0.0f32; 16000];
        let buffer = AudioBuffer::new(samples, 16000);
        assert_eq!(buffer.len(), 16000);
        assert_eq!(buffer.sample_rate, 16000);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_audio_buffer_duration() {
        let samples = vec![0.0f32; 16000];
        let buffer = AudioBuffer::new(samples, 16000);
        assert!((buffer.duration_secs() - 1.0).abs() < 1e-6);
        assert!((buffer.duration_ms() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_audio_buffer_empty() {
        let buffer = AudioBuffer::empty(16000);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_audio_buffer_append() {
        let mut buffer1 = AudioBuffer::new(vec![0.5f32; 100], 16000);
        let buffer2 = AudioBuffer::new(vec![0.3f32; 50], 16000);
        buffer1.append(&buffer2).unwrap();
        assert_eq!(buffer1.len(), 150);
    }

    #[test]
    fn test_audio_buffer_append_rate_mismatch() {
        let mut buffer1 = AudioBuffer::new(vec![0.5f32; 100], 16000);
        let buffer2 = AudioBuffer::new(vec![0.3f32; 50], 24000);
        assert!(buffer1.append(&buffer2).is_err());
    }

    #[test]
    fn test_save_and_load_wav() {
        let original = AudioBuffer::new(
            (0..1600)
                .map(|i| (i as f32 * 0.01 * std::f32::consts::PI).sin())
                .collect(),
            16000,
        );

        let tmp = NamedTempFile::new().unwrap();
        original.save(tmp.path()).unwrap();
        let loaded = load_wav(tmp.path()).unwrap();

        assert_eq!(loaded.sample_rate, original.sample_rate);
        assert_eq!(loaded.len(), original.len());

        // Check samples are approximately equal (16-bit quantization introduces error)
        for (a, b) in original.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 0.001, "Sample mismatch: {} vs {}", a, b);
        }
    }
}
