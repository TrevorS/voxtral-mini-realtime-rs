//! Mel-spectrogram computation for Voxtral.
//!
//! Computes log mel spectrograms from audio samples using the Voxtral audio
//! input specifications (16kHz, 128 mel bins, hop=160, window=400).

use num_complex::Complex;
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sample rate of input audio (default: 16000)
    pub sample_rate: u32,
    /// FFT window size (default: 400)
    pub n_fft: usize,
    /// Hop length between frames (default: 160)
    pub hop_length: usize,
    /// Window length (defaults to n_fft)
    pub win_length: Option<usize>,
    /// Number of mel bands (default: 128)
    pub n_mels: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (defaults to sample_rate / 2)
    pub fmax: Option<f32>,
    /// Global log mel maximum for normalization (default: 1.5)
    pub log_mel_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: None,
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            log_mel_max: 1.5,
        }
    }
}

impl MelConfig {
    /// Voxtral-optimized configuration.
    pub fn voxtral() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: Some(400),
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            log_mel_max: 1.5,
        }
    }
}

/// Mel-spectrogram extractor.
pub struct MelSpectrogram {
    config: MelConfig,
    /// Precomputed mel filterbank
    mel_basis: Vec<Vec<f32>>,
    /// Precomputed Hann window
    window: Vec<f32>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram extractor with given configuration.
    pub fn new(config: MelConfig) -> Self {
        let win_length = config.win_length.unwrap_or(config.n_fft);
        let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

        let mel_basis = Self::create_mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            config.fmin,
            fmax,
        );

        let window = Self::hann_window(win_length);

        Self {
            config,
            mel_basis,
            window,
        }
    }

    /// Create a new extractor with Voxtral-optimized settings.
    pub fn voxtral() -> Self {
        Self::new(MelConfig::voxtral())
    }

    /// Get the configuration.
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Compute mel spectrogram from audio samples.
    ///
    /// Returns a 2D vector of shape `[n_frames, n_mels]`.
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let stft = self.stft(samples);

        // Compute power spectrogram
        let power_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|c| c.norm_sqr()).collect())
            .collect();

        self.apply_mel_filterbank(&power_spec)
    }

    /// Compute log mel spectrogram (the format Voxtral expects).
    ///
    /// Returns a 2D vector of shape `[n_frames, n_mels]` with log-compressed
    /// and normalized values.
    pub fn compute_log(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mel = self.compute(samples);
        let log_mel_max = self.config.log_mel_max;

        mel.into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|v| {
                        // Log compression with floor, then normalize by global max
                        let log_val = (v.max(1e-10)).ln();
                        // Clamp to reasonable range and normalize
                        (log_val / log_mel_max).clamp(-1.0, 1.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute log mel spectrogram and return as flat vector.
    ///
    /// Returns flattened data in row-major order `[n_frames * n_mels]`.
    pub fn compute_log_flat(&self, samples: &[f32]) -> Vec<f32> {
        self.compute_log(samples).into_iter().flatten().collect()
    }

    /// Number of frames for a given number of samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        let pad_length = (self.config.n_fft - self.config.hop_length) / 2;
        let padded_len = num_samples + 2 * pad_length;
        (padded_len - self.config.n_fft) / self.config.hop_length + 1
    }

    /// Short-time Fourier transform.
    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.window.len();

        // Reflect-pad signal (center=True behavior)
        let pad_length = (n_fft - hop_length) / 2;
        let mut padded = Vec::with_capacity(pad_length + samples.len() + pad_length);

        // Left reflect padding
        for i in (1..=pad_length).rev() {
            let idx = i.min(samples.len().saturating_sub(1));
            padded.push(samples.get(idx).copied().unwrap_or(0.0));
        }
        padded.extend_from_slice(samples);
        // Right reflect padding
        for i in 0..pad_length {
            let idx = samples.len().saturating_sub(2).saturating_sub(i);
            padded.push(samples.get(idx).copied().unwrap_or(0.0));
        }

        // Setup FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        let n_frames = (padded.len() - n_fft) / hop_length + 1;
        let mut result = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop_length;

            // Apply window and prepare FFT input
            let mut buffer: Vec<FftComplex<f32>> = (0..n_fft)
                .map(|j| {
                    let sample = if j < win_length && start + j < padded.len() {
                        padded[start + j] * self.window[j]
                    } else {
                        0.0
                    };
                    FftComplex::new(sample, 0.0)
                })
                .collect();

            // Perform FFT
            fft.process(&mut buffer);

            // Take positive frequencies only (n_fft/2 + 1)
            let frame: Vec<Complex<f32>> = buffer
                .iter()
                .take(n_fft / 2 + 1)
                .map(|c| Complex::new(c.re, c.im))
                .collect();

            result.push(frame);
        }

        result
    }

    /// Apply mel filterbank to power spectrogram.
    fn apply_mel_filterbank(&self, power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        power_spec
            .iter()
            .map(|frame| {
                self.mel_basis
                    .iter()
                    .map(|filter| filter.iter().zip(frame.iter()).map(|(f, p)| f * p).sum())
                    .collect()
            })
            .collect()
    }

    /// Convert frequency in Hz to mel scale (Slaney / O'Shaughnessy).
    fn hz_to_mel(f: f32) -> f32 {
        const F_SP: f32 = 200.0 / 3.0; // 66.667 Hz per mel below break
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
        const LOGSTEP: f32 = 0.068_751_74; // ln(6.4) / 27

        if f < MIN_LOG_HZ {
            f / F_SP
        } else {
            MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
        }
    }

    /// Convert mel value to Hz (Slaney / O'Shaughnessy).
    fn mel_to_hz(m: f32) -> f32 {
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
        const LOGSTEP: f32 = 0.068_751_74;

        if m < MIN_LOG_MEL {
            m * F_SP
        } else {
            MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOGSTEP).exp()
        }
    }

    /// Create mel filterbank matrix (matches librosa.filters.mel defaults).
    fn create_mel_filterbank(
        sample_rate: u32,
        n_fft: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;

        // Create linearly spaced mel points
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // FFT bin center frequencies
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
            .collect();

        // Build triangular filterbank
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

        for i in 0..n_mels {
            let f_lower = hz_points[i];
            let f_center = hz_points[i + 1];
            let f_upper = hz_points[i + 2];

            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= f_lower && freq <= f_center && f_center > f_lower {
                    filterbank[i][j] = (freq - f_lower) / (f_center - f_lower);
                } else if freq > f_center && freq <= f_upper && f_upper > f_center {
                    filterbank[i][j] = (f_upper - freq) / (f_upper - f_center);
                }
            }

            // Slaney area-normalization
            let band_width = hz_points[i + 2] - hz_points[i];
            if band_width > 0.0 {
                let enorm = 2.0 / band_width;
                for val in &mut filterbank[i] {
                    *val *= enorm;
                }
            }
        }

        filterbank
    }

    /// Create Hann window.
    fn hann_window(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / length as f32).cos()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_config_default() {
        let config = MelConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.n_mels, 128);
    }

    #[test]
    fn test_mel_config_voxtral() {
        let config = MelConfig::voxtral();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.n_mels, 128);
        assert!((config.log_mel_max - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_mel_spectrogram_creation() {
        let mel = MelSpectrogram::voxtral();
        assert_eq!(mel.config().n_mels, 128);
        assert_eq!(mel.mel_basis.len(), 128);
        assert_eq!(mel.mel_basis[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn test_hann_window() {
        let window = MelSpectrogram::hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_mel_silence() {
        let mel = MelSpectrogram::voxtral();
        let samples = vec![0.0f32; 16000];
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        assert_eq!(result[0].len(), 128);
        // Silence should produce very small values
        for frame in &result {
            for &val in frame {
                assert!(val < 1e-6);
            }
        }
    }

    #[test]
    fn test_compute_mel_sine_wave() {
        let mel = MelSpectrogram::voxtral();
        // Generate 440Hz sine wave (A4 note)
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        // Should have non-zero energy
        let total_energy: f32 = result.iter().flat_map(|f| f.iter()).sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_compute_log_mel() {
        let mel = MelSpectrogram::voxtral();
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let result = mel.compute_log(&samples);
        assert!(!result.is_empty());
        // Log mel values should be in [-1, 1] range after normalization
        for frame in &result {
            for &val in frame {
                assert!(val >= -1.0 && val <= 1.0, "Value out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_num_frames() {
        let mel = MelSpectrogram::voxtral();
        // 1 second of audio at 16kHz
        let n_frames = mel.num_frames(16000);
        // With hop=160, should get ~100 frames per second
        assert!(n_frames >= 99 && n_frames <= 101);
    }

    #[test]
    fn test_hz_mel_conversion() {
        // 1000 Hz should be at the mel scale break point
        let mel_1000 = MelSpectrogram::hz_to_mel(1000.0);
        let hz_back = MelSpectrogram::mel_to_hz(mel_1000);
        assert!((hz_back - 1000.0).abs() < 1.0);

        // Test low frequency
        let mel_100 = MelSpectrogram::hz_to_mel(100.0);
        let hz_100 = MelSpectrogram::mel_to_hz(mel_100);
        assert!((hz_100 - 100.0).abs() < 1.0);

        // Test high frequency
        let mel_8000 = MelSpectrogram::hz_to_mel(8000.0);
        let hz_8000 = MelSpectrogram::mel_to_hz(mel_8000);
        assert!((hz_8000 - 8000.0).abs() < 10.0);
    }
}
