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

    /// Compute log mel spectrogram (Whisper-style normalization).
    ///
    /// Returns a 2D vector of shape `[n_frames, n_mels]` with log-compressed
    /// and normalized values in roughly [-1, 1] range.
    ///
    /// Normalization follows vLLM's Voxtral implementation:
    /// 1. log10(mel) with floor at 1e-10
    /// 2. Dynamic range limit using global_log_mel_max (or per-audio max if not set)
    /// 3. Linear scale: (log_spec + 4.0) / 4.0
    pub fn compute_log(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mel = self.compute(samples);

        // Step 1: log10 with floor
        let mut log_mel: Vec<Vec<f32>> = mel
            .into_iter()
            .map(|frame| frame.into_iter().map(|v| v.max(1e-10).log10()).collect())
            .collect();

        // Step 2: Apply dynamic range limit
        // Use global_log_mel_max if set (1.5 for Voxtral Realtime), otherwise compute from audio
        let log_spec_max = if self.config.log_mel_max > 0.0 {
            self.config.log_mel_max
        } else {
            log_mel
                .iter()
                .flat_map(|frame| frame.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let min_val = log_spec_max - 8.0;

        for frame in &mut log_mel {
            for v in frame.iter_mut() {
                *v = v.max(min_val);
            }
        }

        // Step 3: Linear scale to roughly [-1, 1]
        // Note: vLLM doesn't clamp, but Whisper does. Following vLLM for Voxtral.
        for frame in &mut log_mel {
            for v in frame.iter_mut() {
                *v = (*v + 4.0) / 4.0;
            }
        }

        log_mel
    }

    /// Compute log mel spectrogram and return as flat vector.
    ///
    /// Returns flattened data in row-major order `[n_frames * n_mels]`.
    pub fn compute_log_flat(&self, samples: &[f32]) -> Vec<f32> {
        self.compute_log(samples).into_iter().flatten().collect()
    }

    /// Number of frames for a given number of samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        // Matches torch.stft center=True behavior, minus 1 to match Voxtral's
        // magnitudes = stft[..., :-1] which drops the last frame
        let pad_length = self.config.n_fft / 2;
        let padded_len = num_samples + 2 * pad_length;
        // Drop last frame to match Python reference
        (padded_len - self.config.n_fft) / self.config.hop_length
    }

    /// Short-time Fourier transform.
    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.window.len();

        // Reflect-pad signal (center=True behavior, matching torch.stft)
        // torch.stft pads by n_fft//2 on each side
        let pad_length = n_fft / 2;
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

        // Drop last frame to match Python's magnitudes = stft[..., :-1]
        let n_frames = (padded.len() - n_fft) / hop_length;
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

    /// Create Hann window (periodic mode, matching torch.hann_window default).
    ///
    /// Uses periodic formula: 0.5 * (1 - cos(2*pi*n/N)) for n in [0, N)
    /// This matches torch.hann_window(N, periodic=True) which is the default.
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
        // For periodic window of length 4: [0, 0.5, 1, 0.5]
        // For symmetric: [0, 0.75, 0.75, 0]
        // Check we have periodic
        assert!(
            (window[2] - 1.0).abs() < 1e-6,
            "Expected window[2]=1.0 (periodic), got {}",
            window[2]
        );

        // Test 400-length window matches torch periodic
        let window400 = MelSpectrogram::hann_window(400);
        // torch.hann_window(400, periodic=True)[1] = 0.0000616908
        println!("window[1] = {:.10}", window400[1]);
        assert!(
            (window400[1] - 6.1690807e-05).abs() < 1e-8,
            "Window should match torch periodic, got {}",
            window400[1]
        );
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
        // Log mel values should be roughly in [-1, 2] range after normalization
        // (not clamped, per vLLM implementation)
        for frame in &result {
            for &val in frame {
                assert!(
                    val >= -2.0 && val <= 3.0,
                    "Value out of expected range: {}",
                    val
                );
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

    #[test]
    fn test_filterbank_matches_python() {
        use ndarray::ArrayD;
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let reference_path = "test_data/mel_filterbank_reference.npy";
        if !std::path::Path::new(reference_path).exists() {
            println!("Skipping filterbank comparison - reference file not found");
            return;
        }

        let file = File::open(reference_path).expect("Failed to open reference");
        // Reference may be float64
        let py_fb: ArrayD<f64> = ArrayD::<f64>::read_npy(file).expect("Failed to read reference");

        // Python shape is [201, 128], we need [128, 201]
        let mel = MelSpectrogram::voxtral();
        assert_eq!(mel.mel_basis.len(), 128);
        assert_eq!(mel.mel_basis[0].len(), 201);

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut count = 0usize;

        for (mel_idx, filter) in mel.mel_basis.iter().enumerate() {
            for (freq_idx, &rust_val) in filter.iter().enumerate() {
                // Python is [freq, mel], transposed
                let py_val = py_fb[[freq_idx, mel_idx]] as f32;
                let diff = (rust_val - py_val).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as f64;
                count += 1;
            }
        }

        let mean_diff = sum_diff / count as f64;
        println!(
            "Filterbank comparison: max_diff={:.6}, mean_diff={:.6}",
            max_diff, mean_diff
        );

        // Our filterbank should match Python's within floating point precision
        assert!(
            max_diff < 1e-3,
            "Filterbank max diff {} exceeds tolerance",
            max_diff
        );
    }

    #[test]
    fn test_mel_spectrogram_matches_python() {
        use crate::audio::{io::load_wav, pad::pad_audio, pad::PadConfig};
        use ndarray::ArrayD;
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let audio_path = "test_data/mary_had_lamb.wav";
        let reference_path = "test_data/reference_mel_padded.npy";

        if !std::path::Path::new(audio_path).exists()
            || !std::path::Path::new(reference_path).exists()
        {
            println!("Skipping mel comparison - required files not found");
            return;
        }

        // Load and pad audio
        let audio = load_wav(audio_path).expect("Failed to load audio");
        let pad_config = PadConfig::voxtral();
        let padded = pad_audio(&audio, &pad_config);

        // Compute mel
        let mel = MelSpectrogram::voxtral();
        let rust_mel = mel.compute_log(&padded.samples);

        // Load Python reference (shape [128, 1992])
        let file = File::open(reference_path).expect("Failed to open reference");
        let py_mel: ArrayD<f32> = ArrayD::<f32>::read_npy(file).expect("Failed to read reference");

        let n_frames = rust_mel.len();
        let n_mels = if n_frames > 0 { rust_mel[0].len() } else { 0 };

        println!("Rust mel shape: [{}, {}]", n_mels, n_frames);
        println!("Python mel shape: {:?}", py_mel.shape());

        // Compare (note: Rust is [frames, mels], Python is [mels, frames])
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut count = 0usize;
        let mut max_diff_loc = (0, 0);

        for (frame_idx, frame) in rust_mel.iter().enumerate() {
            for (mel_idx, &rust_val) in frame.iter().enumerate() {
                if frame_idx >= py_mel.shape()[1] || mel_idx >= py_mel.shape()[0] {
                    continue;
                }
                let py_val = py_mel[[mel_idx, frame_idx]];
                let diff = (rust_val - py_val).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_loc = (mel_idx, frame_idx);
                }
                sum_diff += diff as f64;
                count += 1;
            }
        }

        let mean_diff = sum_diff / count as f64;
        println!(
            "Mel comparison: max_diff={:.6} at {:?}, mean_diff={:.6}",
            max_diff, max_diff_loc, mean_diff
        );

        // Check some specific values
        println!("Sample values at frame 500:");
        println!("  Rust mel[500][64] = {:.6}", rust_mel[500][64]);
        println!("  Python mel[64, 500] = {:.6}", py_mel[[64, 500]]);

        // Should match within reasonable tolerance
        assert!(
            max_diff < 0.01,
            "Mel max diff {:.6} exceeds tolerance",
            max_diff
        );
    }
}
