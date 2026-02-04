//! Audio processing for Voxtral.
//!
//! Handles WAV I/O, resampling, and mel spectrogram computation.

pub mod io;
pub mod mel;
pub mod resample;

pub use io::{load_wav, save_wav, AudioBuffer};
pub use mel::{MelConfig, MelSpectrogram};
pub use resample::resample_to_16k;
