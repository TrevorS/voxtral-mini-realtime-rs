//! Audio processing for Voxtral.
//!
//! Handles WAV I/O, resampling, mel spectrogram computation, and chunking.

pub mod chunk;
pub mod io;
pub mod mel;
pub mod pad;
pub mod resample;

pub use chunk::{chunk_audio, needs_chunking, AudioChunk, ChunkConfig, ChunkIterator};
pub use io::{load_wav, save_wav, AudioBuffer};
pub use mel::{MelConfig, MelSpectrogram};
pub use pad::{num_audio_tokens, pad_audio, PadConfig};
pub use resample::resample_to_16k;
