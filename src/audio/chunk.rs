//! Audio chunking for processing long audio files.
//!
//! Voxtral has a `max_source_positions` limit (default 1500 mel frames)
//! that constrains how much audio can be processed at once. This module
//! provides utilities to chunk audio appropriately.

/// Configuration for audio chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum mel frames per chunk (from max_source_positions)
    pub max_mel_frames: usize,
    /// Hop length for mel spectrogram (samples per frame)
    pub hop_length: usize,
    /// Sample rate of audio
    pub sample_rate: u32,
    /// Overlap between chunks in mel frames (for continuity)
    pub overlap_frames: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_mel_frames: 1500, // HuggingFace default
            hop_length: 160,
            sample_rate: 16000,
            overlap_frames: 0, // No overlap by default for streaming
        }
    }
}

impl ChunkConfig {
    /// Create config from Voxtral parameters.
    pub fn voxtral() -> Self {
        Self {
            max_mel_frames: 1500,
            hop_length: 160,
            sample_rate: 16000,
            overlap_frames: 0,
        }
    }

    /// Create config with custom max mel frames.
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_mel_frames = max_frames;
        self
    }

    /// Create config with overlap for better chunk boundaries.
    pub fn with_overlap(mut self, overlap_frames: usize) -> Self {
        self.overlap_frames = overlap_frames;
        self
    }

    /// Maximum audio samples per chunk.
    pub fn max_samples_per_chunk(&self) -> usize {
        self.max_mel_frames * self.hop_length
    }

    /// Step size in samples between chunk starts.
    pub fn step_samples(&self) -> usize {
        (self.max_mel_frames - self.overlap_frames) * self.hop_length
    }

    /// Maximum duration per chunk in seconds.
    pub fn max_duration_secs(&self) -> f32 {
        self.max_samples_per_chunk() as f32 / self.sample_rate as f32
    }
}

/// An audio chunk with metadata.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples for this chunk
    pub samples: Vec<f32>,
    /// Start position in original audio (samples)
    pub start_sample: usize,
    /// End position in original audio (samples)
    pub end_sample: usize,
    /// Chunk index (0-based)
    pub index: usize,
    /// Whether this is the last chunk
    pub is_last: bool,
}

impl AudioChunk {
    /// Duration of this chunk in seconds.
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
        self.samples.len() as f32 / sample_rate as f32
    }

    /// Start time in original audio (seconds).
    pub fn start_time(&self, sample_rate: u32) -> f32 {
        self.start_sample as f32 / sample_rate as f32
    }

    /// End time in original audio (seconds).
    pub fn end_time(&self, sample_rate: u32) -> f32 {
        self.end_sample as f32 / sample_rate as f32
    }
}

/// Iterator over audio chunks.
pub struct ChunkIterator<'a> {
    samples: &'a [f32],
    config: ChunkConfig,
    position: usize,
    index: usize,
}

impl<'a> ChunkIterator<'a> {
    /// Create a new chunk iterator.
    pub fn new(samples: &'a [f32], config: ChunkConfig) -> Self {
        Self {
            samples,
            config,
            position: 0,
            index: 0,
        }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = AudioChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.samples.len() {
            return None;
        }

        let start_sample = self.position;
        let max_end = start_sample + self.config.max_samples_per_chunk();
        let end_sample = max_end.min(self.samples.len());
        let is_last = end_sample >= self.samples.len();

        let chunk = AudioChunk {
            samples: self.samples[start_sample..end_sample].to_vec(),
            start_sample,
            end_sample,
            index: self.index,
            is_last,
        };

        // Move position by step size (accounts for overlap)
        self.position += self.config.step_samples();
        self.index += 1;

        Some(chunk)
    }
}

/// Chunk audio samples according to max_source_positions limit.
///
/// # Arguments
/// * `samples` - Audio samples at target sample rate
/// * `config` - Chunking configuration
///
/// # Returns
/// Vector of audio chunks
pub fn chunk_audio(samples: &[f32], config: &ChunkConfig) -> Vec<AudioChunk> {
    ChunkIterator::new(samples, config.clone()).collect()
}

/// Check if audio needs chunking.
pub fn needs_chunking(num_samples: usize, config: &ChunkConfig) -> bool {
    num_samples > config.max_samples_per_chunk()
}

/// Calculate number of chunks for given audio length.
pub fn num_chunks(num_samples: usize, config: &ChunkConfig) -> usize {
    if num_samples == 0 {
        return 0;
    }
    let step = config.step_samples();
    if step == 0 {
        return 1;
    }
    num_samples.div_ceil(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_defaults() {
        let config = ChunkConfig::default();
        assert_eq!(config.max_mel_frames, 1500);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.max_samples_per_chunk(), 240000); // 1500 * 160
        assert!((config.max_duration_secs() - 15.0).abs() < 0.01); // 15 seconds
    }

    #[test]
    fn test_no_chunking_needed() {
        let config = ChunkConfig::default();
        let samples = vec![0.0f32; 100000]; // ~6.25 seconds, under limit

        assert!(!needs_chunking(samples.len(), &config));

        let chunks = chunk_audio(&samples, &config);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
        assert_eq!(chunks[0].samples.len(), 100000);
    }

    #[test]
    fn test_chunking_needed() {
        let config = ChunkConfig::default();
        let samples = vec![0.0f32; 500000]; // ~31 seconds, needs chunking

        assert!(needs_chunking(samples.len(), &config));

        let chunks = chunk_audio(&samples, &config);
        assert_eq!(chunks.len(), 3); // 240000 + 240000 + 20000

        assert!(!chunks[0].is_last);
        assert!(!chunks[1].is_last);
        assert!(chunks[2].is_last);

        assert_eq!(chunks[0].start_sample, 0);
        assert_eq!(chunks[1].start_sample, 240000);
        assert_eq!(chunks[2].start_sample, 480000);
    }

    #[test]
    fn test_chunking_with_overlap() {
        let config = ChunkConfig::default().with_overlap(100); // 100 frame overlap
        let samples = vec![0.0f32; 500000];

        let chunks = chunk_audio(&samples, &config);

        // With overlap, step is (1500-100)*160 = 224000
        // So we need more chunks
        assert!(chunks.len() > 2);

        // Verify overlap exists
        let step = config.step_samples();
        assert_eq!(step, 224000);
    }

    #[test]
    fn test_chunk_times() {
        let config = ChunkConfig::default();
        let samples = vec![0.0f32; 500000];
        let chunks = chunk_audio(&samples, &config);

        let first = &chunks[0];
        assert!((first.start_time(16000) - 0.0).abs() < 0.001);
        assert!((first.duration_secs(16000) - 15.0).abs() < 0.01);

        let second = &chunks[1];
        assert!((second.start_time(16000) - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_num_chunks() {
        let config = ChunkConfig::default();

        assert_eq!(num_chunks(0, &config), 0);
        assert_eq!(num_chunks(100000, &config), 1); // Under limit
        assert_eq!(num_chunks(240000, &config), 1); // Exactly at limit
        assert_eq!(num_chunks(240001, &config), 2); // Just over
        assert_eq!(num_chunks(500000, &config), 3);
    }
}
