//! Sequential end-to-end benchmarks for Q4 GGUF inference pipeline.
//!
//! Benchmarks each stage of the Q4 pipeline in order:
//!   1. model_load  — GGUF parse + GPU upload
//!   2. preprocess  — WAV load, resample, chunk, mel spectrogram
//!   3. encode      — audio encoder + adapter (mel → embeddings)
//!   4. transcribe  — full encode + autoregressive decode (mel → tokens)
//!
//! Long audio is automatically chunked (max 1200 mel frames) to stay within
//! Metal's shared memory limits, matching the real CLI pipeline.
//!
//! Requires `models/voxtral-q4.gguf` — benchmarks are skipped if absent.
//!
//! Run:
//!   cargo bench --features "wgpu,cli,hub" q4_pipeline

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::{Path, PathBuf};
use std::time::Duration;

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

use voxtral_mini_realtime::audio::{
    chunk::{chunk_audio, needs_chunking, ChunkConfig},
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
    AudioBuffer,
};
use voxtral_mini_realtime::gguf::loader::Q4ModelLoader;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;

type Backend = Wgpu;

const GGUF_PATH: &str = "models/voxtral-q4.gguf";
const DELAY_TOKENS: usize = 6;
const MAX_MEL_FRAMES: usize = 1200;

struct AudioFixture {
    name: &'static str,
    path: &'static str,
}

const AUDIO_FILES: &[AudioFixture] = &[
    AudioFixture {
        name: "mary_3s",
        path: "test_data/mary_had_lamb.wav",
    },
    AudioFixture {
        name: "apollo11_13s",
        path: "test_data/examples_data_apollo11_one_small_step.wav",
    },
];

/// Preprocessed audio ready for inference — one mel tensor per chunk.
struct PreprocessedAudio {
    chunks: Vec<Tensor<Backend, 3>>,
}

/// Load audio, resample, chunk if needed, compute mel tensors.
fn preprocess_audio(
    audio_path: &str,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> PreprocessedAudio {
    let audio = load_wav(audio_path).expect("Failed to load audio");
    let mut audio = if audio.sample_rate != 16000 {
        resample_to_16k(&audio).expect("Failed to resample")
    } else {
        audio
    };
    audio.peak_normalize(0.95);

    let chunk_config = ChunkConfig::voxtral().with_max_frames(MAX_MEL_FRAMES);
    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());
    let pad_config = PadConfig::voxtral();

    let sample_chunks = if needs_chunking(audio.samples.len(), &chunk_config) {
        chunk_audio(&audio.samples, &chunk_config)
    } else {
        vec![voxtral_mini_realtime::audio::AudioChunk {
            samples: audio.samples.clone(),
            start_sample: 0,
            end_sample: audio.samples.len(),
            index: 0,
            is_last: true,
        }]
    };

    let chunks = sample_chunks
        .into_iter()
        .map(|chunk| {
            let chunk_audio = AudioBuffer::new(chunk.samples, audio.sample_rate);
            audio_to_mel(&chunk_audio, &mel_extractor, &pad_config, device)
        })
        .collect();

    PreprocessedAudio { chunks }
}

/// Convert an audio buffer to a mel spectrogram tensor.
fn audio_to_mel(
    audio: &AudioBuffer,
    mel_extractor: &MelSpectrogram,
    pad_config: &PadConfig,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Tensor<Backend, 3> {
    let padded = pad_audio(audio, pad_config);
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = mel[0].len();

    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }
    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
    Tensor::from_data(TensorData::new(mel_flat, [1, n_mels, n_frames]), device)
}

// ---------------------------------------------------------------------------
// Stage 1: Model loading
// ---------------------------------------------------------------------------

fn bench_model_load(c: &mut Criterion) {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("Skipping q4_pipeline benchmarks: {GGUF_PATH} not found");
        return;
    }

    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let mut group = c.benchmark_group("q4_pipeline/model_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("gguf_load", |b| {
        b.iter(|| {
            let path = PathBuf::from(GGUF_PATH);
            let mut loader = Q4ModelLoader::from_file(&path).expect("Failed to open GGUF");
            let model = loader.load(&device).expect("Failed to load Q4 model");
            let _ = model.encoder();
            model
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 2: Audio preprocessing (CPU-bound)
// ---------------------------------------------------------------------------

fn bench_preprocess(c: &mut Criterion) {
    if !Path::new(GGUF_PATH).exists() {
        return;
    }

    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let mut group = c.benchmark_group("q4_pipeline/preprocess");
    group.sample_size(20);

    for fixture in AUDIO_FILES {
        if !Path::new(fixture.path).exists() {
            eprintln!("Skipping {}: file not found", fixture.path);
            continue;
        }
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.name),
            &fixture.path,
            |b, &path| {
                b.iter(|| preprocess_audio(path, &device));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 3: Audio encoding (GPU — encoder + adapter, per chunk)
// ---------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    if !Path::new(GGUF_PATH).exists() {
        return;
    }

    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let path = PathBuf::from(GGUF_PATH);
    let mut loader = Q4ModelLoader::from_file(&path).expect("Failed to open GGUF");
    let model = loader.load(&device).expect("Failed to load Q4 model");

    let mut group = c.benchmark_group("q4_pipeline/encode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for fixture in AUDIO_FILES {
        if !Path::new(fixture.path).exists() {
            continue;
        }
        let preprocessed = preprocess_audio(fixture.path, &device);
        let n_chunks = preprocessed.chunks.len();

        // Benchmark encoding all chunks (label includes chunk count)
        let label = if n_chunks > 1 {
            format!("{}_{}chunks", fixture.name, n_chunks)
        } else {
            fixture.name.to_string()
        };

        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |b, _| {
            b.iter(|| {
                for mel in &preprocessed.chunks {
                    let embeds = model.encode_audio(mel.clone());
                    // GPU sync — force pipeline flush
                    embeds.clone().slice([0..1, 0..1, 0..1]).to_data();
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 4: Full transcription (encode + autoregressive decode, per chunk)
// ---------------------------------------------------------------------------

fn bench_transcribe(c: &mut Criterion) {
    if !Path::new(GGUF_PATH).exists() {
        return;
    }

    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let path = PathBuf::from(GGUF_PATH);
    let mut loader = Q4ModelLoader::from_file(&path).expect("Failed to open GGUF");
    let model = loader.load(&device).expect("Failed to load Q4 model");

    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(DELAY_TOKENS as f32, &device);

    let mut group = c.benchmark_group("q4_pipeline/transcribe");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for fixture in AUDIO_FILES {
        if !Path::new(fixture.path).exists() {
            continue;
        }
        let preprocessed = preprocess_audio(fixture.path, &device);
        let n_chunks = preprocessed.chunks.len();

        let label = if n_chunks > 1 {
            format!("{}_{}chunks", fixture.name, n_chunks)
        } else {
            fixture.name.to_string()
        };

        group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |b, _| {
            b.iter(|| {
                for mel in &preprocessed.chunks {
                    model.transcribe_streaming(mel.clone(), t_embed.clone());
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 5: Full E2E (preprocess + transcribe, all chunks)
// ---------------------------------------------------------------------------

fn bench_e2e(c: &mut Criterion) {
    if !Path::new(GGUF_PATH).exists() {
        return;
    }

    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let path = PathBuf::from(GGUF_PATH);
    let mut loader = Q4ModelLoader::from_file(&path).expect("Failed to open GGUF");
    let model = loader.load(&device).expect("Failed to load Q4 model");

    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(DELAY_TOKENS as f32, &device);

    let mut group = c.benchmark_group("q4_pipeline/e2e");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for fixture in AUDIO_FILES {
        if !Path::new(fixture.path).exists() {
            continue;
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.name),
            &fixture.path,
            |b, &audio_path| {
                b.iter(|| {
                    let preprocessed = preprocess_audio(audio_path, &device);
                    for mel in &preprocessed.chunks {
                        model.transcribe_streaming(mel.clone(), t_embed.clone());
                    }
                });
            },
        );
    }

    group.finish();
}

// Run stages in order: load → preprocess → encode → transcribe → e2e
criterion_group!(
    name = benches;
    config = Criterion::default().with_output_color(true);
    targets = bench_model_load, bench_preprocess, bench_encode, bench_transcribe, bench_e2e
);
criterion_main!(benches);
