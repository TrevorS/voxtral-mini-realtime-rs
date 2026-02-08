//! End-to-end benchmark binary for Voxtral inference.
//!
//! Measures stage-level timing (preprocess, encode, decode) and computes
//! RTF (real-time factor) and tokens/sec.

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "e2e-bench")]
#[command(about = "End-to-end benchmark for Voxtral inference")]
struct Cli {
    /// Path to audio file (WAV format)
    #[arg(short, long)]
    audio: Vec<String>,

    /// Path to Q4 GGUF model file
    #[arg(long, requires = "tokenizer")]
    gguf: Option<String>,

    /// Path to f32 model directory
    #[arg(short, long, default_value = "models/voxtral", conflicts_with = "gguf")]
    model: String,

    /// Path to tokenizer JSON
    #[arg(long)]
    tokenizer: Option<String>,

    /// Delay in tokens (1 token = 80ms)
    #[arg(short, long, default_value = "6")]
    delay: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "1")]
    warmup: usize,

    /// Number of timed iterations
    #[arg(long, default_value = "3")]
    iterations: usize,

    /// Write JSON results to file
    #[arg(long)]
    json_output: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    audio_file: String,
    audio_duration_secs: f32,
    preprocess_ms: f64,
    encode_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    rtf: f64,
    decode_tokens: usize,
    tokens_per_sec: f64,
    peak_memory_kb: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    results: Vec<BenchmarkResult>,
    iterations: usize,
    warmup: usize,
    delay_tokens: usize,
}

/// Read peak RSS from /proc/self/status (Linux only).
fn peak_rss_kb() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok())
            })
        })
}

/// Preprocess audio: load, resample, pad, compute mel, build tensor.
fn preprocess_audio(
    audio_path: &str,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<(Tensor<Backend, 3>, f32)> {
    let audio = load_wav(audio_path).context("Failed to load audio")?;
    let audio_duration = audio.duration_secs();

    let audio = if audio.sample_rate != 16000 {
        resample_to_16k(&audio).context("Failed to resample")?
    } else {
        audio
    };

    let pad_config = PadConfig::voxtral();
    let padded = pad_audio(&audio, &pad_config);

    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

    if n_frames == 0 {
        bail!("Audio too short to produce mel frames");
    }

    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }
    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
    let mel_tensor: Tensor<Backend, 3> =
        Tensor::from_data(TensorData::new(mel_flat, [1, n_mels, n_frames]), device);

    Ok((mel_tensor, audio_duration))
}

/// Run Q4 GGUF benchmark with stage-level timing.
fn bench_q4(
    gguf_path: &str,
    audio_path: &str,
    delay: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<BenchmarkResult> {
    use voxtral_mini_realtime::gguf::loader::Q4ModelLoader;

    // Preprocess
    let preprocess_start = Instant::now();
    let (mel_tensor, audio_duration) = preprocess_audio(audio_path, device)?;
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    // Load model (not timed — amortized across iterations)
    let path = PathBuf::from(gguf_path);
    let mut loader = Q4ModelLoader::from_file(&path).context("Failed to open GGUF")?;
    let model = loader.load(device).context("Failed to load Q4 model")?;

    // Time embedding
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(delay as f32, device);

    // Encode
    let encode_start = Instant::now();
    let audio_embeds = model.encode_audio(mel_tensor);
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];
    // Force GPU sync
    let _ = audio_embeds.clone().slice([0..1, 0..1, 0..1]).to_data();
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

    // Decode (lifted from transcribe_streaming with timing)
    let decode_start = Instant::now();

    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);
    let prefix_inputs = prefix_audio + prefix_text_embeds;

    let mut decoder_cache = model.create_decoder_cache();

    let hidden = model.decoder().forward_hidden_with_cache(
        prefix_inputs,
        t_embed.clone(),
        &mut decoder_cache,
    );
    let logits = model.decoder().lm_head(hidden);

    let last_logits =
        logits
            .clone()
            .slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..logits.dims()[2]]);
    let first_pred = last_logits.argmax(2);
    let first_token: i32 = first_pred.into_scalar().elem();

    let mut generated = prefix;
    generated.push(first_token);

    for pos in (PREFIX_LEN + 1)..seq_len {
        let new_token = generated[pos - 1];
        let token_tensor =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![new_token], [1, 1]), device);
        let text_embed = model.decoder().embed_tokens(token_tensor);

        let audio_pos = audio_embeds
            .clone()
            .slice([0..1, (pos - 1)..pos, 0..d_model]);
        let input = audio_pos + text_embed;

        let hidden =
            model
                .decoder()
                .forward_hidden_with_cache(input, t_embed.clone(), &mut decoder_cache);
        let logits = model.decoder().lm_head(hidden);

        let pred = logits.argmax(2);
        let next_token: i32 = pred.into_scalar().elem();
        generated.push(next_token);
    }

    let decode_tokens = generated.len() - PREFIX_LEN;
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = preprocess_ms + encode_ms + decode_ms;
    let total_secs = total_ms / 1000.0;
    let rtf = total_secs / audio_duration as f64;
    let tokens_per_sec = if decode_ms > 0.0 {
        decode_tokens as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        audio_file: audio_path.to_string(),
        audio_duration_secs: audio_duration,
        preprocess_ms,
        encode_ms,
        decode_ms,
        total_ms,
        rtf,
        decode_tokens,
        tokens_per_sec,
        peak_memory_kb: peak_rss_kb(),
    })
}

/// Run f32 SafeTensors benchmark with stage-level timing.
fn bench_f32(
    model_dir: &str,
    audio_path: &str,
    delay: usize,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<BenchmarkResult> {
    use voxtral_mini_realtime::models::loader::VoxtralModelLoader;

    // Preprocess
    let preprocess_start = Instant::now();
    let (mel_tensor, audio_duration) = preprocess_audio(audio_path, device)?;
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    // Load model
    let model_dir_path = PathBuf::from(model_dir);
    let safetensors_path = model_dir_path.join("consolidated.safetensors");
    if !safetensors_path.exists() {
        bail!("Model not found at {}", safetensors_path.display());
    }

    let loader =
        VoxtralModelLoader::from_file(&safetensors_path).context("Failed to open model weights")?;
    let model = loader.load(device).context("Failed to load model")?;

    // Time embedding
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(delay as f32, device);

    // Encode
    let encode_start = Instant::now();
    let audio_embeds = model.encode_audio(mel_tensor);
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];
    let _ = audio_embeds.clone().slice([0..1, 0..1, 0..1]).to_data();
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

    // Decode
    let decode_start = Instant::now();

    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);
    let prefix_inputs = prefix_audio + prefix_text_embeds;

    let mut decoder_cache = model.create_decoder_cache();

    let hidden = model.decoder().forward_hidden_with_cache(
        prefix_inputs,
        t_embed.clone(),
        &mut decoder_cache,
    );
    let logits = model.decoder().lm_head(hidden);

    let vocab_size = logits.dims()[2];
    let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
    let first_pred = last_logits.argmax(2);
    let first_token: i32 = first_pred.into_scalar().elem();

    let mut generated = prefix;
    generated.push(first_token);

    for pos in PREFIX_LEN + 1..seq_len {
        let new_token = generated[pos - 1];
        let token_tensor =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![new_token], [1, 1]), device);
        let text_embed = model.decoder().embed_tokens(token_tensor);

        let audio_pos = audio_embeds
            .clone()
            .slice([0..1, (pos - 1)..pos, 0..d_model]);
        let input = audio_pos + text_embed;

        let hidden =
            model
                .decoder()
                .forward_hidden_with_cache(input, t_embed.clone(), &mut decoder_cache);
        let logits = model.decoder().lm_head(hidden);

        let pred = logits.argmax(2);
        let next_token: i32 = pred.into_scalar().elem();
        generated.push(next_token);
    }

    let decode_tokens = generated.len() - PREFIX_LEN;
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = preprocess_ms + encode_ms + decode_ms;
    let total_secs = total_ms / 1000.0;
    let rtf = total_secs / audio_duration as f64;
    let tokens_per_sec = if decode_ms > 0.0 {
        decode_tokens as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        audio_file: audio_path.to_string(),
        audio_duration_secs: audio_duration,
        preprocess_ms,
        encode_ms,
        decode_ms,
        total_ms,
        rtf,
        decode_tokens,
        tokens_per_sec,
        peak_memory_kb: peak_rss_kb(),
    })
}

fn print_table(results: &[BenchmarkResult]) {
    println!(
        "\n{:<30} {:>8} {:>10} {:>10} {:>10} {:>10} {:>6} {:>8} {:>10}",
        "Audio",
        "Dur (s)",
        "Pre (ms)",
        "Enc (ms)",
        "Dec (ms)",
        "Total(ms)",
        "RTF",
        "Tok/s",
        "RSS (KB)"
    );
    println!("{}", "-".repeat(115));

    for r in results {
        println!(
            "{:<30} {:>8.2} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>6.3} {:>8.1} {:>10}",
            r.audio_file.rsplit('/').next().unwrap_or(&r.audio_file),
            r.audio_duration_secs,
            r.preprocess_ms,
            r.encode_ms,
            r.decode_ms,
            r.total_ms,
            r.rtf,
            r.tokens_per_sec,
            r.peak_memory_kb
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
        );
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let cli = Cli::parse();
    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();

    if cli.audio.is_empty() {
        bail!("At least one --audio path is required");
    }

    let use_q4 = cli.gguf.is_some();
    let model_label = if use_q4 { "Q4 GGUF" } else { "f32 SafeTensors" };

    println!("Voxtral E2E Benchmark");
    println!("  Model: {model_label}");
    println!("  Delay: {} tokens ({}ms)", cli.delay, cli.delay * 80);
    println!("  Warmup: {}, Iterations: {}", cli.warmup, cli.iterations);
    println!("  Audio files: {}", cli.audio.len());

    // Warmup
    if cli.warmup > 0 {
        println!("\nWarmup...");
        for _ in 0..cli.warmup {
            for audio_path in &cli.audio {
                if let Some(ref gguf) = cli.gguf {
                    bench_q4(gguf, audio_path, cli.delay, &device)?;
                } else {
                    bench_f32(&cli.model, audio_path, cli.delay, &device)?;
                }
            }
        }
    }

    // Timed iterations
    println!("\nBenchmarking ({} iterations)...", cli.iterations);
    let mut all_results: Vec<Vec<BenchmarkResult>> = Vec::new();

    for iter in 0..cli.iterations {
        let mut iter_results = Vec::new();
        for audio_path in &cli.audio {
            let result = if let Some(ref gguf) = cli.gguf {
                bench_q4(gguf, audio_path, cli.delay, &device)?
            } else {
                bench_f32(&cli.model, audio_path, cli.delay, &device)?
            };
            iter_results.push(result);
        }
        println!("  Iteration {} complete", iter + 1);
        all_results.push(iter_results);
    }

    // Average results across iterations
    let mut averaged: Vec<BenchmarkResult> = Vec::new();
    let n_files = cli.audio.len();
    let n_iter = cli.iterations as f64;

    for file_idx in 0..n_files {
        let first = &all_results[0][file_idx];
        let mut avg = BenchmarkResult {
            audio_file: first.audio_file.clone(),
            audio_duration_secs: first.audio_duration_secs,
            preprocess_ms: 0.0,
            encode_ms: 0.0,
            decode_ms: 0.0,
            total_ms: 0.0,
            rtf: 0.0,
            decode_tokens: first.decode_tokens,
            tokens_per_sec: 0.0,
            peak_memory_kb: first.peak_memory_kb,
        };

        for iter_results in &all_results {
            let r = &iter_results[file_idx];
            avg.preprocess_ms += r.preprocess_ms;
            avg.encode_ms += r.encode_ms;
            avg.decode_ms += r.decode_ms;
            avg.total_ms += r.total_ms;
            avg.rtf += r.rtf;
            avg.tokens_per_sec += r.tokens_per_sec;
        }

        avg.preprocess_ms /= n_iter;
        avg.encode_ms /= n_iter;
        avg.decode_ms /= n_iter;
        avg.total_ms /= n_iter;
        avg.rtf /= n_iter;
        avg.tokens_per_sec /= n_iter;

        averaged.push(avg);
    }

    print_table(&averaged);

    // JSON output
    if let Some(ref json_path) = cli.json_output {
        let report = BenchmarkReport {
            results: averaged,
            iterations: cli.iterations,
            warmup: cli.warmup,
            delay_tokens: cli.delay,
        };
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(json_path, &json)?;
        println!("\nJSON results written to {json_path}");
    }

    Ok(())
}
