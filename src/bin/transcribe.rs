//! CLI for Voxtral transcription (f32 SafeTensors or Q4 GGUF).

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use clap::Parser;
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::info;

use voxtral_mini_realtime::audio::{
    chunk::{chunk_audio, needs_chunking, ChunkConfig},
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
    AudioBuffer,
};
use voxtral_mini_realtime::gguf::model::Q4VoxtralModel;
use voxtral_mini_realtime::models::voxtral::VoxtralModel;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "voxtral-transcribe")]
#[command(about = "Transcribe audio using Voxtral Mini 4B Realtime")]
struct Cli {
    /// Path to audio file (WAV format)
    #[arg(short, long)]
    audio: String,

    /// Path to f32 model directory (containing consolidated.safetensors and tekken.json)
    #[arg(short, long, default_value = "models/voxtral", conflicts_with = "gguf")]
    model: String,

    /// Path to Q4 GGUF model file (use instead of --model for quantized inference)
    #[arg(long, conflicts_with = "model", requires = "tokenizer")]
    gguf: Option<String>,

    /// Path to tokenizer JSON (defaults to <model>/tekken.json)
    #[arg(long)]
    tokenizer: Option<String>,

    /// Delay in tokens (1 token = 80ms)
    #[arg(short, long, default_value = "6")]
    delay: usize,

    /// Max mel frames per chunk (lower is safer for smaller Apple GPUs)
    #[arg(long, default_value_t = 1200)]
    max_mel_frames: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let cli = Cli::parse();
    let device = Default::default();

    // Resolve tokenizer path
    let tokenizer_path = match &cli.tokenizer {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(&cli.model).join("tekken.json"),
    };
    if !tokenizer_path.exists() {
        bail!("Tokenizer not found at {}", tokenizer_path.display());
    }

    info!("Loading tokenizer from {}", tokenizer_path.display());
    let tokenizer =
        VoxtralTokenizer::from_file(&tokenizer_path).context("Failed to load tokenizer")?;

    // Load and preprocess audio
    let start = Instant::now();
    info!(path = %cli.audio, "Loading audio");
    let audio = load_wav(&cli.audio).context("Failed to load audio")?;
    info!(
        duration_secs = format!("{:.2}", audio.duration_secs()),
        sample_rate = audio.sample_rate,
        samples = audio.samples.len(),
        "Audio loaded"
    );

    let audio = if audio.sample_rate != 16000 {
        info!("Resampling to 16 kHz");
        resample_to_16k(&audio).context("Failed to resample audio")?
    } else {
        audio
    };

    if cli.max_mel_frames == 0 {
        bail!("--max-mel-frames must be greater than 0");
    }

    // We add left/right streaming pad before mel extraction, so effective mel
    // frames per chunk are higher than raw chunk frames.
    let chunk_config = ChunkConfig::voxtral().with_max_frames(cli.max_mel_frames);
    let chunks = if needs_chunking(audio.samples.len(), &chunk_config) {
        let chunks = chunk_audio(&audio.samples, &chunk_config);
        info!(
            total_chunks = chunks.len(),
            max_mel_frames = cli.max_mel_frames,
            max_chunk_duration_secs = format!("{:.2}", chunk_config.max_duration_secs()),
            "Audio exceeds chunk limit; transcribing in chunks"
        );
        chunks
    } else {
        vec![voxtral_mini_realtime::audio::AudioChunk {
            samples: audio.samples.clone(),
            start_sample: 0,
            end_sample: audio.samples.len(),
            index: 0,
            is_last: true,
        }]
    };

    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Audio preprocessing complete"
    );

    let pad_config = PadConfig::voxtral();
    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());

    // Time embedding (same for all chunks)
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(cli.delay as f32, &device);

    // Dispatch to f32 or Q4 path (model loaded once, reused across chunks).
    // Convert known kernel panics into actionable CLI guidance.
    let chunk_texts = run_transcription_with_chunk_hint(cli.max_mel_frames, || {
        if let Some(gguf_path) = &cli.gguf {
            transcribe_chunks_q4(
                gguf_path,
                &chunks,
                audio.sample_rate,
                &mel_extractor,
                &pad_config,
                t_embed,
                &device,
                &tokenizer,
            )
        } else {
            transcribe_chunks_f32(
                &cli.model,
                &chunks,
                audio.sample_rate,
                &mel_extractor,
                &pad_config,
                t_embed,
                &device,
                &tokenizer,
            )
        }
    })?;

    let text = chunk_texts.join(" ").trim().to_string();

    println!("{}", text);
    Ok(())
}

fn format_duration_hms(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{hours:02}:{mins:02}:{secs:02}")
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    "unknown panic payload".to_string()
}

fn chunk_hint_for_panic(panic_msg: &str, max_mel_frames: usize) -> Option<String> {
    let is_shared_mem_kernel_panic = panic_msg.contains("Unable to launch matmul")
        || panic_msg.contains("shared memory bytes")
        || panic_msg.contains("hardware limit is");

    if !is_shared_mem_kernel_panic {
        return None;
    }

    let suggested = if max_mel_frames > 1000 {
        1000
    } else if max_mel_frames > 900 {
        900
    } else if max_mel_frames > 800 {
        800
    } else if max_mel_frames > 700 {
        700
    } else {
        600
    };

    Some(format!(
        "GPU kernel launch failed due to shared-memory limits.\n\
         Try a smaller chunk size, e.g. `--max-mel-frames {suggested}` (current: {max_mel_frames}).\n\
         On Apple GPUs (M1/M2/M3), smaller chunk sizes are often required."
    ))
}

fn run_transcription_with_chunk_hint<F>(max_mel_frames: usize, transcribe: F) -> Result<Vec<String>>
where
    F: FnOnce() -> Result<Vec<String>>,
{
    match catch_unwind(AssertUnwindSafe(transcribe)) {
        Ok(result) => result,
        Err(payload) => {
            let panic_msg = panic_payload_to_string(payload.as_ref());
            if let Some(hint) = chunk_hint_for_panic(&panic_msg, max_mel_frames) {
                bail!("{hint}\nOriginal panic: {panic_msg}");
            }
            bail!("Transcription panicked: {panic_msg}");
        }
    }
}

fn mel_tensor_from_audio(
    audio: &AudioBuffer,
    mel_extractor: &MelSpectrogram,
    pad_config: &PadConfig,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Tensor<Backend, 3>> {
    let padded = pad_audio(audio, pad_config);
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

    if n_frames == 0 {
        bail!("Audio too short to produce mel frames");
    }
    info!(frames = n_frames, bins = n_mels, "Mel spectrogram computed");

    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }
    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();

    Ok(Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
        device,
    ))
}

fn decode_generated_text(tokenizer: &VoxtralTokenizer, generated: &[i32]) -> Result<String> {
    let text_tokens: Vec<u32> = generated
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    tokenizer
        .decode(&text_tokens)
        .context("Failed to decode tokens")
}

/// f32 SafeTensors model loader.
fn load_f32_model(
    model_dir: &str,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<VoxtralModel<Backend>> {
    use voxtral_mini_realtime::models::loader::VoxtralModelLoader;

    let model_dir = PathBuf::from(model_dir);
    let safetensors_path = model_dir.join("consolidated.safetensors");

    if !safetensors_path.exists() {
        bail!(
            "Model not found at {}\nRun: hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir {}",
            safetensors_path.display(),
            model_dir.display()
        );
    }

    let start = Instant::now();
    info!("Loading f32 model");
    let loader =
        VoxtralModelLoader::from_file(&safetensors_path).context("Failed to open model weights")?;
    let model: VoxtralModel<Backend> = loader.load(device).context("Failed to load model")?;
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "f32 model loaded"
    );

    Ok(model)
}

/// f32 inference on one mel tensor.
fn transcribe_f32_with_model(
    model: &VoxtralModel<Backend>,
    mel_tensor: Tensor<Backend, 3>,
    t_embed: Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Vec<i32>> {
    use burn::tensor::Int;

    let audio_embeds = model.encode_audio(mel_tensor);
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];
    info!(tokens = seq_len, "Audio encoded");

    let start = Instant::now();
    info!("Decoding");

    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN {
        info!(
            tokens = seq_len,
            required = PREFIX_LEN,
            "Audio too short for decoding"
        );
        return Ok(Vec::new());
    }

    let mut decoder_cache = model.create_decoder_cache();

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
        burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

    let prefix_inputs = prefix_audio + prefix_text_embeds;
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
        let token_tensor = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![new_token], [1, 1]),
            device,
        );
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

    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        tokens = generated.len() - PREFIX_LEN,
        "Decoding complete"
    );

    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}

/// Q4 GGUF model loader.
fn load_q4_model(
    gguf_path: &str,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Q4VoxtralModel> {
    use voxtral_mini_realtime::gguf::loader::Q4ModelLoader;

    let path = PathBuf::from(gguf_path);
    if !path.exists() {
        bail!("GGUF model not found at {}", path.display());
    }

    let start = Instant::now();
    info!("Loading Q4 GGUF model");
    let mut loader = Q4ModelLoader::from_file(&path).context("Failed to open GGUF")?;
    let model = loader.load(device).context("Failed to load Q4 model")?;
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Q4 model loaded"
    );

    Ok(model)
}

/// Q4 inference on one mel tensor.
fn transcribe_q4_with_model(
    model: &Q4VoxtralModel,
    mel_tensor: Tensor<Backend, 3>,
    t_embed: Tensor<Backend, 3>,
) -> Vec<i32> {
    let start = Instant::now();
    info!("Encoding + decoding");
    let generated = model.transcribe_streaming(mel_tensor, t_embed);
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        tokens = generated.len(),
        "Q4 inference complete"
    );

    generated
}

fn transcribe_chunks_q4(
    gguf_path: &str,
    chunks: &[voxtral_mini_realtime::audio::AudioChunk],
    sample_rate: u32,
    mel_extractor: &MelSpectrogram,
    pad_config: &PadConfig,
    t_embed: Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    tokenizer: &VoxtralTokenizer,
) -> Result<Vec<String>> {
    let model = load_q4_model(gguf_path, device)?;
    let mut texts = Vec::new();
    let total_chunks = chunks.len();
    let transcription_start = Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = Instant::now();
        let done = i;
        let avg_chunk_secs = if done > 0 {
            transcription_start.elapsed().as_secs_f64() / done as f64
        } else {
            0.0
        };
        let remaining = total_chunks.saturating_sub(done);
        let eta = Duration::from_secs_f64(avg_chunk_secs * remaining as f64);

        info!(
            chunk = format!("{}/{}", i + 1, total_chunks),
            start_sec = format!("{:.2}", chunk.start_time(sample_rate)),
            end_sec = format!("{:.2}", chunk.end_time(sample_rate)),
            elapsed = format_duration_hms(transcription_start.elapsed()),
            eta = format_duration_hms(eta),
            "Transcribing chunk"
        );
        let chunk_audio = AudioBuffer::new(chunk.samples.clone(), sample_rate);
        let mel_tensor = mel_tensor_from_audio(&chunk_audio, mel_extractor, pad_config, device)?;
        let generated = transcribe_q4_with_model(&model, mel_tensor, t_embed.clone());
        let text = decode_generated_text(tokenizer, &generated)?;
        if !text.trim().is_empty() {
            texts.push(text.trim().to_string());
        }

        let completed = i + 1;
        let avg_chunk_secs = transcription_start.elapsed().as_secs_f64() / completed as f64;
        let remaining = total_chunks.saturating_sub(completed);
        let eta = Duration::from_secs_f64(avg_chunk_secs * remaining as f64);
        info!(
            chunk = format!("{}/{}", completed, total_chunks),
            chunk_time = format_duration_hms(chunk_start.elapsed()),
            elapsed = format_duration_hms(transcription_start.elapsed()),
            eta = format_duration_hms(eta),
            "Chunk complete"
        );
    }

    info!(
        total_chunks,
        total_time = format_duration_hms(transcription_start.elapsed()),
        "Transcription complete"
    );

    Ok(texts)
}

fn transcribe_chunks_f32(
    model_dir: &str,
    chunks: &[voxtral_mini_realtime::audio::AudioChunk],
    sample_rate: u32,
    mel_extractor: &MelSpectrogram,
    pad_config: &PadConfig,
    t_embed: Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    tokenizer: &VoxtralTokenizer,
) -> Result<Vec<String>> {
    let model = load_f32_model(model_dir, device)?;
    let mut texts = Vec::new();
    let total_chunks = chunks.len();
    let transcription_start = Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = Instant::now();
        let done = i;
        let avg_chunk_secs = if done > 0 {
            transcription_start.elapsed().as_secs_f64() / done as f64
        } else {
            0.0
        };
        let remaining = total_chunks.saturating_sub(done);
        let eta = Duration::from_secs_f64(avg_chunk_secs * remaining as f64);

        info!(
            chunk = format!("{}/{}", i + 1, total_chunks),
            start_sec = format!("{:.2}", chunk.start_time(sample_rate)),
            end_sec = format!("{:.2}", chunk.end_time(sample_rate)),
            elapsed = format_duration_hms(transcription_start.elapsed()),
            eta = format_duration_hms(eta),
            "Transcribing chunk"
        );
        let chunk_audio = AudioBuffer::new(chunk.samples.clone(), sample_rate);
        let mel_tensor = mel_tensor_from_audio(&chunk_audio, mel_extractor, pad_config, device)?;
        let generated = transcribe_f32_with_model(&model, mel_tensor, t_embed.clone(), device)?;
        let text = decode_generated_text(tokenizer, &generated)?;
        if !text.trim().is_empty() {
            texts.push(text.trim().to_string());
        }

        let completed = i + 1;
        let avg_chunk_secs = transcription_start.elapsed().as_secs_f64() / completed as f64;
        let remaining = total_chunks.saturating_sub(completed);
        let eta = Duration::from_secs_f64(avg_chunk_secs * remaining as f64);
        info!(
            chunk = format!("{}/{}", completed, total_chunks),
            chunk_time = format_duration_hms(chunk_start.elapsed()),
            elapsed = format_duration_hms(transcription_start.elapsed()),
            eta = format_duration_hms(eta),
            "Chunk complete"
        );
    }

    info!(
        total_chunks,
        total_time = format_duration_hms(transcription_start.elapsed()),
        "Transcription complete"
    );

    Ok(texts)
}
