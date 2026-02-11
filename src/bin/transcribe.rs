//! CLI for Voxtral transcription (f32 SafeTensors or Q4 GGUF).

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "voxtral-transcribe")]
#[command(about = "Transcribe audio using Voxtral Mini 4B Realtime")]
struct Cli {
    /// Path to audio file (WAV format). Can be specified multiple times for batch processing.
    #[arg(short, long, required_unless_present = "audio_list")]
    audio: Vec<String>,

    /// File containing audio paths (one per line). Loads model once, processes all files.
    #[arg(long, conflicts_with = "audio")]
    audio_list: Option<String>,

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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let device = Default::default();

    // Collect audio paths from --audio args or --audio-list file
    let audio_paths: Vec<String> = if let Some(list_path) = &cli.audio_list {
        std::fs::read_to_string(list_path)
            .with_context(|| format!("Failed to read audio list: {list_path}"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .collect()
    } else {
        cli.audio.clone()
    };

    if audio_paths.is_empty() {
        bail!("No audio files specified");
    }

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

    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());
    let pad_config = PadConfig::voxtral();
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(cli.delay as f32, &device);

    // Load model once
    let model_state = load_model(&cli, &device)?;

    for audio_path in &audio_paths {
        let text = transcribe_one(
            audio_path,
            &model_state,
            &tokenizer,
            &mel_extractor,
            &pad_config,
            &t_embed,
            &device,
        )?;
        println!("{text}");
    }
    Ok(())
}

/// Loaded model — either f32 or Q4.
#[allow(clippy::large_enum_variant)]
enum ModelState {
    F32 {
        model: voxtral_mini_realtime::models::voxtral::VoxtralModel<Backend>,
    },
    Q4 {
        model: voxtral_mini_realtime::gguf::model::Q4VoxtralModel,
    },
}

fn load_model(
    cli: &Cli,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<ModelState> {
    if let Some(gguf_path) = &cli.gguf {
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
        Ok(ModelState::Q4 { model })
    } else {
        use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
        let model_dir = PathBuf::from(&cli.model);
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
        let loader = VoxtralModelLoader::from_file(&safetensors_path)
            .context("Failed to open model weights")?;
        let model = loader.load(device).context("Failed to load model")?;
        info!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            "f32 model loaded"
        );
        Ok(ModelState::F32 { model })
    }
}

/// Preprocess one audio file and run inference against the already-loaded model.
fn transcribe_one(
    audio_path: &str,
    model_state: &ModelState,
    tokenizer: &VoxtralTokenizer,
    mel_extractor: &MelSpectrogram,
    pad_config: &PadConfig,
    t_embed: &Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<String> {
    let audio = load_wav(audio_path).with_context(|| format!("Failed to load {audio_path}"))?;

    let mut audio = if audio.sample_rate != 16000 {
        resample_to_16k(&audio).context("Failed to resample audio")?
    } else {
        audio
    };
    audio.peak_normalize(0.95);

    let padded = pad_audio(&audio, pad_config);
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
    let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
        device,
    );

    let generated = match model_state {
        ModelState::Q4 { model } => model.transcribe_streaming(mel_tensor, t_embed.clone()),
        ModelState::F32 { model } => {
            transcribe_f32_with_model(model, mel_tensor, t_embed.clone(), device)?
        }
    };

    let text_tokens: Vec<u32> = generated
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    tokenizer
        .decode(&text_tokens)
        .context("Failed to decode tokens")
}

/// f32 inference with an already-loaded model.
fn transcribe_f32_with_model(
    model: &voxtral_mini_realtime::models::voxtral::VoxtralModel<Backend>,
    mel_tensor: Tensor<Backend, 3>,
    t_embed: Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Vec<i32>> {
    use burn::tensor::Int;

    let audio_embeds = model.encode_audio(mel_tensor);
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];

    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN {
        return Ok(Vec::new());
    }

    let mut decoder_cache = model.create_decoder_cache_preallocated(seq_len, device);

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

    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}
