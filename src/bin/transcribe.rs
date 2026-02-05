//! CLI for Voxtral transcription.

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use clap::Parser;
use std::path::PathBuf;

use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::models::voxtral::VoxtralModel;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "voxtral-transcribe")]
#[command(about = "Transcribe audio using Voxtral Mini 4B Realtime")]
struct Cli {
    /// Path to audio file (WAV format)
    #[arg(short, long)]
    audio: String,

    /// Path to model directory (containing consolidated.safetensors and tekken.json)
    #[arg(short, long, default_value = "models/voxtral")]
    model: String,

    /// Delay in tokens (1 token = 80ms)
    #[arg(short, long, default_value = "6")]
    delay: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Default::default();

    let model_dir = PathBuf::from(&cli.model);
    let safetensors_path = model_dir.join("consolidated.safetensors");
    let tokenizer_path = model_dir.join("tekken.json");

    if !safetensors_path.exists() {
        bail!(
            "Model not found at {}\nRun: hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir {}",
            safetensors_path.display(),
            model_dir.display()
        );
    }
    if !tokenizer_path.exists() {
        bail!("Tokenizer not found at {}", tokenizer_path.display());
    }

    // Load tokenizer
    eprintln!("Loading tokenizer...");
    let tokenizer =
        VoxtralTokenizer::from_file(&tokenizer_path).context("Failed to load tokenizer")?;

    // Load model
    eprintln!("Loading model...");
    let loader =
        VoxtralModelLoader::from_file(&safetensors_path).context("Failed to open model weights")?;
    let model = loader
        .load::<Backend>(&device)
        .context("Failed to load model")?;

    // Load and preprocess audio
    eprintln!("Loading audio: {}", cli.audio);
    let audio = load_wav(&cli.audio).context("Failed to load audio")?;
    eprintln!(
        "  {:.2}s, {} Hz, {} samples",
        audio.duration_secs(),
        audio.sample_rate,
        audio.samples.len()
    );

    let audio = if audio.sample_rate != 16000 {
        eprintln!("  Resampling to 16 kHz...");
        resample_to_16k(&audio).context("Failed to resample audio")?
    } else {
        audio
    };

    // Pad and extract mel spectrogram
    let pad_config = PadConfig::voxtral();
    let padded = pad_audio(&audio, &pad_config);

    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

    if n_frames == 0 {
        bail!("Audio too short to produce mel frames");
    }
    eprintln!("  Mel: {} frames x {} bins", n_frames, n_mels);

    // Transpose to [n_mels, n_frames] and build tensor
    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }
    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
    let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
        &device,
    );

    // Encode audio
    eprintln!("Encoding audio...");
    let audio_embeds = model.encode_audio(mel_tensor);
    let seq_len = audio_embeds.dims()[1];
    eprintln!("  {} audio tokens", seq_len);

    // Time embedding
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(cli.delay as f32, &device);

    // Decode
    eprintln!("Decoding...");
    let generated = transcribe_with_cache(&model, audio_embeds, t_embed, &device);

    // Filter control tokens and decode to text
    let text_tokens: Vec<u32> = generated
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    let text = tokenizer
        .decode(&text_tokens)
        .context("Failed to decode tokens")?;

    println!("{}", text);
    Ok(())
}

/// Autoregressive transcription with KV cache.
fn transcribe_with_cache(
    model: &VoxtralModel<Backend>,
    audio_embeds: Tensor<Backend, 3>,
    t_embed: Tensor<Backend, 3>,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Vec<i32> {
    use burn::tensor::Int;

    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];

    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN {
        eprintln!(
            "Warning: audio too short ({} tokens, need {})",
            seq_len, PREFIX_LEN
        );
        return Vec::new();
    }

    let mut decoder_cache = model.create_decoder_cache();

    // Build prefix: BOS + 37 STREAMING_PAD
    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    // Embed prefix tokens
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

    generated.into_iter().skip(PREFIX_LEN).collect()
}
