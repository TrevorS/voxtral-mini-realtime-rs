//! Compare transcription accuracy across quantized model variants.
//!
//! Uses Wgpu for the original model and NdArray for quantized models
//! (Wgpu doesn't support Q8S dequantization yet).
//!
//! Usage:
//!   cargo run --release --bin test_accuracy --features "cli cpu" -- --variant original
//!   cargo run --release --bin test_accuracy --features "cli cpu" -- --variant q8-full
//!   cargo run --release --bin test_accuracy --features "cli cpu" -- --variant q8-v32k

use anyhow::{Context, Result};
use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::ElementConversion;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use std::path::PathBuf;

use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::decoder::{DecoderParts, LanguageModel};
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::sharding;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::models::voxtral::{VoxtralModel, VoxtralModelConfig};
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

#[derive(Parser)]
#[command(name = "test_accuracy")]
#[command(about = "Compare transcription accuracy across quantized model variants")]
struct Cli {
    /// Model variant to test
    #[arg(short, long)]
    variant: String,

    /// Audio file to transcribe
    #[arg(short, long, default_value = "test_data/mary_had_lamb.wav")]
    audio: PathBuf,

    /// Path to tokenizer
    #[arg(long, default_value = "models/voxtral/tekken.json")]
    tokenizer: PathBuf,
}

/// Variant-specific model paths and config.
struct ModelVariant {
    name: &'static str,
    vocab_size: usize,
    /// None = load from SafeTensors, Some(path) = load from .mpk.gz
    quantized_path: Option<PathBuf>,
}

fn get_variant(name: &str) -> Result<ModelVariant> {
    match name {
        "original" => Ok(ModelVariant {
            name: "original (BF16 via Wgpu)",
            vocab_size: 131072,
            quantized_path: None,
        }),
        "q8-full" => Ok(ModelVariant {
            name: "q8-full (Q8 all, 131K vocab, NdArray)",
            vocab_size: 131072,
            quantized_path: Some(PathBuf::from("models/quantized/voxtral-q8-full")),
        }),
        "q8-v32k" => Ok(ModelVariant {
            name: "q8-v32k (Q8 all, 32K vocab, NdArray)",
            vocab_size: 32768,
            quantized_path: Some(PathBuf::from("models/quantized/voxtral-q8-v32k")),
        }),
        "mixed-q8" => Ok(ModelVariant {
            name: "mixed-q8 (Q8 decoder, full encoder, NdArray)",
            vocab_size: 131072,
            quantized_path: Some(PathBuf::from("models/quantized/voxtral-mixed-q8")),
        }),
        "phased" => Ok(ModelVariant {
            name: "phased (Q8 shards, encoder→decoder, Wgpu)",
            vocab_size: 131072,
            quantized_path: None, // handled specially
        }),
        "phased-streamed" => Ok(ModelVariant {
            name: "phased-streamed (Q8 per-layer decoder, Wgpu)",
            vocab_size: 131072,
            quantized_path: None, // handled specially
        }),
        _ => anyhow::bail!(
            "Unknown variant: {}. Options: original, q8-full, q8-v32k, mixed-q8, phased, phased-streamed",
            name
        ),
    }
}

/// ModuleMapper that dequantizes all float parameters.
/// Wgpu can LOAD quantized tensors but can't use them in computation.
/// This mapper walks the module tree and dequantizes each parameter.
struct Dequantizer;

impl<B: Backend> ModuleMapper<B> for Dequantizer {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.dequantize();
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn quantized_model_config(vocab_size: usize) -> VoxtralModelConfig {
    let mut config = VoxtralModelConfig::voxtral();
    config.decoder =
        voxtral_mini_realtime::models::decoder::LanguageModelConfig::new(vocab_size, 3072, 26, 32)
            .with_sliding_window(Some(8192));
    config
}

/// Load a quantized .mpk.gz model for Wgpu inference.
///
/// Wgpu doesn't support Q8S dequantization kernels, so we:
/// 1. Load on NdArray (supports Q8S)
/// 2. Dequantize all weights to f32 via ModuleMapper
/// 3. Save to temp file (now plain f32)
/// 4. Load on Wgpu
fn load_quantized_for_wgpu(
    path: &PathBuf,
    vocab_size: usize,
    wgpu_device: &<burn::backend::Wgpu as Backend>::Device,
) -> Result<VoxtralModel<burn::backend::Wgpu>> {
    use burn::backend::ndarray::NdArray;

    println!("Loading quantized model from {}.mpk.gz...", path.display());

    let config = quantized_model_config(vocab_size);
    let ndarray_device = Default::default();
    let model = config.init::<NdArray>(&ndarray_device);

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
    let model = model
        .load_file(path, &recorder, &ndarray_device)
        .map_err(|e| anyhow::anyhow!("Failed to load quantized model: {}", e))?;

    // Dequantize all weights on NdArray (which supports Q8S)
    println!("  Dequantizing weights on CPU...");
    let model = model.map(&mut Dequantizer);

    // Save dequantized (f32) model to temp file, freeing NdArray memory
    let temp_path = PathBuf::from("/tmp/voxtral-accuracy-dequantized");
    println!("  Saving dequantized weights to temp file...");
    model
        .save_file(&temp_path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save temp model: {}", e))?;

    // Load f32 weights on Wgpu
    println!("  Loading dequantized weights on GPU...");
    let wgpu_model = config.init::<burn::backend::Wgpu>(wgpu_device);
    let wgpu_model = wgpu_model
        .load_file(&temp_path, &recorder, wgpu_device)
        .map_err(|e| anyhow::anyhow!("Failed to load on Wgpu: {}", e))?;

    // Clean up temp file
    let _ = std::fs::remove_file(temp_path.with_extension("mpk.gz"));

    Ok(wgpu_model)
}

fn extract_mel(audio_path: &PathBuf) -> Result<(Vec<f32>, usize, usize)> {
    println!("Loading audio from {}...", audio_path.display());
    let audio = load_wav(audio_path).context("Failed to load audio")?;
    println!(
        "  Duration: {:.2}s ({} samples, {}Hz)",
        audio.duration_secs(),
        audio.samples.len(),
        audio.sample_rate
    );

    let audio = if audio.sample_rate != 16000 {
        println!("  Resampling to 16kHz...");
        resample_to_16k(&audio).context("Failed to resample")?
    } else {
        audio
    };

    let pad_config = PadConfig::voxtral();
    let padded = pad_audio(&audio, &pad_config);

    println!("Extracting mel spectrogram...");
    let mel_config = MelConfig::voxtral();
    let mel_extractor = MelSpectrogram::new(mel_config);
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };
    println!("  Mel shape: [{}, {}]", n_mels, n_frames);

    // Transpose to [n_mels, n_frames]
    let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_transposed[mel_idx][frame_idx] = val;
        }
    }
    let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
    Ok((mel_flat, n_mels, n_frames))
}

/// Autoregressive transcription with KV cache.
fn transcribe<B: Backend>(
    model: &VoxtralModel<B>,
    audio_embeds: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
    device: &B::Device,
) -> Vec<i32> {
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];

    const PREFIX_LEN: usize = 38;
    if seq_len < PREFIX_LEN {
        eprintln!(
            "Audio too short ({} positions, need {})",
            seq_len, PREFIX_LEN
        );
        return Vec::new();
    }

    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    let mut decoder_cache = model.create_decoder_cache();

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_tensor = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

    let prefix_inputs = prefix_audio + prefix_text_embeds;

    println!("  Running prefix ({} positions)...", PREFIX_LEN);
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

    let total_steps = seq_len - PREFIX_LEN - 1;
    println!("  Generating {} tokens...", total_steps);

    for (step, pos) in (PREFIX_LEN + 1..seq_len).enumerate() {
        if (step + 1) % 10 == 0 || step + 1 == total_steps {
            println!("    step {}/{}", step + 1, total_steps);
        }

        let new_token = generated[pos - 1];
        let token_tensor = Tensor::<B, 2, Int>::from_data(
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

/// Autoregressive transcription using a standalone LanguageModel (no VoxtralModel wrapper).
/// Used in phased inference where encoder and decoder are loaded separately.
fn transcribe_decoder<B: Backend>(
    decoder: &LanguageModel<B>,
    audio_embeds: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
    device: &B::Device,
) -> Vec<i32> {
    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];

    const PREFIX_LEN: usize = 38;
    if seq_len < PREFIX_LEN {
        eprintln!(
            "Audio too short ({} positions, need {})",
            seq_len, PREFIX_LEN
        );
        return Vec::new();
    }

    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    let mut decoder_cache = decoder.create_cache();

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_tensor = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = decoder.embed_tokens(prefix_tensor);

    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

    let prefix_inputs = prefix_audio + prefix_text_embeds;

    println!("  Running prefix ({} positions)...", PREFIX_LEN);
    let hidden =
        decoder.forward_hidden_with_cache(prefix_inputs, t_embed.clone(), &mut decoder_cache);
    let logits = decoder.lm_head(hidden);

    let vocab_size = logits.dims()[2];
    let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
    let first_pred = last_logits.argmax(2);
    let first_token: i32 = first_pred.into_scalar().elem();

    let mut generated = prefix;
    generated.push(first_token);

    let total_steps = seq_len - PREFIX_LEN - 1;
    println!("  Generating {} tokens...", total_steps);

    for (step, pos) in (PREFIX_LEN + 1..seq_len).enumerate() {
        if (step + 1) % 10 == 0 || step + 1 == total_steps {
            println!("    step {}/{}", step + 1, total_steps);
        }

        let new_token = generated[pos - 1];
        let token_tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![new_token], [1, 1]),
            device,
        );
        let text_embed = decoder.embed_tokens(token_tensor);

        let audio_pos = audio_embeds
            .clone()
            .slice([0..1, (pos - 1)..pos, 0..d_model]);

        let input = audio_pos + text_embed;

        let hidden = decoder.forward_hidden_with_cache(input, t_embed.clone(), &mut decoder_cache);
        let logits = decoder.lm_head(hidden);

        let pred = logits.argmax(2);
        let next_token: i32 = pred.into_scalar().elem();
        generated.push(next_token);
    }

    generated.into_iter().skip(PREFIX_LEN).collect()
}

fn run_inference<B: Backend>(
    model: &VoxtralModel<B>,
    mel_flat: &[f32],
    n_mels: usize,
    n_frames: usize,
    tokenizer: &VoxtralTokenizer,
    variant_name: &str,
    device: &B::Device,
) -> Result<()> {
    let mel_tensor: Tensor<B, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat.to_vec(), [1, n_mels, n_frames]),
        device,
    );

    println!("Running encoder...");
    let audio_embeds = model.encode_audio(mel_tensor);
    println!("  Audio sequence length: {}", audio_embeds.dims()[1]);

    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<B>(6.0, device);

    println!("\nTranscribing...");
    let start = std::time::Instant::now();
    let generated_tokens = transcribe(model, audio_embeds, t_embed, device);
    let elapsed = start.elapsed();

    let pad_count = generated_tokens.iter().filter(|&&t| t == 32).count();
    let word_count = generated_tokens.iter().filter(|&&t| t == 33).count();
    let text_count = generated_tokens.iter().filter(|&&t| t >= 1000).count();
    println!(
        "  {} tokens: {} pad, {} word, {} text",
        generated_tokens.len(),
        pad_count,
        word_count,
        text_count
    );
    println!("  Inference time: {:.1}s", elapsed.as_secs_f64());

    let text_tokens: Vec<u32> = generated_tokens
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    match tokenizer.decode(&text_tokens) {
        Ok(text) => {
            println!("\n=== Transcription ({}) ===", variant_name);
            println!("{}", text);
            println!("===========================");
        }
        Err(e) => {
            eprintln!("Failed to decode tokens: {}", e);
        }
    }

    println!("\nToken IDs: {:?}", generated_tokens);
    Ok(())
}

/// Run phased inference: encoder shards → audio_embeds → decoder shard → text.
///
/// This simulates the wasm32 deployment where encoder and decoder are never
/// loaded simultaneously, keeping peak memory under 4 GiB.
fn run_phased_inference(
    mel_flat: &[f32],
    n_mels: usize,
    n_frames: usize,
    tokenizer: &VoxtralTokenizer,
    device: &<burn::backend::Wgpu as Backend>::Device,
) -> Result<()> {
    use burn::backend::ndarray::NdArray;
    use burn::backend::Wgpu;

    let config = quantized_model_config(131072);
    let shard_dir = std::path::Path::new("models/shards");

    let mel_tensor: Tensor<Wgpu, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat.to_vec(), [1, n_mels, n_frames]),
        device,
    );

    // === Phase 1: Encode ===
    // Load encoder + adapter on NdArray (supports Q8S), dequantize, transfer to Wgpu
    println!("=== Phase 1: Encode ===");
    let start = std::time::Instant::now();

    let recorder = burn::record::NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
    let ndarray_device: <NdArray as Backend>::Device = Default::default();

    println!("Loading encoder shard (NdArray)...");
    let encoder = config.encoder.init::<NdArray>(&ndarray_device);
    let encoder = encoder
        .load_file(shard_dir.join("encoder"), &recorder, &ndarray_device)
        .map_err(|e| anyhow::anyhow!("Failed to load encoder shard: {}", e))?;

    println!("Loading adapter shard (NdArray)...");
    let adapter = config.adapter.init::<NdArray>(&ndarray_device);
    let adapter = adapter
        .load_file(shard_dir.join("adapter"), &recorder, &ndarray_device)
        .map_err(|e| anyhow::anyhow!("Failed to load adapter shard: {}", e))?;

    // Dequantize on NdArray, save to temp, load on Wgpu (same pattern as quantized models)
    println!("Dequantizing encoder...");
    let encoder = encoder.map(&mut Dequantizer);
    let adapter = adapter.map(&mut Dequantizer);

    let encoder_tmp = PathBuf::from("/tmp/voxtral-phased-encoder");
    let adapter_tmp = PathBuf::from("/tmp/voxtral-phased-adapter");
    encoder
        .save_file(&encoder_tmp, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save encoder temp: {}", e))?;
    adapter
        .save_file(&adapter_tmp, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save adapter temp: {}", e))?;

    println!("Loading encoder on GPU...");
    let encoder_gpu = config.encoder.init::<Wgpu>(device);
    let encoder_gpu = encoder_gpu
        .load_file(&encoder_tmp, &recorder, device)
        .map_err(|e| anyhow::anyhow!("Failed to load encoder on Wgpu: {}", e))?;
    let adapter_gpu = config.adapter.init::<Wgpu>(device);
    let adapter_gpu = adapter_gpu
        .load_file(&adapter_tmp, &recorder, device)
        .map_err(|e| anyhow::anyhow!("Failed to load adapter on Wgpu: {}", e))?;

    let _ = std::fs::remove_file(encoder_tmp.with_extension("mpk.gz"));
    let _ = std::fs::remove_file(adapter_tmp.with_extension("mpk.gz"));

    // Run encoder phase
    println!("Running encoder...");
    let (audio_embeds_data, audio_embeds_shape) = sharding::run_encoder_phase(
        &encoder_gpu,
        &adapter_gpu,
        mel_tensor,
        config.reshape_factor,
    );

    println!(
        "  Audio embeds: [{}, {}, {}] ({:.2} MB)",
        audio_embeds_shape[0],
        audio_embeds_shape[1],
        audio_embeds_shape[2],
        audio_embeds_data.len() as f64 * 4.0 / 1e6
    );

    let encode_time = start.elapsed();
    println!("  Encode phase: {:.1}s", encode_time.as_secs_f64());

    // Drop encoder + adapter to free memory
    drop(encoder_gpu);
    drop(adapter_gpu);
    println!("  Encoder freed.");

    // === Phase 2: Decode ===
    println!("\n=== Phase 2: Decode ===");
    let start = std::time::Instant::now();

    println!("Loading decoder shard (NdArray)...");
    let decoder = config.decoder.init::<NdArray>(&ndarray_device);
    let decoder = decoder
        .load_file(shard_dir.join("decoder"), &recorder, &ndarray_device)
        .map_err(|e| anyhow::anyhow!("Failed to load decoder shard: {}", e))?;

    println!("Dequantizing decoder...");
    let decoder = decoder.map(&mut Dequantizer);

    let decoder_tmp = PathBuf::from("/tmp/voxtral-phased-decoder");
    println!("Saving dequantized decoder...");
    decoder
        .save_file(&decoder_tmp, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save decoder temp: {}", e))?;

    println!("Loading decoder on GPU...");
    let decoder_gpu = config.decoder.init::<Wgpu>(device);
    let decoder_gpu = decoder_gpu
        .load_file(&decoder_tmp, &recorder, device)
        .map_err(|e| anyhow::anyhow!("Failed to load decoder on Wgpu: {}", e))?;

    let _ = std::fs::remove_file(decoder_tmp.with_extension("mpk.gz"));

    // Reconstruct audio_embeds tensor on Wgpu
    let audio_embeds: Tensor<Wgpu, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(audio_embeds_data, audio_embeds_shape),
        device,
    );

    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Wgpu>(6.0, device);

    println!("\nTranscribing...");
    let generated_tokens = transcribe_decoder(&decoder_gpu, audio_embeds, t_embed, device);
    let decode_time = start.elapsed();

    let pad_count = generated_tokens.iter().filter(|&&t| t == 32).count();
    let word_count = generated_tokens.iter().filter(|&&t| t == 33).count();
    let text_count = generated_tokens.iter().filter(|&&t| t >= 1000).count();
    println!(
        "  {} tokens: {} pad, {} word, {} text",
        generated_tokens.len(),
        pad_count,
        word_count,
        text_count
    );
    println!("  Decode phase: {:.1}s", decode_time.as_secs_f64());

    let text_tokens: Vec<u32> = generated_tokens
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    match tokenizer.decode(&text_tokens) {
        Ok(text) => {
            println!("\n=== Transcription (phased) ===");
            println!("{}", text);
            println!("==============================");
        }
        Err(e) => {
            eprintln!("Failed to decode tokens: {}", e);
        }
    }

    println!("\nToken IDs: {:?}", generated_tokens);
    Ok(())
}

/// Run phased-streamed inference entirely on NdArray (CPU).
///
/// NdArray supports Q8S natively, so we skip the dequantize→temp→Wgpu dance.
/// Loads the decoder from 28 individual shard files and compares output
/// token-for-token against the monolithic decoder.
fn run_phased_streamed_inference(
    mel_flat: &[f32],
    n_mels: usize,
    n_frames: usize,
    tokenizer: &VoxtralTokenizer,
    _wgpu_device: &<burn::backend::Wgpu as Backend>::Device,
) -> Result<()> {
    use burn::backend::ndarray::NdArray;

    let config = quantized_model_config(131072);
    let streamed_dir = std::path::Path::new("models/shards-streamed");
    let monolithic_dir = std::path::Path::new("models/shards");
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
    let device: <NdArray as Backend>::Device = Default::default();

    let mel_tensor: Tensor<NdArray, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat.to_vec(), [1, n_mels, n_frames]),
        &device,
    );

    // === Phase 1: Encode (shared — use streamed dir, same encoder/adapter) ===
    println!("=== Phase 1: Encode (NdArray) ===");
    let start = std::time::Instant::now();

    let encoder = config.encoder.init::<NdArray>(&device);
    let encoder = encoder
        .load_file(streamed_dir.join("encoder"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load encoder: {}", e))?;

    let adapter = config.adapter.init::<NdArray>(&device);
    let adapter = adapter
        .load_file(streamed_dir.join("adapter"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load adapter: {}", e))?;

    let (audio_embeds_data, audio_embeds_shape) =
        sharding::run_encoder_phase(&encoder, &adapter, mel_tensor, config.reshape_factor);

    println!(
        "  Audio embeds: [{}, {}, {}]",
        audio_embeds_shape[0], audio_embeds_shape[1], audio_embeds_shape[2],
    );
    println!("  Encode: {:.1}s", start.elapsed().as_secs_f64());

    drop(encoder);
    drop(adapter);

    // === Phase 2a: Decode with MONOLITHIC shard ===
    println!("\n=== Phase 2a: Monolithic decoder (NdArray) ===");
    let start = std::time::Instant::now();

    let decoder_mono = config.decoder.init::<NdArray>(&device);
    let decoder_mono = decoder_mono
        .load_file(monolithic_dir.join("decoder"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load monolithic decoder: {}", e))?;

    println!("  Loaded: {:.1}s", start.elapsed().as_secs_f64());

    let audio_embeds_mono: Tensor<NdArray, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(audio_embeds_data.clone(), audio_embeds_shape),
        &device,
    );
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<NdArray>(6.0, &device);

    println!("  Transcribing...");
    let mono_tokens =
        transcribe_decoder(&decoder_mono, audio_embeds_mono, t_embed.clone(), &device);
    println!("  Done: {:.1}s", start.elapsed().as_secs_f64());
    drop(decoder_mono);

    // === Phase 2b: Decode with STREAMED per-layer shards ===
    println!("\n=== Phase 2b: Streamed decoder (NdArray) ===");
    let start = std::time::Instant::now();

    // Embeddings
    println!("  Loading decoder_embeddings...");
    let emb = config.decoder.init_embeddings::<NdArray>(&device);
    let emb = emb
        .load_file(streamed_dir.join("decoder_embeddings"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load decoder embeddings: {}", e))?;

    // Layers
    let mut layers = Vec::with_capacity(config.decoder.n_layers);
    for i in 0..config.decoder.n_layers {
        let name = format!("decoder_layer_{:02}", i);
        println!("  Loading {}...", name);
        let layer = config.decoder.init_single_layer::<NdArray>(&device);
        let layer = layer
            .load_file(streamed_dir.join(&name), &recorder, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", name, e))?;
        layers.push(layer);
    }

    // Norm
    println!("  Loading decoder_norm...");
    let norm = config.decoder.init_norm::<NdArray>(&device);
    let norm = norm
        .load_file(streamed_dir.join("decoder_norm"), &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load decoder norm: {}", e))?;

    // Assemble
    let rope = config.decoder.init_rope::<NdArray>(&device);
    let parts = DecoderParts {
        tok_embeddings: emb,
        layers,
        norm,
        d_model: config.decoder.d_model,
    };
    let decoder_streamed = LanguageModel::from_parts(parts, rope);

    println!("  Assembled: {:.1}s", start.elapsed().as_secs_f64());

    let audio_embeds_str: Tensor<NdArray, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(audio_embeds_data, audio_embeds_shape),
        &device,
    );

    println!("  Transcribing...");
    let streamed_tokens = transcribe_decoder(&decoder_streamed, audio_embeds_str, t_embed, &device);
    println!("  Done: {:.1}s", start.elapsed().as_secs_f64());

    // === Compare ===
    println!("\n=== Comparison ===");
    println!("  Monolithic tokens: {}", mono_tokens.len());
    println!("  Streamed tokens:   {}", streamed_tokens.len());

    if mono_tokens == streamed_tokens {
        println!("  MATCH: Token-for-token identical!");
    } else {
        let mismatches: Vec<_> = mono_tokens
            .iter()
            .zip(streamed_tokens.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect();
        println!("  MISMATCH: {} differences", mismatches.len());
        for (i, (a, b)) in mismatches.iter().take(10) {
            println!("    position {}: mono={} vs streamed={}", i, a, b);
        }
    }

    // Decode and show text
    let text_tokens: Vec<u32> = streamed_tokens
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    match tokenizer.decode(&text_tokens) {
        Ok(text) => {
            println!("\n=== Transcription (phased-streamed) ===");
            println!("{}", text);
            println!("=======================================");
        }
        Err(e) => {
            eprintln!("Failed to decode tokens: {}", e);
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let variant = get_variant(&cli.variant)?;

    println!("=== Accuracy Test: {} ===", variant.name);
    println!();

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer =
        VoxtralTokenizer::from_file(&cli.tokenizer).context("Failed to load tokenizer")?;

    // Extract mel (backend-independent)
    let (mel_flat, n_mels, n_frames) = extract_mel(&cli.audio)?;

    use burn::backend::Wgpu;
    let device: <Wgpu as burn::tensor::backend::Backend>::Device = Default::default();

    // Phased variants use different code paths
    if cli.variant == "phased" {
        return run_phased_inference(&mel_flat, n_mels, n_frames, &tokenizer, &device);
    }
    if cli.variant == "phased-streamed" {
        return run_phased_streamed_inference(&mel_flat, n_mels, n_frames, &tokenizer, &device);
    }

    // All other models use Wgpu for GPU inference.
    // Quantized models are loaded on NdArray, dequantized, then transferred to Wgpu
    // (Wgpu can load Q8 tensors but can't run compute on them yet).
    let model: VoxtralModel<Wgpu> = match &variant.quantized_path {
        None => {
            println!("Loading original model from SafeTensors (Wgpu)...");
            let loader = VoxtralModelLoader::from_file("models/voxtral/consolidated.safetensors")
                .context("Failed to create model loader")?;
            loader
                .load::<Wgpu>(&device)
                .context("Failed to load model")?
        }
        Some(path) => load_quantized_for_wgpu(path, variant.vocab_size, &device)?,
    };

    let param_count = model.num_params();
    println!(
        "  Parameters: {} ({:.2}B)",
        param_count,
        param_count as f64 / 1e9
    );

    run_inference(
        &model,
        &mel_flat,
        n_mels,
        n_frames,
        &tokenizer,
        &cli.variant,
        &device,
    )
}
