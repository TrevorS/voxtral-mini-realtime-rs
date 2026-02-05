//! Quantize Voxtral model to reduce size for WASM/browser deployment.
//!
//! Generates quantized models at various precision levels (Q4, Q8, mixed)
//! for testing accuracy vs size tradeoffs.
//!
//! Usage:
//!   cargo run --bin quantize_model -- --preset q4-model --output models/voxtral-q4
//!   cargo run --bin quantize_model -- --list-presets
//!   cargo run --bin quantize_model -- shard --input models/quantized/voxtral-q8-full

use anyhow::{Context, Result};
use burn::backend::ndarray::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::sharding::save_phased_shards;
use voxtral_mini_realtime::models::voxtral::VoxtralModelConfig;
use voxtral_mini_realtime::quantization::{quantize_model, QuantConfig};

type Backend = NdArray;

#[derive(Parser)]
#[command(name = "quantize_model")]
#[command(about = "Quantize Voxtral model for WASM/browser deployment")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available quantization presets
    ListPresets,

    /// Quantize a model with the specified preset
    Quantize {
        /// Path to input model (SafeTensors format)
        #[arg(short, long, default_value = "models/voxtral/consolidated.safetensors")]
        input: PathBuf,

        /// Output path (without extension, .mpk.gz will be added)
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization preset name
        #[arg(short, long)]
        preset: String,

        /// Maximum vocabulary size (truncates embedding table to save memory).
        /// Use 32768 for wasm32 deployment (saves ~302 MB).
        #[arg(long)]
        max_vocab: Option<usize>,
    },

    /// Generate all quantization variants
    GenerateAll {
        /// Path to input model (SafeTensors format)
        #[arg(short, long, default_value = "models/voxtral/consolidated.safetensors")]
        input: PathBuf,

        /// Output directory for quantized models
        #[arg(short, long, default_value = "models/quantized")]
        output_dir: PathBuf,
    },

    /// Split a quantized model into phased shards for browser deployment.
    ///
    /// Decomposes a .mpk.gz model into separate encoder, adapter, and decoder
    /// shards that can be loaded sequentially to fit within wasm32's 4 GiB limit.
    Shard {
        /// Path to quantized model (without .mpk.gz extension)
        #[arg(short, long, default_value = "models/quantized/voxtral-q8-full")]
        input: PathBuf,

        /// Output directory for shard files
        #[arg(short, long, default_value = "models/shards")]
        output_dir: PathBuf,

        /// Vocabulary size of the quantized model
        #[arg(long, default_value = "131072")]
        vocab_size: usize,

        /// Save decoder as per-layer streamed shards (28 files) instead of
        /// one monolithic file. Reduces peak memory when loading in WASM.
        #[arg(long, default_value = "false")]
        streamed: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ListPresets => {
            list_presets();
            Ok(())
        }
        Commands::Quantize {
            input,
            output,
            preset,
            max_vocab,
        } => quantize_single(&input, &output, &preset, max_vocab),
        Commands::GenerateAll { input, output_dir } => generate_all(&input, &output_dir),
        Commands::Shard {
            input,
            output_dir,
            vocab_size,
            streamed,
        } => shard_model(&input, &output_dir, vocab_size, streamed),
    }
}

fn list_presets() {
    println!("Available quantization presets:");
    println!();

    // Estimated original size in BF16
    let original_bytes: u64 = 8_860_000_000;

    for (name, config) in QuantConfig::all_presets() {
        let est_size = config.estimate_size(original_bytes);
        let reduction = (1.0 - est_size as f64 / original_bytes as f64) * 100.0;
        println!(
            "  {:<20} {:.2} GB ({:.0}% reduction)",
            name,
            est_size as f64 / 1e9,
            reduction
        );
        println!("    {}", config.description());
        println!();
    }
}

fn get_preset(name: &str) -> Result<QuantConfig> {
    for (preset_name, config) in QuantConfig::all_presets() {
        if preset_name == name {
            return Ok(config);
        }
    }
    anyhow::bail!(
        "Unknown preset: {}. Use 'list-presets' to see available presets.",
        name
    );
}

fn quantize_single(
    input: &PathBuf,
    output: &PathBuf,
    preset_name: &str,
    max_vocab: Option<usize>,
) -> Result<()> {
    let config = get_preset(preset_name)?;

    println!("Quantizing model:");
    println!("  Input:  {}", input.display());
    println!("  Output: {}.mpk.gz", output.display());
    println!("  Preset: {} ({})", preset_name, config.description());
    if let Some(mv) = max_vocab {
        println!("  Vocab:  {} tokens (truncated)", mv);
    }
    println!();

    // Load the model
    println!("Loading model...");
    let device = Default::default();
    let loader = VoxtralModelLoader::from_file(input).context("Failed to create model loader")?;
    let model = loader
        .load_with_options::<Backend>(&device, max_vocab)
        .context("Failed to load model")?;

    // Count original parameters
    let param_count = model.num_params();
    println!(
        "  Parameters: {} ({:.2}B)",
        param_count,
        param_count as f64 / 1e9
    );

    // Quantize
    println!("Quantizing...");
    let quantized = quantize_model(model, &config);

    // Save
    println!("Saving...");
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();

    // Create parent directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    quantized
        .save_file(output, &recorder)
        .context("Failed to save quantized model")?;

    // Report output size
    let output_with_ext = output.with_extension("mpk.gz");
    if output_with_ext.exists() {
        let size = std::fs::metadata(&output_with_ext)?.len();
        println!("  Output size: {:.2} GB", size as f64 / 1e9);
    }

    println!("Done!");
    Ok(())
}

fn shard_model(
    input: &PathBuf,
    output_dir: &std::path::Path,
    vocab_size: usize,
    streamed: bool,
) -> Result<()> {
    println!("Splitting quantized model into phased shards:");
    println!("  Input:      {}.mpk.gz", input.display());
    println!("  Output dir: {}", output_dir.display());
    println!("  Vocab size: {}", vocab_size);
    println!(
        "  Decoder:    {}",
        if streamed {
            "streamed (per-layer)"
        } else {
            "monolithic"
        }
    );
    println!();

    // Build config matching the quantized model
    let mut config = VoxtralModelConfig::voxtral();
    config.decoder =
        voxtral_mini_realtime::models::decoder::LanguageModelConfig::new(vocab_size, 3072, 26, 32)
            .with_sliding_window(Some(8192));

    // Load the quantized model on NdArray
    println!("Loading quantized model...");
    let device = Default::default();
    let model = config.init::<Backend>(&device);
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
    let model = model
        .load_file(input, &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load quantized model: {}", e))?;

    let param_count = model.num_params();
    println!(
        "  Parameters: {} ({:.2}B)",
        param_count,
        param_count as f64 / 1e9
    );

    // Save as phased shards
    println!();
    save_phased_shards(model, output_dir, streamed)?;

    Ok(())
}

fn generate_all(input: &PathBuf, output_dir: &PathBuf) -> Result<()> {
    println!("Generating all quantization variants...");
    println!("  Input:      {}", input.display());
    println!("  Output dir: {}", output_dir.display());
    println!();

    std::fs::create_dir_all(output_dir)?;

    // Load model once
    println!("Loading model...");
    let device = Default::default();
    let loader = VoxtralModelLoader::from_file(input).context("Failed to create model loader")?;

    for (name, config) in QuantConfig::all_presets() {
        println!();
        println!("=== {} ===", name);
        println!("  {}", config.description());

        // Reload the model for each variant (quantization consumes the model)
        let model = loader
            .load::<Backend>(&device)
            .context("Failed to load model")?;

        // Quantize
        println!("  Quantizing...");
        let quantized = quantize_model(model, &config);

        // Save
        let output_path = output_dir.join(format!("voxtral-{}", name));
        println!("  Saving to {}.mpk.gz...", output_path.display());

        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
        quantized
            .save_file(&output_path, &recorder)
            .context("Failed to save quantized model")?;

        // Report size
        let output_with_ext = output_path.with_extension("mpk.gz");
        if output_with_ext.exists() {
            let size = std::fs::metadata(&output_with_ext)?.len();
            println!("  Size: {:.2} GB", size as f64 / 1e9);
        }
    }

    println!();
    println!("All variants generated!");
    Ok(())
}
