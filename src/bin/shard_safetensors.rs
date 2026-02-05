//! Split consolidated.safetensors into per-component shards for browser loading.
//!
//! Each shard is a standalone .safetensors file containing only the tensors
//! for one model component, gzipped for transport. The tensor names are
//! preserved exactly from the original file so VoxtralModelLoader works
//! unchanged.
//!
//! Usage:
//!   cargo run --features cli --bin shard_safetensors -- \
//!     --model models/voxtral/consolidated.safetensors \
//!     --output models/shards-safetensors

use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "shard_safetensors")]
#[command(about = "Split consolidated.safetensors into per-component shards")]
struct Args {
    /// Path to consolidated.safetensors
    #[arg(long, default_value = "models/voxtral/consolidated.safetensors")]
    model: PathBuf,

    /// Output directory for shards
    #[arg(long, default_value = "models/shards-safetensors")]
    output: PathBuf,
}

/// Tensor name prefixes matching weights.rs
const ENCODER_PREFIX: &str = "mm_streams_embeddings.embedding_module.whisper_encoder";
const ADAPTER_PREFIX: &str = "mm_streams_embeddings.embedding_module.audio_language_projection";
const TOK_EMBEDDINGS: &str = "mm_streams_embeddings.embedding_module.tok_embeddings.weight";
const FINAL_NORM: &str = "norm.weight";
const DECODER_LAYER_PREFIX: &str = "layers.";

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Reading {}...", args.model.display());
    let bytes = fs::read(&args.model)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;

    let tensor_names: Vec<String> = safetensors
        .names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    println!("Found {} tensors", tensor_names.len());

    fs::create_dir_all(&args.output)?;

    // Classify each tensor into a shard
    let mut encoder_tensors: Vec<&str> = Vec::new();
    let mut adapter_tensors: Vec<&str> = Vec::new();
    let mut embeddings_tensors: Vec<&str> = Vec::new();
    let mut norm_tensors: Vec<&str> = Vec::new();
    let mut layer_tensors: HashMap<usize, Vec<&str>> = HashMap::new();

    for name in &tensor_names {
        let name_str: &str = name;
        if name_str.starts_with(ENCODER_PREFIX) {
            encoder_tensors.push(name_str);
        } else if name_str.starts_with(ADAPTER_PREFIX) {
            adapter_tensors.push(name_str);
        } else if name_str == TOK_EMBEDDINGS {
            embeddings_tensors.push(name_str);
        } else if name_str == FINAL_NORM {
            norm_tensors.push(name_str);
        } else if let Some(rest) = name_str.strip_prefix(DECODER_LAYER_PREFIX) {
            // Extract layer index from "layers.{i}.rest"
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                    layer_tensors.entry(layer_idx).or_default().push(name_str);
                }
            }
        }
    }

    // Write shards
    write_shard(&safetensors, &encoder_tensors, &args.output, "encoder")?;
    write_shard(&safetensors, &adapter_tensors, &args.output, "adapter")?;
    write_shard(
        &safetensors,
        &embeddings_tensors,
        &args.output,
        "decoder_embeddings",
    )?;

    let mut layer_indices: Vec<usize> = layer_tensors.keys().copied().collect();
    layer_indices.sort();
    for layer_idx in &layer_indices {
        let tensors = &layer_tensors[layer_idx];
        let name = format!("decoder_layer_{:02}", layer_idx);
        write_shard(&safetensors, tensors, &args.output, &name)?;
    }

    write_shard(&safetensors, &norm_tensors, &args.output, "decoder_norm")?;

    // Write manifest
    let manifest = build_manifest(&args.output, layer_indices.len())?;
    let manifest_path = args.output.join("manifest.json");
    fs::write(&manifest_path, manifest)?;
    println!("Manifest: {}", manifest_path.display());

    println!("Sharding complete!");
    Ok(())
}

/// Write a subset of tensors as a gzipped safetensors file.
fn write_shard(
    safetensors: &SafeTensors,
    tensor_names: &[&str],
    output_dir: &Path,
    shard_name: &str,
) -> anyhow::Result<()> {
    use safetensors::tensor::Dtype;

    // Collect tensor metadata and data for serialization
    let mut tensors: Vec<(&str, safetensors::tensor::TensorView<'_>)> = Vec::new();

    for &name in tensor_names {
        let view = safetensors.tensor(name)?;
        tensors.push((name, view));
    }

    // Build the data map for safetensors::serialize
    let data_map: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, view)| {
            (
                name.to_string(),
                view.dtype(),
                view.shape().to_vec(),
                view.data().to_vec(),
            )
        })
        .collect();

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = data_map
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let st_bytes = safetensors::tensor::serialize(views, None)?;

    // Gzip compress
    let gz_path = output_dir.join(format!("{}.safetensors.gz", shard_name));
    let file = fs::File::create(&gz_path)?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder.write_all(&st_bytes)?;
    encoder.finish()?;

    let size = fs::metadata(&gz_path)?.len();
    println!(
        "  {} -> {:.2} MB ({} tensors)",
        gz_path.display(),
        size as f64 / 1e6,
        tensor_names.len()
    );

    Ok(())
}

/// Build a JSON manifest listing all shard files.
fn build_manifest(output_dir: &Path, n_layers: usize) -> anyhow::Result<String> {
    let mut shards = Vec::new();

    // Fixed shards
    for name in &["encoder", "adapter", "decoder_embeddings"] {
        let filename = format!("{}.safetensors.gz", name);
        let path = output_dir.join(&filename);
        if path.exists() {
            let size = fs::metadata(&path)?.len();
            shards.push(format!(
                r#"    {{"type": "{}", "filename": "{}", "size": {}}}"#,
                name, filename, size
            ));
        }
    }

    // Layer shards
    for i in 0..n_layers {
        let filename = format!("decoder_layer_{:02}.safetensors.gz", i);
        let path = output_dir.join(&filename);
        if path.exists() {
            let size = fs::metadata(&path)?.len();
            shards.push(format!(
                r#"    {{"type": "decoder_layer", "filename": "{}", "size": {}}}"#,
                filename, size
            ));
        }
    }

    // Norm shard
    let norm_filename = "decoder_norm.safetensors.gz";
    let norm_path = output_dir.join(norm_filename);
    if norm_path.exists() {
        let size = fs::metadata(&norm_path)?.len();
        shards.push(format!(
            r#"    {{"type": "decoder_norm", "filename": "{}", "size": {}}}"#,
            norm_filename, size
        ));
    }

    let total_size: u64 = shards
        .iter()
        .filter_map(|s| {
            // Extract size from the JSON string
            s.rsplit("size\": ")
                .next()?
                .strip_suffix('}')?
                .parse::<u64>()
                .ok()
        })
        .sum();

    Ok(format!(
        r#"{{
  "format": "safetensors",
  "total_size": {},
  "decoder_format": "streamed",
  "decoder_layers": {},
  "shards": [
{}
  ]
}}"#,
        total_size,
        n_layers,
        shards.join(",\n")
    ))
}
