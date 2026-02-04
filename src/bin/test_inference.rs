//! Test inference with pretrained Voxtral model.
//!
//! Loads an audio file, extracts mel spectrogram, runs through the model,
//! and decodes the output to text. Fully Rust-native audio pipeline.

use burn::backend::Wgpu;
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type TestBackend = Wgpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav> [--use-python-embeds]", args[0]);
        eprintln!();
        eprintln!("Example: {} test.wav", args[0]);
        eprintln!("         {} test.wav --use-python-embeds", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let use_python_embeds = args.iter().any(|a| a == "--use-python-embeds");
    let use_python_mel = args.iter().any(|a| a == "--use-python-mel");
    let model_path = "models/voxtral/consolidated.safetensors";
    let tokenizer_path = "models/voxtral/tekken.json";

    // Check required files exist
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Error: Model not found at {}", model_path);
        eprintln!("Run: ./scripts/download_model.py");
        std::process::exit(1);
    }

    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("Error: Tokenizer not found at {}", tokenizer_path);
        std::process::exit(1);
    }

    let device = Default::default();

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = VoxtralTokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    // Load model
    println!("Loading model...");
    let loader = VoxtralModelLoader::from_file(model_path).expect("Failed to create model loader");
    let model = loader
        .load::<TestBackend>(&device)
        .expect("Failed to load model");

    // Prepare time embedding (t=6 for 480ms delay at 12.5 Hz frame rate)
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<TestBackend>(6.0, &device);

    // Get audio embeddings
    let audio_embeds: Tensor<TestBackend, 3> = if use_python_embeds {
        // Use pre-computed embeddings from Python for validation
        println!("Loading pre-computed audio embeddings from Python...");
        use ndarray::ArrayD;
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let file = File::open("test_data/reference_audio_embeds_padded.npy")
            .expect("Failed to open npy file");
        let embeds_arr: ArrayD<f32> =
            ArrayD::<f32>::read_npy(file).expect("Failed to read npy file");
        let shape = embeds_arr.shape();
        let seq_len = shape[0];
        let dim = shape[1];
        println!("  Shape: [1, {}, {}]", seq_len, dim);

        let embeds_flat: Vec<f32> = embeds_arr.iter().cloned().collect();
        Tensor::from_data(
            burn::tensor::TensorData::new(embeds_flat, [1, seq_len, dim]),
            &device,
        )
    } else if use_python_mel {
        // Use Python mel with Rust encoder (for debugging)
        println!("Loading mel spectrogram from Python...");
        use ndarray::ArrayD;
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let file =
            File::open("test_data/reference_mel_padded.npy").expect("Failed to open npy file");
        let mel_arr: ArrayD<f32> = ArrayD::<f32>::read_npy(file).expect("Failed to read npy file");
        let shape = mel_arr.shape();
        let n_mels = shape[0];
        let n_frames = shape[1];
        println!("  Python mel shape: [{}, {}]", n_mels, n_frames);

        // Stats for comparison
        let min_val = mel_arr.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = mel_arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_val: f32 = mel_arr.iter().sum::<f32>() / mel_arr.len() as f32;
        println!(
            "  Range: [{:.4}, {:.4}], mean: {:.4}",
            min_val, max_val, mean_val
        );

        let mel_flat: Vec<f32> = mel_arr.iter().cloned().collect();
        let mel_tensor: Tensor<TestBackend, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &device,
        );

        println!("Running Rust encoder with Python mel...");
        model.encode_audio(mel_tensor)
    } else {
        // Pure Rust pipeline
        println!("Loading audio from {}...", audio_path);
        let audio = load_wav(audio_path).expect("Failed to load audio");
        println!(
            "  Original duration: {:.2}s ({} samples)",
            audio.duration_secs(),
            audio.samples.len()
        );

        let audio = if audio.sample_rate != 16000 {
            println!("  Resampling to 16kHz...");
            resample_to_16k(&audio).expect("Failed to resample audio")
        } else {
            audio
        };

        // Apply left-padding for streaming alignment
        let pad_config = PadConfig::voxtral();
        println!(
            "  Left-padding with {} samples ({} tokens)",
            pad_config.left_pad_samples(),
            pad_config.n_left_pad_tokens
        );
        let padded_audio = pad_audio(&audio, &pad_config);
        println!(
            "  Padded duration: {:.2}s ({} samples)",
            padded_audio.duration_secs(),
            padded_audio.samples.len()
        );

        // Extract mel spectrogram
        println!("Extracting mel spectrogram...");
        let mel_config = MelConfig::voxtral();
        let mel_extractor = MelSpectrogram::new(mel_config);
        let mel = mel_extractor.compute_log(&padded_audio.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };
        println!("  Rust mel shape: [{}, {}]", n_mels, n_frames);

        // Stats for comparison
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for frame in &mel {
            for &val in frame {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
                sum += val as f64;
                count += 1;
            }
        }
        let mean_val = sum / count as f64;
        println!(
            "  Range: [{:.4}, {:.4}], mean: {:.4}",
            min_val, max_val, mean_val
        );

        // Transpose to [n_mels, n_frames] and create tensor
        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        let mel_tensor: Tensor<TestBackend, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &device,
        );

        // Run encoder
        println!("Running encoder...");
        model.encode_audio(mel_tensor)
    };

    println!("  Audio sequence length: {}", audio_embeds.dims()[1]);

    // Run streaming transcription
    println!("\nRunning transcription...");
    let generated_tokens = transcribe_with_audio_embeds(&model, audio_embeds, t_embed, &device);

    // Show token distribution
    let streaming_pad_count = generated_tokens.iter().filter(|&&t| t == 32).count();
    let streaming_word_count = generated_tokens.iter().filter(|&&t| t == 33).count();
    let text_token_count = generated_tokens.iter().filter(|&&t| t >= 1000).count();
    println!(
        "  Generated {} tokens: {} pad, {} word markers, {} text",
        generated_tokens.len(),
        streaming_pad_count,
        streaming_word_count,
        text_token_count
    );

    // Filter out control tokens and decode text
    let text_tokens: Vec<u32> = generated_tokens
        .iter()
        .filter(|&&t| t >= 1000)
        .map(|&t| t as u32)
        .collect();

    match tokenizer.decode(&text_tokens) {
        Ok(text) => {
            println!("\n=== Transcription ===");
            println!("{}", text);
            println!("=====================");
        }
        Err(e) => {
            eprintln!("\nFailed to decode tokens: {}", e);
        }
    }
}

/// Transcribe using pre-computed audio embeddings with KV cache optimization.
///
/// Uses KV cache for O(n) inference instead of O(n²) recomputation.
/// Uses prefix length 38 (not 39) to avoid position 38 anomaly.
fn transcribe_with_audio_embeds<B: burn::tensor::backend::Backend>(
    model: &voxtral_mini_realtime::models::voxtral::VoxtralModel<B>,
    audio_embeds: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
    device: &B::Device,
) -> Vec<i32> {
    use burn::tensor::Int;

    let seq_len = audio_embeds.dims()[1];
    let d_model = audio_embeds.dims()[2];

    // Use prefix length 38 (not 39!) to avoid position 38 anomaly
    const PREFIX_LEN: usize = 38;

    // Check if audio is long enough for streaming inference
    if seq_len < PREFIX_LEN {
        eprintln!(
            "Warning: Audio too short ({} positions, need at least {}). \
             Returning empty transcription.",
            seq_len, PREFIX_LEN
        );
        return Vec::new();
    }
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    // Create KV cache for the decoder
    let mut decoder_cache = model.create_decoder_cache();

    // Build prefix: BOS + 37 STREAMING_PAD = 38 tokens
    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    // Embed prefix tokens
    let prefix_tensor = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
        device,
    );
    let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

    // Slice audio embeddings for prefix positions
    let prefix_audio = audio_embeds
        .clone()
        .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

    // Combine for prefix
    let prefix_inputs = prefix_audio + prefix_text_embeds;

    // Run forward for prefix (fills cache with 38 positions)
    let hidden = model.decoder().forward_hidden_with_cache(
        prefix_inputs,
        t_embed.clone(),
        &mut decoder_cache,
    );
    let logits = model.decoder().lm_head(hidden);

    // Get prediction at last prefix position (37) - this predicts token 38
    let vocab_size = logits.dims()[2];
    let last_logits = logits
        .clone()
        .slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
    let first_pred = last_logits.argmax(2);
    let first_token: i32 = first_pred.into_scalar().elem();

    let mut generated = prefix.clone();
    generated.push(first_token);

    // Autoregressive generation with KV cache (O(n) instead of O(n²))
    for pos in PREFIX_LEN + 1..seq_len {
        // Only embed the SINGLE new token (not all previous tokens)
        let new_token = generated[pos - 1];
        let token_tensor = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![new_token], [1, 1]),
            device,
        );
        let text_embed = model.decoder().embed_tokens(token_tensor);

        // Only slice the SINGLE new audio position (not all previous positions)
        let audio_pos = audio_embeds
            .clone()
            .slice([0..1, (pos - 1)..pos, 0..d_model]);

        // Combine single position
        let input = audio_pos + text_embed;

        // Forward with cache - only processes 1 token, reuses cached KV
        let hidden =
            model
                .decoder()
                .forward_hidden_with_cache(input, t_embed.clone(), &mut decoder_cache);
        let logits = model.decoder().lm_head(hidden);

        // Logits has shape [1, 1, vocab] - take the single prediction
        let pred = logits.argmax(2);
        let next_token: i32 = pred.into_scalar().elem();

        generated.push(next_token);
    }

    // Return generated tokens (skip prefix)
    generated.into_iter().skip(PREFIX_LEN).collect()
}
