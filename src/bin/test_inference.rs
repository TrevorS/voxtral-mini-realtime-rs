//! Test inference with pretrained Voxtral model.
//!
//! Loads an audio file, extracts mel spectrogram, runs through the model,
//! and decodes the output to text.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use voxtral_mini_realtime::audio::{
    io::load_wav,
    mel::{MelConfig, MelSpectrogram},
    resample::resample_to_16k,
};
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;
use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

type TestBackend = Wgpu;

fn main() {
    // Check arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!();
        eprintln!("Example: {} test.wav", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
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
    println!("  Vocab size: {}", tokenizer.vocab_size());

    // Load audio
    println!("\nLoading audio from {}...", audio_path);
    let audio = load_wav(audio_path).expect("Failed to load audio");
    println!("  Sample rate: {} Hz", audio.sample_rate);
    println!("  Duration: {:.2}s", audio.duration_secs());
    println!("  Samples: {}", audio.samples.len());

    // Resample to 16kHz if needed
    let audio = if audio.sample_rate != 16000 {
        println!("  Resampling to 16kHz...");
        resample_to_16k(&audio).expect("Failed to resample audio")
    } else {
        audio
    };
    let samples = &audio.samples;
    println!("  Samples after resampling: {}", samples.len());

    // Try to load reference mel from Python if available
    let mel_tensor: Tensor<TestBackend, 3> = if std::path::Path::new("test_data/reference_mel.npy")
        .exists()
    {
        println!("\nLoading reference mel from Python...");
        use ndarray::ArrayD;
        use ndarray_npy::ReadNpyExt;
        use std::fs::File;

        let file = File::open("test_data/reference_mel.npy").expect("Failed to open npy file");
        let mel_arr: ArrayD<f32> = ArrayD::<f32>::read_npy(file).expect("Failed to read npy file");
        let shape = mel_arr.shape();
        let n_mels = shape[0];
        let n_frames = shape[1];
        println!("  Reference mel shape: [{}, {}]", n_mels, n_frames);

        let mel_flat: Vec<f32> = mel_arr.iter().cloned().collect();
        Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &device,
        )
    } else {
        // Fall back to Rust mel computation
        println!("\nExtracting mel spectrogram (Rust)...");
        let mel_config = MelConfig::voxtral();
        let mel_extractor = MelSpectrogram::new(mel_config);
        let mel = mel_extractor.compute_log(samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };
        println!("  Mel shape: [frames={}, mels={}]", n_frames, n_mels);

        // Transpose from [n_frames, n_mels] to [n_mels, n_frames]
        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }

        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &device,
        )
    };
    let n_frames = mel_tensor.dims()[2];
    println!("  Mel tensor shape: {:?}", mel_tensor.dims());

    // Load model
    println!("\nLoading model (this may take a few minutes)...");
    let loader = VoxtralModelLoader::from_file(model_path).expect("Failed to create model loader");
    let model = loader
        .load::<TestBackend>(&device)
        .expect("Failed to load model");

    // Debug: check mel input
    println!("\nDebug: checking mel input...");
    let mel_data = mel_tensor.clone().to_data();
    let mel_slice = mel_data.as_slice::<f32>().unwrap();
    let mel_min = mel_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let mel_max = mel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mel_mean: f32 = mel_slice.iter().sum::<f32>() / mel_slice.len() as f32;
    println!(
        "  Mel stats: min={:.4}, max={:.4}, mean={:.4}",
        mel_min, mel_max, mel_mean
    );
    println!(
        "  First few mel values: {:?}",
        &mel_slice[..10.min(mel_slice.len())]
    );

    // Compare with zeros input
    println!("\nDebug: testing with zeros input...");
    let zeros_mel: Tensor<TestBackend, 3> = Tensor::zeros([1, 128, n_frames], &device);
    let zeros_hidden = model.encode_audio(zeros_mel.clone());
    let zh_data = zeros_hidden.to_data();
    let zh_slice = zh_data.as_slice::<f32>().unwrap();
    println!(
        "  Zeros hidden first 5: {:?}",
        &zh_slice[..5.min(zh_slice.len())]
    );

    // Debug: check encoder output
    println!("\nDebug: checking encoder output...");
    let audio_hidden = model.encode_audio(mel_tensor.clone());
    let ah_data = audio_hidden.clone().to_data();
    let ah_slice = ah_data.as_slice::<f32>().unwrap();
    let ah_min = ah_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let ah_max = ah_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ah_mean: f32 = ah_slice.iter().sum::<f32>() / ah_slice.len() as f32;
    println!("  Audio hidden shape: {:?}", audio_hidden.dims());
    println!(
        "  Audio hidden stats: min={:.4}, max={:.4}, mean={:.4}",
        ah_min, ah_max, ah_mean
    );
    println!(
        "  First few values: {:?}",
        &ah_slice[..10.min(ah_slice.len())]
    );

    // Try different t_embed values
    // The ADA RMSNorm formula is: (1 + scale) * rms_norm(x)
    // where scale = W2(SiLU(W0(t_embed)))
    // If t_embed is zeros, scale becomes ~0, so output = rms_norm(x)
    // If t_embed has values, it modulates the normalization
    // Let's try zeros first (simplest case)
    let t_embed: Tensor<TestBackend, 3> = Tensor::zeros([1, 1, 3072], &device);
    println!("\nDebug: Using zeros for t_embed");
    let te_data = t_embed.clone().to_data();
    let te_slice = te_data.as_slice::<f32>().unwrap();
    println!(
        "  t_embed first 5: {:?}",
        &te_slice[..5.min(te_slice.len())]
    );

    // Run inference
    println!("\nRunning inference...");
    let logits = model.forward(mel_tensor, t_embed);
    println!("  Logits shape: {:?}", logits.dims());

    // Debug: print logits stats
    let logits_data = logits.clone().to_data();
    let logits_slice = logits_data.as_slice::<f32>().unwrap();
    let min_val = logits_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = logits_slice
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mean_val: f32 = logits_slice.iter().sum::<f32>() / logits_slice.len() as f32;
    println!(
        "  Logits stats: min={:.4}, max={:.4}, mean={:.4}",
        min_val, max_val, mean_val
    );

    // Print first few logits for first position
    println!(
        "  First position logits (first 10): {:?}",
        &logits_slice[..10]
    );

    // Find the actual argmax for first position
    let vocab_size = 131072;
    let first_pos_logits = &logits_slice[..vocab_size];
    let (max_idx, max_val) = first_pos_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "  Position 0 argmax: token {} with logit {:.4}",
        max_idx, max_val
    );

    // Show top 5 tokens for first position
    let mut indexed: Vec<(usize, f32)> = first_pos_logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    println!("  Top 5 tokens at position 0:");
    for (idx, val) in indexed.iter().take(5) {
        println!("    token {}: logit {:.4}", idx, val);
    }

    // Also check position 10 if we have enough tokens
    let actual_seq_len = logits_slice.len() / vocab_size;
    if actual_seq_len > 10 {
        let pos10_logits = &logits_slice[10 * vocab_size..(10 + 1) * vocab_size];
        let mut indexed10: Vec<(usize, f32)> = pos10_logits.iter().cloned().enumerate().collect();
        indexed10.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        println!("  Top 5 tokens at position 10:");
        for (idx, val) in indexed10.iter().take(5) {
            println!("    token {}: logit {:.4}", idx, val);
        }
    }

    // Get argmax predictions
    let [_batch, seq_len, _vocab] = logits.dims();
    let predictions = logits.argmax(2); // [batch, seq]
    let pred_data = predictions.to_data();
    let pred_slice = pred_data.as_slice::<i32>().unwrap();

    println!("  Predicted {} tokens", seq_len);

    // Decode tokens
    let token_ids: Vec<u32> = pred_slice.iter().map(|&x| x as u32).collect();
    println!(
        "\nToken IDs (first 20): {:?}",
        &token_ids[..token_ids.len().min(20)]
    );

    match tokenizer.decode(&token_ids) {
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
