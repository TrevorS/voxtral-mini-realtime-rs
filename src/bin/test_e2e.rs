//! End-to-end test for Voxtral model.
//!
//! Tests the full forward pass from mel spectrogram to logits.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use voxtral_mini_realtime::models::voxtral::VoxtralModelConfig;

type TestBackend = Wgpu;

fn main() {
    let device = Default::default();

    println!("Creating Voxtral model configuration...");
    let config = VoxtralModelConfig::voxtral();

    println!(
        "  Encoder: {} layers, d_model={}",
        config.encoder.n_layers, config.encoder.d_model
    );
    println!(
        "  Decoder: {} layers, d_model={}",
        config.decoder.n_layers, config.decoder.d_model
    );
    println!("  Vocab size: {}", config.decoder.vocab_size);

    println!("\nInitializing model (with random weights)...");
    let model = config.init::<TestBackend>(&device);

    println!("\nCreating test inputs...");
    // Mel spectrogram: [batch=1, n_mels=128, time=320]
    // This represents ~0.2 seconds of audio (320 * 10ms = 3.2s at 100Hz mel rate)
    // After 4x conv downsample: 320 -> 80
    // After 4x reshape: 80 -> 20 tokens
    let mel = Tensor::<TestBackend, 3>::zeros([1, 128, 320], &device);

    // T-embed for decoder (ADA RMSNorm)
    let t_embed_dec = Tensor::<TestBackend, 3>::zeros([1, 1, config.decoder.d_model], &device);

    println!("  Mel shape: {:?}", mel.dims());
    println!("  T-embed shape: {:?}", t_embed_dec.dims());

    println!("\nRunning forward pass...");
    let logits = model.forward(mel, t_embed_dec);

    println!("\nOutput:");
    println!("  Logits shape: {:?}", logits.dims());
    println!("  Expected: [1, 20, {}]", config.decoder.vocab_size);

    // Verify shapes
    let dims = logits.dims();
    assert_eq!(dims[0], 1, "Batch size mismatch");
    assert_eq!(dims[1], 20, "Sequence length mismatch (320 / 4 / 4 = 20)");
    assert_eq!(dims[2], config.decoder.vocab_size, "Vocab size mismatch");

    println!("\n✓ End-to-end forward pass successful!");

    // Test streaming with KV cache
    println!("\n--- Testing streaming inference with KV cache ---\n");

    let mut encoder_cache = model.create_encoder_cache();
    let mut decoder_cache = model.create_decoder_cache();

    // Chunk 1: First 160 frames (after processing: 10 tokens)
    let mel_chunk1 = Tensor::<TestBackend, 3>::zeros([1, 128, 160], &device);
    let t_embed1 = Tensor::<TestBackend, 3>::zeros([1, 1, config.decoder.d_model], &device);

    println!("Processing chunk 1 ({} mel frames)...", 160);
    let logits1 =
        model.forward_with_cache(mel_chunk1, t_embed1, &mut encoder_cache, &mut decoder_cache);
    println!("  Output shape: {:?}", logits1.dims());
    println!("  Encoder cache seq_len: {}", encoder_cache.seq_len());
    println!("  Decoder cache seq_len: {}", decoder_cache.seq_len());

    // Chunk 2: Next 160 frames (incremental processing)
    let mel_chunk2 = Tensor::<TestBackend, 3>::zeros([1, 128, 160], &device);
    let t_embed2 = Tensor::<TestBackend, 3>::zeros([1, 1, config.decoder.d_model], &device);

    println!("\nProcessing chunk 2 ({} mel frames)...", 160);
    let logits2 =
        model.forward_with_cache(mel_chunk2, t_embed2, &mut encoder_cache, &mut decoder_cache);
    println!("  Output shape: {:?}", logits2.dims());
    println!("  Encoder cache seq_len: {}", encoder_cache.seq_len());
    println!("  Decoder cache seq_len: {}", decoder_cache.seq_len());

    println!("\n✓ Streaming inference with KV cache successful!");
}
