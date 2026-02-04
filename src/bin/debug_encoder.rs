//! Debug encoder components.
//!
//! Loads mel and runs through encoder components one at a time.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use voxtral_mini_realtime::models::loader::VoxtralModelLoader;

type TestBackend = Wgpu;

fn main() {
    let model_path = "models/voxtral/consolidated.safetensors";

    // Check required files exist
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Error: Model not found at {}", model_path);
        std::process::exit(1);
    }

    let device = Default::default();

    // Load reference mel from Python
    println!("Loading reference mel...");
    use ndarray::ArrayD;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;

    let file = File::open("test_data/reference_mel.npy").expect("Failed to open npy file");
    let mel_arr: ArrayD<f32> = ArrayD::<f32>::read_npy(file).expect("Failed to read npy file");
    let shape = mel_arr.shape();
    println!("  Mel shape: {:?}", shape);

    let n_mels = shape[0];
    let n_frames = shape[1];

    // Convert to [1, n_mels, n_frames] tensor
    let mel_data: Vec<f32> = mel_arr.iter().cloned().collect();
    let mel_tensor: Tensor<TestBackend, 3> = Tensor::from_data(
        burn::tensor::TensorData::new(mel_data, [1, n_mels, n_frames]),
        &device,
    );

    println!("  Mel tensor shape: {:?}", mel_tensor.dims());

    // Load model
    println!("\nLoading model...");
    let loader = VoxtralModelLoader::from_file(model_path).expect("Failed to create model loader");
    let model = loader
        .load::<TestBackend>(&device)
        .expect("Failed to load model");

    // Get encoder reference (we need to expose it for this debug)
    // For now, just run encode_audio and check the output

    println!("\nRunning encode_audio...");
    let audio_embeds = model.encode_audio(mel_tensor);

    let ae_data = audio_embeds.to_data();
    let ae_slice = ae_data.as_slice::<f32>().unwrap();
    let ae_min = ae_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let ae_max = ae_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ae_mean: f32 = ae_slice.iter().sum::<f32>() / ae_slice.len() as f32;

    println!("  Audio embeds shape: {:?}", audio_embeds.dims());
    println!(
        "  Stats: min={:.4}, max={:.4}, mean={:.4}",
        ae_min, ae_max, ae_mean
    );
    println!(
        "  First 10 values: {:?}",
        &ae_slice[..10.min(ae_slice.len())]
    );

    // Save for comparison with Python
    use ndarray::Array2;
    let [_, seq_len, dim] = audio_embeds.dims();
    let audio_arr: Array2<f32> = Array2::from_shape_vec((seq_len, dim), ae_slice.to_vec()).unwrap();

    use ndarray_npy::WriteNpyExt;
    let mut file =
        File::create("test_data/rust_encoder_output.npy").expect("Failed to create file");
    audio_arr.write_npy(&mut file).expect("Failed to write npy");
    println!("\nSaved Rust encoder output to test_data/rust_encoder_output.npy");
}
