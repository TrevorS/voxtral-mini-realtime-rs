//! Integration test: f32 -> quantize -> Q4Tensor -> q4_matmul -> compare vs f32 matmul
//!
//! Round-trip tests that exercise the full Q4 pipeline from CPU-side quantization
//! through GPU kernel execution.

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use voxtral_mini_realtime::gguf::{q4_matmul, Q4Tensor};

type TestBackend = Wgpu;

// =============================================================================
// CPU-side Q4_0 helpers (duplicated from unit tests since integration tests
// cannot access #[cfg(test)] functions from the library)
// =============================================================================

const Q4_BLOCK_SIZE: usize = 32;
const Q4_BLOCK_BYTES: usize = 18;

fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Q4_BLOCK_SIZE, 0);
    let n_blocks = data.len() / Q4_BLOCK_SIZE;
    let mut output = Vec::with_capacity(n_blocks * Q4_BLOCK_BYTES);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q4_BLOCK_SIZE..(block_idx + 1) * Q4_BLOCK_SIZE];

        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = half::f16::from_f32(d);
        output.extend_from_slice(&d_f16.to_le_bytes());

        for i in 0..16 {
            let v0 = block[i];
            let v1 = block[i + 16];
            let q0 = ((v0 * id + 8.5) as u8).min(15);
            let q1 = ((v1 * id + 8.5) as u8).min(15);
            output.push(q0 | (q1 << 4));
        }
    }
    output
}

fn dequantize_q4_0_to_f32(q4_bytes: &[u8], n_elements: usize) -> Vec<f32> {
    assert_eq!(n_elements % Q4_BLOCK_SIZE, 0);
    let n_blocks = n_elements / Q4_BLOCK_SIZE;
    assert_eq!(q4_bytes.len(), n_blocks * Q4_BLOCK_BYTES);
    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let offset = block_idx * Q4_BLOCK_BYTES;
        let d_bits = u16::from_le_bytes([q4_bytes[offset], q4_bytes[offset + 1]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        let base = block_idx * Q4_BLOCK_SIZE;
        for i in 0..16 {
            let byte = q4_bytes[offset + 2 + i];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output[base + i] = lo * d;
            output[base + i + 16] = hi * d;
        }
    }
    output
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn test_q4_roundtrip_small() {
    let device = Default::default();

    let k = 64;
    let n = 64;
    let seq = 5;

    // Deterministic f32 data — weights in [N, K] layout (out_features, in_features)
    let weights_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.1)
        .collect();
    let act_f32: Vec<f32> = (0..seq * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.2)
        .collect();

    // Path A: Pure f32 matmul (ground truth)
    // weights_f32 is in [N, K] layout, transpose to [K, N] for Burn matmul
    let act_tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(act_f32, [1, seq, k]), &device);
    let weight_tensor =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(weights_f32.clone(), [n, k]), &device);
    let expected_f32 = act_tensor
        .clone()
        .matmul(weight_tensor.transpose().unsqueeze::<3>());

    // Path B: Q4 quantize -> GPU matmul
    let q4_bytes = quantize_f32_to_q4_0(&weights_f32);
    let q4_tensor =
        Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("Failed to create Q4Tensor");
    let output_q4 = q4_matmul(act_tensor, &q4_tensor);

    // Compare (includes full quantization noise)
    let out_data = output_q4.to_data();
    let exp_data = expected_f32.to_data();
    let out_slice = out_data.as_slice::<f32>().unwrap();
    let exp_slice = exp_data.as_slice::<f32>().unwrap();

    let mut max_diff: f32 = 0.0;
    let mut mean_diff: f32 = 0.0;
    for (a, b) in out_slice.iter().zip(exp_slice.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        mean_diff += diff;
    }
    mean_diff /= out_slice.len() as f32;

    println!(
        "Round-trip max diff: {:.4e}, mean diff: {:.4e}",
        max_diff, mean_diff
    );
    // K=64 with weights ~0.1 range: quantization noise accumulates
    assert!(
        max_diff < 0.5,
        "Round-trip max diff {:.4e} exceeds tolerance 0.5",
        max_diff
    );
}

#[test]
fn test_q4_roundtrip_model_shapes() {
    let device = Default::default();

    // Real decoder shapes (both dims must be multiples of 32)
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1, 3072, 3072, "attention_wq"),
        (10, 3072, 9216, "ffn_w1"),
        (1, 9216, 3072, "ffn_w2"),
    ];

    for &(seq, k, n, name) in shapes {
        println!(
            "Testing round-trip shape: {} [1x{}x{}] x [{}x{}]^T",
            name, seq, k, n, k
        );

        // Weights in [N, K] layout (out_features, in_features)
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.0001).sin() * 0.02)
            .collect();
        let acts: Vec<f32> = (0..seq * k)
            .map(|i| ((i as f32) * 0.0003).cos() * 0.1)
            .collect();

        let q4_bytes = quantize_f32_to_q4_0(&weights);

        let act_tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(acts, [1, seq, k]), &device);
        let q4_tensor =
            Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("Failed to create Q4Tensor");
        let output = q4_matmul(act_tensor, &q4_tensor);

        assert_eq!(output.dims(), [1, seq, n], "{} shape mismatch", name);
        println!("  {}: output shape {:?} -- OK", name, output.dims());
    }
}

#[test]
fn test_q4_roundtrip_vs_dequantized() {
    let device = Default::default();

    let k = 128;
    let n = 64;
    let seq = 8;

    // Weights in [N, K] layout (out_features, in_features)
    let weights_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.003).sin() * 0.05)
        .collect();
    let act_f32: Vec<f32> = (0..seq * k)
        .map(|i| ((i as f32) * 0.002).cos() * 0.1)
        .collect();

    // Quantize and dequantize to get the "true" Q4 weights
    let q4_bytes = quantize_f32_to_q4_0(&weights_f32);
    let weights_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);

    // Path A: f32 matmul with dequantized weights (isolates kernel correctness)
    // Dequantized weights are [N, K], transpose for Burn's standard matmul
    let act_tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(act_f32, [1, seq, k]), &device);
    let weight_deq_tensor =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(weights_deq, [n, k]), &device);
    let expected = act_tensor
        .clone()
        .matmul(weight_deq_tensor.transpose().unsqueeze::<3>());

    // Path B: Q4 GPU matmul
    let q4_tensor =
        Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).expect("Failed to create Q4Tensor");
    let output = q4_matmul(act_tensor, &q4_tensor);

    let out_data = output.to_data();
    let exp_data = expected.to_data();
    let out_slice = out_data.as_slice::<f32>().unwrap();
    let exp_slice = exp_data.as_slice::<f32>().unwrap();

    let mut max_diff: f32 = 0.0;
    for (a, b) in out_slice.iter().zip(exp_slice.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!(
        "Round-trip vs dequantized max diff: {:.4e} (kernel correctness)",
        max_diff
    );
    // This should be tiny: only GPU accumulation order differences
    assert!(
        max_diff < 1e-2,
        "Kernel correctness max diff {:.4e} exceeds tolerance 1e-2",
        max_diff
    );
}
