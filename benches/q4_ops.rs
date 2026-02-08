//! GPU Q4 kernel micro-benchmarks (no model weights needed).
//!
//! Benchmarks `q4_matmul` at real model shapes using synthetic Q4_0 data.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

use voxtral_mini_realtime::gguf::{q4_matmul, Q4Tensor};

const Q4_BLOCK_SIZE: usize = 32;

/// Quantize f32 data to Q4_0 format (test helper, mirrors src/gguf/tests.rs).
fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Q4_BLOCK_SIZE, 0);
    let n_blocks = data.len() / Q4_BLOCK_SIZE;
    let mut output = Vec::with_capacity(n_blocks * 18);

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

/// Prepare a Q4Tensor from random-ish f32 data at the given shape [N, K].
fn make_q4_weights(
    n: usize,
    k: usize,
    device: &<Wgpu as burn::tensor::backend::Backend>::Device,
) -> Q4Tensor {
    let weight_data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0007).cos() * 0.05)
        .collect();
    let q4_bytes = quantize_f32_to_q4_0(&weight_data);
    Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], device).expect("Failed to create Q4Tensor")
}

fn bench_q4_matmul(c: &mut Criterion) {
    let device: <Wgpu as burn::tensor::backend::Backend>::Device = Default::default();

    // (batch, seq, K, N, description)
    let shapes: &[(usize, usize, usize, usize, &str)] = &[
        (1, 1, 3072, 3072, "dec_attn_wq_1tok"),
        (1, 38, 3072, 3072, "dec_attn_wq_prefill"),
        (1, 1, 3072, 9216, "dec_ffn_w1_1tok"),
        (1, 38, 3072, 9216, "dec_ffn_w1_prefill"),
        (1, 1, 1280, 5120, "enc_ffn_w1"),
        (1, 100, 1280, 1280, "enc_attn_wq_100pos"),
    ];

    let mut group = c.benchmark_group("q4_matmul");

    for &(batch, seq, k, n, desc) in shapes {
        let q4_weights = make_q4_weights(n, k, &device);
        let act_data: Vec<f32> = (0..batch * seq * k)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{desc}_[{batch},{seq},{k}]x[{n},{k}]")),
            &(),
            |b, _| {
                b.iter(|| {
                    let activations = Tensor::<Wgpu, 3>::from_data(
                        TensorData::new(act_data.clone(), [batch, seq, k]),
                        &device,
                    );
                    let output = q4_matmul(black_box(activations), &q4_weights);
                    // Force GPU sync by reading output data
                    output.to_data()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_q4_matmul);
criterion_main!(benches);
