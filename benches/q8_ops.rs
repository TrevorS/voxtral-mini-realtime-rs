//! GPU Q8 kernel micro-benchmarks (no model weights needed).
//!
//! Benchmarks `q8_matmul` at real model shapes using synthetic Q8_0 data,
//! mirroring `q4_ops.rs` so Q4_0 and Q8_0 throughput can be compared directly.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

use voxtral_mini_realtime::gguf::{q8_matmul, Q4Tensor};

const Q8_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;

/// Quantize f32 data to Q8_0 format (test helper, mirrors src/gguf/tests.rs).
fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Q8_BLOCK_SIZE, 0);
    let n_blocks = data.len() / Q8_BLOCK_SIZE;
    let mut output = Vec::with_capacity(n_blocks * Q8_BLOCK_BYTES);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * Q8_BLOCK_SIZE..(block_idx + 1) * Q8_BLOCK_SIZE];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = half::f16::from_f32(d);
        output.extend_from_slice(&d_f16.to_le_bytes());

        for &v in block {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            output.push(q as u8);
        }
    }
    output
}

/// Prepare a Q8 tensor from random-ish f32 data at the given shape [N, K].
fn make_q8_weights(
    n: usize,
    k: usize,
    device: &<Wgpu as burn::tensor::backend::Backend>::Device,
) -> Q4Tensor {
    let weight_data: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0007).cos() * 0.05)
        .collect();
    let q8_bytes = quantize_f32_to_q8_0(&weight_data);
    Q4Tensor::from_q8_bytes(&q8_bytes, [n, k], device).expect("Failed to create Q8 tensor")
}

fn bench_q8_matmul(c: &mut Criterion) {
    let device: <Wgpu as burn::tensor::backend::Backend>::Device = Default::default();

    // (batch, seq, K, N, description) — same shapes as q4_ops.rs
    let shapes: &[(usize, usize, usize, usize, &str)] = &[
        (1, 1, 3072, 3072, "dec_attn_wq_1tok"),
        (1, 38, 3072, 3072, "dec_attn_wq_prefill"),
        (1, 1, 3072, 9216, "dec_ffn_w1_1tok"),
        (1, 38, 3072, 9216, "dec_ffn_w1_prefill"),
        (1, 1, 1280, 5120, "enc_ffn_w1"),
        (1, 100, 1280, 1280, "enc_attn_wq_100pos"),
    ];

    let mut group = c.benchmark_group("q8_matmul");

    for &(batch, seq, k, n, desc) in shapes {
        let q8_weights = make_q8_weights(n, k, &device);
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
                    let output = q8_matmul(black_box(activations), &q8_weights);
                    // Force GPU sync by reading output data
                    output.to_data()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_q8_matmul);
criterion_main!(benches);
