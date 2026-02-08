//! CPU micro-benchmarks for audio preprocessing (no model weights needed).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f32::consts::PI;
use std::hint::black_box;

use voxtral_mini_realtime::audio::mel::{MelConfig, MelSpectrogram};
use voxtral_mini_realtime::audio::pad::{pad_audio, PadConfig};
use voxtral_mini_realtime::audio::resample::resample_to_16k;
use voxtral_mini_realtime::audio::AudioBuffer;

/// Generate a 16kHz sine wave of the given duration in seconds.
fn sine_16k(duration_secs: f32) -> Vec<f32> {
    let n = (16000.0 * duration_secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect()
}

/// Generate a 48kHz sine wave of the given duration in seconds.
fn sine_48k(duration_secs: f32) -> Vec<f32> {
    let n = (48000.0 * duration_secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
        .collect()
}

fn bench_mel_spectrogram(c: &mut Criterion) {
    let mel = MelSpectrogram::new(MelConfig::voxtral());

    let mut group = c.benchmark_group("mel_spectrogram");
    for duration in [1.0, 5.0, 15.0, 30.0] {
        let samples = sine_16k(duration);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration}s")),
            &samples,
            |b, samples| {
                b.iter(|| mel.compute_log(black_box(samples)));
            },
        );
    }
    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_48k_to_16k");
    for duration in [1.0, 5.0, 15.0] {
        let samples = sine_48k(duration);
        let audio = AudioBuffer::new(samples, 48000);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration}s")),
            &audio,
            |b, audio| {
                b.iter(|| resample_to_16k(black_box(audio)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_pad_audio(c: &mut Criterion) {
    let config = PadConfig::voxtral();
    let samples = sine_16k(15.0);
    let audio = AudioBuffer::new(samples, 16000);

    c.bench_function("pad_audio_15s", |b| {
        b.iter(|| pad_audio(black_box(&audio), black_box(&config)));
    });
}

criterion_group!(
    benches,
    bench_mel_spectrogram,
    bench_resample,
    bench_pad_audio
);
criterion_main!(benches);
