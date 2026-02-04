# Validation Strategy

Component-by-component validation of the Rust implementation against Python reference.

## Validation Approach

1. **Export Python reference values** at each stage
2. **Run Rust implementation** with same inputs
3. **Compare outputs** with tolerance thresholds
4. **Track max differences** to catch regressions

## Target Tolerances

| Precision | Typical Max Diff | Notes |
|-----------|-----------------|-------|
| F32 exact | < 1e-6 | Pure math operations |
| F32 accumulated | < 1e-4 | Multi-layer forward passes |
| BF16 compute | < 1e-3 | Expected precision loss |

## Component Validation Checklist

### Audio Processing

| Component | Status | Max Diff | Notes |
|-----------|--------|----------|-------|
| Hann window | ✅ | < 1e-6 | Pure Rust FFT |
| FFT | ✅ | < 1e-5 | rustfft crate |
| Mel filterbank | ✅ | < 1e-3 | Slaney normalization |
| Mel spectrogram | ✅ | < 1e-3 | global_log_mel_max=1.5 |
| Log normalization | ✅ | exact | Fixed value normalization |

### Audio Encoder

| Component | Status | Max Diff | Notes |
|-----------|--------|----------|-------|
| Conv1d layer 1 | ✅ | < 1e-4 | 128→1280, stride=2 |
| Conv1d layer 2 | ✅ | < 1e-4 | 1280→1280, stride=2 |
| RMSNorm | ✅ | < 1e-5 | Standard variant |
| ADA RMSNorm | ✅ | < 1e-4 | GELU activation (not SiLU!) |
| RoPE | ✅ | < 1e-5 | theta=1M, interleaved layout |
| Causal attention | ✅ | < 1e-4 | MHA with sliding window 750 |
| SwiGLU MLP | ✅ | < 1e-4 | gate * silu(up), with biases |
| Single layer | ✅ | < 1e-3 | Full layer forward |
| 32-layer stack | ✅ | < 1e-2 | Matches Python reference |

### Language Model

| Component | Status | Max Diff | Notes |
|-----------|--------|----------|-------|
| Token embedding | ✅ | exact | vocab=131072, tied weights |
| RMSNorm | ✅ | < 1e-5 | eps=1e-5 |
| ADA RMSNorm (decoder) | ✅ | < 1e-4 | GELU, t_cond from TimeEmbedding |
| RoPE | ✅ | < 1e-5 | theta=1M, head_dim=128 |
| GQA attention | ✅ | < 1e-4 | 32Q/8KV with repeat_kv |
| Sliding window mask | ✅ | exact | 8192 tokens |
| SwiGLU MLP | ✅ | < 1e-4 | hidden=9216, no biases |
| Single layer | ✅ | < 1e-3 | Full layer forward |
| 26-layer stack | ✅ | < 1e-2 | Matches Python reference |
| LM head | ✅ | exact | Tied with embeddings |

### Integration

| Component | Status | Max Diff | Notes |
|-----------|--------|----------|-------|
| Audio reshape | ✅ | exact | [T, 1280] → [T/4, 5120] |
| Adapter projection | ✅ | < 1e-4 | Linear → GELU → Linear |
| KV cache | ✅ | exact | Concat-based, window eviction |
| TimeEmbedding | ✅ | < 1e-5 | Sinusoidal, t=6.0 |
| End-to-end forward | ✅ | < 1e-2 | Audio → logits |
| **Transcription** | ✅ | **WORKING** | Python verified, Rust ready |

### Streaming Inference

| Component | Status | Notes |
|-----------|--------|-------|
| Position 38 anomaly | ✅ Identified | Use prefix 38 not 39 |
| Autoregressive loop Python | ✅ Complete | Verified correct transcription |
| Autoregressive loop Rust | ✅ Complete | Output matches Python exactly |
| Tokenizer decoding | ✅ Fixed | Text tokens offset by 1000 from vocab indices |
| Left-pad audio alignment | ✅ Complete | `scripts/generate_padded_reference.py` |
| KV cache streaming | 🔲 Pending | Future optimization work |

## Test Data Requirements

### Reference Inputs

```
test_data/
├── audio/
│   ├── short.wav         # 1-2 seconds
│   ├── medium.wav        # 5-10 seconds
│   └── long.wav          # 30+ seconds (streaming test)
├── mel/
│   ├── short_mel.npy     # Pre-computed mel spectrogram
│   └── short_mel_log.npy # Log-normalized
└── reference/
    ├── audio_encoder_layer0.npy
    ├── audio_encoder_final.npy
    ├── adapter_output.npy
    ├── llm_layer0.npy
    ├── llm_final.npy
    └── logits.npy
```

### Python Export Script

```python
import torch
import numpy as np
from transformers import AutoModel

def export_reference_values(model_path, audio_path, output_dir):
    model = AutoModel.from_pretrained(model_path)

    # Load and process audio
    audio = load_audio(audio_path)
    mel = compute_mel(audio)

    # Export mel
    np.save(f"{output_dir}/mel.npy", mel.numpy())

    # Forward through encoder with hooks
    encoder_outputs = []
    def hook(module, input, output):
        encoder_outputs.append(output.detach().cpu().numpy())

    for i, layer in enumerate(model.audio_encoder.layers):
        layer.register_forward_hook(hook)

    encoder_out = model.audio_encoder(mel)

    for i, out in enumerate(encoder_outputs):
        np.save(f"{output_dir}/audio_encoder_layer{i}.npy", out)

    np.save(f"{output_dir}/audio_encoder_final.npy",
            encoder_out.detach().cpu().numpy())

    # Continue for adapter, LLM layers, etc.
```

## Running Validation

### 1. Export Reference Values

```bash
cd tools
uv sync
uv run python export_reference_values.py \
    --model-path path/to/voxtral \
    --audio test_data/audio/short.wav \
    --output test_data/reference/
```

### 2. Run Rust Validation Tests

```bash
cargo test --test reference_validation -- --nocapture
```

### 3. Check Results

```
=== Audio Encoder Layer 0 ===
  max_diff: 2.3e-6
  PASS

=== Audio Encoder Full ===
  max_diff: 4.1e-5
  PASS

=== Adapter ===
  max_diff: 1.2e-5
  PASS

...
```

## Known Implementation Pitfalls

### 1. RoPE Formula Variants

There are multiple RoPE implementations. Voxtral uses the standard formula:

```rust
// Correct: interleaved rotation
rotated[2i] = x[2i] * cos - x[2i+1] * sin
rotated[2i+1] = x[2i+1] * cos + x[2i] * sin
```

### 2. Attention Mask Dtype

Mask values should match attention weight dtype:

```rust
let mask = mask.to_dtype(attn_weights.dtype())?;
```

### 3. Sliding Window Implementation

The sliding window mask is **relative** to current position:

```rust
// Position i can attend to positions max(0, i - window_size) .. i+1
fn sliding_window_mask(seq_len: usize, window_size: usize) -> Tensor {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        let start = i.saturating_sub(window_size);
        for j in start..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device)?
}
```

### 4. GQA repeat_kv

When repeating KV heads for GQA:

```rust
// k, v: [batch, kv_heads, seq, head_dim]
// Need: [batch, q_heads, seq, head_dim]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?                    // [b, h, 1, s, d]
        .expand((b, h, n_rep, s, d))?  // [b, h, n_rep, s, d]
        .reshape((b, h * n_rep, s, d)) // [b, h*n_rep, s, d]
}
```

### 5. Causal Mask Combination

When combining causal + sliding window masks:

```rust
// Both masks use -inf for blocked positions, 0 for allowed
let combined = causal_mask.maximum(&sliding_mask)?;
// OR just use element-wise addition if both use -inf
```

## WER Evaluation

For end-to-end validation, compute Word Error Rate:

```bash
# Generate transcription
cargo run --release --features cli -- \
    --audio test_data/audio/librispeech_sample.wav \
    --model path/to/voxtral \
    > rust_transcription.txt

# Compare with ground truth
python compute_wer.py \
    --reference test_data/transcripts/librispeech_sample.txt \
    --hypothesis rust_transcription.txt
```

Target: WER within 0.1% of Python reference on same audio.

## Streaming Validation

For streaming mode, verify:

1. **Incremental consistency**: Output at time T should match batch mode output for audio[0:T]
2. **Latency**: First token arrives within delay_tokens × 80ms
3. **Memory**: KV cache doesn't grow unbounded

```rust
#[test]
fn test_streaming_consistency() {
    let model = load_model();
    let audio = load_audio("test.wav");

    // Batch mode
    let batch_output = model.transcribe(&audio, None)?;

    // Streaming mode
    let mut streaming_output = String::new();
    for chunk in audio.chunks(1280) {  // 80ms chunks
        if let Some(token) = model.process_chunk(chunk)? {
            streaming_output.push_str(&token);
        }
    }

    assert_eq!(batch_output, streaming_output);
}
```
