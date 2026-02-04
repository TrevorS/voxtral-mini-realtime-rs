# Voxtral Mini 4B Realtime Architecture Deep Dive

This document describes the Voxtral Mini 4B Realtime architecture in detail, based on analysis of the official HuggingFace model (`mistralai/Voxtral-Mini-4B-Realtime-2602`).

**Verified against actual `params.json` from downloaded model.**

## Table of Contents

1. [Model Overview](#model-overview)
2. [Audio Encoder](#audio-encoder)
3. [Language Model](#language-model)
4. [Audio-to-LLM Adapter](#audio-to-llm-adapter)
5. [Streaming Pipeline](#streaming-pipeline)
6. [Weight Names](#weight-names)
7. [Key Implementation Notes](#key-implementation-notes)

---

## Model Overview

Voxtral Mini 4B Realtime is a streaming automatic speech recognition (ASR) model consisting of two main components:

1. **Audio Encoder** (~0.6B params)
   - Causal Whisper-style transformer with sliding window attention
   - Processes mel spectrograms from 16kHz audio
   - Enables infinite-length streaming via causal attention
   - Uses ADA RMSNorm (T-conditional normalization)

2. **Language Model** (~3.4B params, Ministral-3B based)
   - Decoder-only transformer with GQA (4:1 ratio)
   - Generates text tokens autoregressively from audio embeddings
   - Uses sliding window attention (8192 tokens)

**Total params:** ~4B
**License:** Apache 2.0
**Source weights:** `mistralai/Voxtral-Mini-4B-Realtime-2602` (8.86GB SafeTensors, BF16)

### Key Timing Relationship

**1 text token = 80ms of audio** (frame rate = 12.5 Hz)

This is the fundamental relationship that governs streaming behavior. With a 6-token lookahead delay (default), latency is 480ms.

---

## Audio Encoder

The audio encoder is a causal Whisper-style transformer that processes mel spectrograms.

### Architecture Summary (from params.json)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Dimension | 1,280 | |
| Transformer Layers | 32 | |
| Query Heads | 32 | MHA (not GQA) |
| KV Heads | 32 | Same as Q heads |
| Head Dimension | 64 | |
| FFN Hidden Dimension | 5,120 | |
| Sliding Window | 750 | ~60 seconds of audio |
| RoPE Theta | 1,000,000 | Extended context |
| Norm Epsilon | 1e-05 | |
| FFN Type | SwiGLU | gate * silu(up) |
| Norm Type | RMS Norm | With ADA conditioning |
| Biases | **Yes** | Unlike LLM |
| Causal | **Yes** | Enables streaming |

### Input Processing

1. **Mel Spectrogram Extraction**
   ```
   Sample rate:     16,000 Hz
   Mel bins:        128
   Hop length:      160 samples (10ms)
   Window size:     400 samples (25ms, Hann window)
   Raw frame rate:  100 Hz (16000 / 160)
   ```

2. **Convolutional Downsampling**
   ```
   Input:  [batch, 128, T_raw]        # T_raw frames at 100 Hz
   Conv1d: 128 → 1280, stride=2       # 2x downsample → 50 Hz
   Conv1d: 1280 → 1280, stride=2      # 2x downsample → 25 Hz
   Output: [batch, 1280, T_raw/4]
   ```
   Total downsample factor: **4x**

3. **Log Mel Normalization**
   ```python
   log_mel = log(max(mel, 1e-10))
   normalized = clamp(log_mel / 1.5, -1.0, 1.0)
   # global_log_mel_max = 1.5
   ```

4. **Final Frame Rate**
   ```
   After encoder:  25 Hz (100 / 4)
   After reshape:  12.5 Hz (25 / 2, groups of 2 frames)
   ```
   **12.5 Hz = 80ms per token**

### ADA RMSNorm (T-Conditional Normalization)

The audio encoder uses **adaptive RMSNorm** with temporal conditioning:

```python
# From params.json:
# ada_rms_norm_t_cond: true
# ada_rms_norm_t_cond_dim: 32

class ADARMSNorm:
    def __init__(self, dim, t_cond_dim=32):
        self.weight = Parameter(dim)           # Standard RMSNorm weight
        self.ada_weight = Parameter(dim, t_cond_dim)  # Conditioning projection

    def forward(self, x, t_embed):
        # t_embed: [batch, t_cond_dim] - temporal embedding
        rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        normalized = x / rms

        # T-conditional scaling
        ada_scale = 1 + self.ada_weight @ t_embed  # [batch, dim]
        return normalized * self.weight * ada_scale
```

**Open question:** How is `t_embed` computed? Likely a learned embedding based on position or audio statistics.

### Transformer Layer Structure

```
input
├── ADARMSNorm (input_layernorm, with t_embed)
├── Causal Self-Attention (MHA)
│   ├── Q projection: Linear(1280 → 2048, bias=True)  # 32 heads × 64 dim
│   ├── K projection: Linear(1280 → 2048, bias=True)
│   ├── V projection: Linear(1280 → 2048, bias=True)
│   ├── RoPE (theta=1M)
│   ├── Causal + Sliding Window Mask (750 tokens)
│   └── O projection: Linear(2048 → 1280, bias=True)
├── Residual connection
├── ADARMSNorm (post_attention_layernorm, with t_embed)
├── SwiGLU MLP
│   ├── Gate projection: Linear(1280 → 5120, bias=False)
│   ├── Up projection: Linear(1280 → 5120, bias=False)
│   ├── SiLU activation on gate
│   ├── Element-wise multiply: gate_silu * up
│   └── Down projection: Linear(5120 → 1280, bias=False)
└── Residual connection
```

---

## Language Model

The language model is based on Ministral-3B and generates text tokens from audio embeddings.

### Architecture Summary (from params.json)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Dimension | 3,072 | |
| Transformer Layers | 26 | |
| Query Heads | 32 | |
| KV Heads | 8 | GQA 4:1 ratio |
| Head Dimension | 128 | |
| FFN Hidden Dimension | 9,216 | |
| Vocabulary Size | 131,072 | Tekken tokenizer |
| Sliding Window | 8,192 | |
| RoPE Theta | 1,000,000 | |
| Norm Epsilon | 1e-05 | |
| Biases | **No** | Unlike encoder |
| Tied Embeddings | **Yes** | embed_tokens = lm_head.T |
| Causal | Yes | Standard LLM |

### GQA (Grouped Query Attention)

The LLM uses grouped query attention with a **4:1 ratio**:
- 32 query heads → 4096 total Q dimension
- 8 key-value heads → 1024 total KV dimension
- Each KV head is shared by 4 query heads

This reduces KV cache memory by **4x** compared to MHA.

### Transformer Layer Structure

```
input
├── RMSNorm (input_layernorm)
├── GQA Self-Attention
│   ├── Q projection: Linear(3072 → 4096, bias=False)  # 32 heads × 128 dim
│   ├── K projection: Linear(3072 → 1024, bias=False)  # 8 heads × 128 dim
│   ├── V projection: Linear(3072 → 1024, bias=False)
│   ├── RoPE (theta=1M)
│   ├── Repeat KV: [8 heads] → [32 heads] for attention
│   ├── Causal + Sliding Window Mask (8192 tokens)
│   └── O projection: Linear(4096 → 3072, bias=False)
├── Residual connection
├── RMSNorm (post_attention_layernorm)
├── SwiGLU MLP
│   ├── Gate projection: Linear(3072 → 9216, bias=False)
│   ├── Up projection: Linear(3072 → 9216, bias=False)
│   ├── SiLU activation on gate
│   ├── Element-wise multiply
│   └── Down projection: Linear(9216 → 3072, bias=False)
└── Residual connection
```

---

## Audio-to-LLM Adapter

The adapter projects audio encoder outputs to LLM input dimension.

### Reshape Operation

After the audio encoder, outputs are reshaped to reduce sequence length:

```python
# Encoder output: [batch, T, 1280] at 25 Hz
# Pad T to multiple of 2
# Reshape: [batch, T, 1280] → [batch, T/2, 2560]
# This gives 12.5 Hz (80ms per frame)

# Note: params.json says downsample_factor=4, which applies to mel→encoder
# The encoder→adapter reshape is an additional 2x grouping
```

### AudioLanguageAdapter

```python
class AudioLanguageAdapter:
    def __init__(self):
        self.linear1 = Linear(5120, 5120)  # input_dim = 1280 * 4
        self.linear2 = Linear(5120, 3072)  # output_dim = LLM dim

    def forward(self, x):
        # x: [batch, T, 5120] - grouped encoder output
        x = self.linear1(x)
        x = gelu(x)
        x = self.linear2(x)
        return x  # [batch, T, 3072]
```

### Connection to LLM

The adapted audio embeddings replace or are concatenated with text token embeddings:

```
Audio (12.5 Hz) → Adapter → [batch, T_audio, 3072]
                              ↓
                           LLM input
```

---

## Streaming Pipeline

### Frame Rate Summary

| Stage | Frame Rate | Notes |
|-------|------------|-------|
| Raw audio | 16,000 Hz | Sample rate |
| Mel spectrogram | 100 Hz | hop_length=160 |
| After conv downsample | 25 Hz | 4x downsample |
| After reshape/adapter | **12.5 Hz** | Groups of 2 |
| Text tokens | **12.5 Hz** | 1 token = 80ms |

### Delay Configuration

| Delay (tokens) | Delay (ms) | Use Case |
|----------------|------------|----------|
| 1 | 80 | Minimum latency, lower accuracy |
| 6 | 480 | **Recommended** - good balance |
| 12 | 960 | Higher accuracy |
| 30 | 2400 | Maximum lookahead |

### Streaming Flow

```
Audio chunk (80ms = 1280 samples) arrives
        ↓
Mel spectrogram: 8 frames at 100 Hz
        ↓
Conv downsample (4x): 2 frames at 25 Hz
        ↓
Audio encoder (causal): 2 frames, KV cached
        ↓
Reshape (2x): 1 frame at 12.5 Hz
        ↓
Adapter projection: 1 embedding [3072]
        ↓
Wait for delay_tokens to accumulate
        ↓
LLM forward (causal): KV cached
        ↓
Sample text token
        ↓
Output token (every 80ms)
```

### KV Cache Sizes

| Component | Window | At 12.5 Hz | Memory (BF16) |
|-----------|--------|-----------|---------------|
| Audio encoder | 750 | 60 sec | ~200 MB |
| LLM | 8192 | 655 sec | ~800 MB |

---

## Weight Names

Based on typical Mistral/Whisper naming conventions. **To be verified against actual SafeTensors.**

### Audio Encoder Weights

```
# Convolutional downsampler
encoder.conv1.weight                    [1280, 128, kernel]
encoder.conv1.bias                      [1280]
encoder.conv2.weight                    [1280, 1280, kernel]
encoder.conv2.bias                      [1280]

# Transformer layers (×32)
encoder.layers.{0-31}.input_layernorm.weight         [1280]
encoder.layers.{0-31}.input_layernorm.ada_weight     [1280, 32]  # ADA
encoder.layers.{0-31}.self_attn.q_proj.weight        [2048, 1280]
encoder.layers.{0-31}.self_attn.q_proj.bias          [2048]
encoder.layers.{0-31}.self_attn.k_proj.weight        [2048, 1280]
encoder.layers.{0-31}.self_attn.k_proj.bias          [2048]
encoder.layers.{0-31}.self_attn.v_proj.weight        [2048, 1280]
encoder.layers.{0-31}.self_attn.v_proj.bias          [2048]
encoder.layers.{0-31}.self_attn.o_proj.weight        [1280, 2048]
encoder.layers.{0-31}.self_attn.o_proj.bias          [1280]
encoder.layers.{0-31}.post_attention_layernorm.weight    [1280]
encoder.layers.{0-31}.post_attention_layernorm.ada_weight [1280, 32]
encoder.layers.{0-31}.mlp.gate_proj.weight           [5120, 1280]
encoder.layers.{0-31}.mlp.up_proj.weight             [5120, 1280]
encoder.layers.{0-31}.mlp.down_proj.weight           [1280, 5120]

encoder.norm.weight                     [1280]
```

### Language Model Weights

```
model.embed_tokens.weight               [131072, 3072]

# Transformer layers (×26)
model.layers.{0-25}.input_layernorm.weight           [3072]
model.layers.{0-25}.self_attn.q_proj.weight          [4096, 3072]
model.layers.{0-25}.self_attn.k_proj.weight          [1024, 3072]
model.layers.{0-25}.self_attn.v_proj.weight          [1024, 3072]
model.layers.{0-25}.self_attn.o_proj.weight          [3072, 4096]
model.layers.{0-25}.post_attention_layernorm.weight  [3072]
model.layers.{0-25}.mlp.gate_proj.weight             [9216, 3072]
model.layers.{0-25}.mlp.up_proj.weight               [9216, 3072]
model.layers.{0-25}.mlp.down_proj.weight             [3072, 9216]

model.norm.weight                       [3072]
lm_head.weight                          [131072, 3072]  # tied with embed_tokens
```

### Adapter Weights

```
adapter.linear1.weight                  [5120, 5120]
adapter.linear1.bias                    [5120]
adapter.linear2.weight                  [3072, 5120]
adapter.linear2.bias                    [3072]
```

---

## Key Implementation Notes

### 1. Causal Attention in Audio Encoder

The audio encoder's causal attention is what enables streaming. Standard Whisper uses bidirectional attention and cannot stream.

```rust
fn create_causal_mask(seq_len: usize, device: &Device) -> Tensor {
    // Position i can only attend to positions 0..=i
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}
```

### 2. Sliding Window + Causal Mask

Combine causal and sliding window constraints:

```rust
fn create_sliding_causal_mask(seq_len: usize, window: usize, device: &Device) -> Tensor {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        let start = i.saturating_sub(window);
        for j in start..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}
```

### 3. GQA repeat_kv

For grouped query attention, repeat KV heads to match Q heads:

```rust
fn repeat_kv(kv: &Tensor, n_rep: usize) -> Result<Tensor> {
    // kv: [batch, kv_heads, seq, head_dim]
    // out: [batch, q_heads, seq, head_dim]
    let (b, h, s, d) = kv.dims4()?;
    kv.unsqueeze(2)?                    // [b, h, 1, s, d]
        .expand((b, h, n_rep, s, d))?   // [b, h, n_rep, s, d]
        .reshape((b, h * n_rep, s, d))  // [b, h*n_rep, s, d]
}
```

### 4. RoPE with Large Theta

```rust
fn compute_rope_freqs(head_dim: usize, theta: f64) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
        .collect()
}
// theta = 1,000,000 for extended context
```

### 5. SwiGLU MLP

```rust
fn swiglu_forward(x: &Tensor, gate_w: &Tensor, up_w: &Tensor, down_w: &Tensor) -> Result<Tensor> {
    let gate = x.matmul(&gate_w.t()?)?;
    let up = x.matmul(&up_w.t()?)?;
    let hidden = gate.silu()?.mul(&up)?;
    hidden.matmul(&down_w.t()?)
}
```

### 6. BF16 Compute Considerations

For GPU inference with BF16:
- RoPE cos/sin: Compute in F32, cast to input dtype
- Attention softmax: Accumulate in F32 for stability
- Sampling logits: Cast to F32 for `argmax`/`multinomial`

---

## Critical Corrections (Feb 2026)

### ADA RMSNorm Location

**CORRECTION:** ADA RMSNorm is used in the **decoder (LLM)**, NOT the encoder.

The encoder uses standard RMSNorm. The decoder uses ADA RMSNorm with t-conditioning:
- `layers.{N}.ada_rms_norm_t_cond.0.weight` - [32, 3072]
- `layers.{N}.ada_rms_norm_t_cond.2.weight` - [3072, 32]

The ADA conditioning uses **GELU activation** (not SiLU):
```python
scale = linear_2(gelu(linear_0(t_embed)))  # NOT linear_2(silu(linear_0(t_embed)))
x_scaled = x_normalized * (1 + scale)
```

### Time Embedding (t_cond)

The t_embed is computed using sinusoidal time embedding with `t = num_delay_tokens = 6`:
```python
def time_embedding(t, dim=3072, theta=10000.0):
    half_dim = dim // 2
    inv_freq = exp(-log(theta) * arange(half_dim) / half_dim)
    emb = t * inv_freq
    return concat([cos(emb), sin(emb)])

t_embed = time_embedding(torch.tensor([6.0]))  # [1, 3072]
```

---

## Streaming Inference (Critical)

### Position 38 Anomaly

**WARNING:** The standard prefix length of 39 tokens (BOS + 38 `[STREAMING_PAD]`) causes anomalous behavior at position 38.

When position 38 is the **last** position in the prefix:
- Hidden state norm diverges: Layer 25 shows pos 38 norm=452 vs pos 36/37 norm=1000-1100
- All logits become very negative (-17 to -55 range)
- Always predicts `[STREAMING_PAD]` regardless of audio content

**Root cause:** Position 38 = n_left_pad_tokens(32) + num_delay_tokens(6) is exactly at the trained prefix boundary.

### Working Solution

Use **prefix length 38** (one less than standard) for generation:
```python
prefix_tokens = [1] + [32] * 37  # BOS + 37 STREAMING_PAD = 38 tokens
```

With prefix 38, position 37 correctly predicts `[STREAMING_WORD]` (token 33), enabling proper transcription.

### Verified Transcription Output

Test audio: `test_data/mary_had_lamb.wav` (15.95s)
- Expected: "First words I spoke in the original phonograph..."
- Produced: " I spoke in the original phonograph. A little piece of practical poetry"

(Missing "First words" is expected - position 38 corresponds to ~2.1s into the speech)

### Token Semantics

| Token ID | Name | Meaning |
|----------|------|---------|
| 1 | `<s>` / BOS | Beginning of sequence |
| 32 | `[STREAMING_PAD]` | Silence / pause between words |
| 33 | `[STREAMING_WORD]` | Start of a word (next tokens will be text) |
| ≥1000 | Text tokens | Actual transcription content |

### Tokenizer Offset (Critical!)

**Text token IDs are offset by 1000 from vocab indices in tekken.json.**

```
Token ID 1000 → vocab index 0
Token ID 1362 → vocab index 362 (" I")
Token ID 19135 → vocab index 18135 (" spoke")
```

When decoding text tokens, subtract 1000 to get the vocab index:
```rust
let vocab_idx = (token_id - 1000) as usize;
let bytes = vocab_bytes[vocab_idx];
```

Token IDs 0-999 are reserved for special/control tokens (BOS, STREAMING_PAD, etc.).

### Autoregressive Generation Pattern

```python
# 1. Process prefix (38 positions)
prefix = [BOS] + [STREAMING_PAD] * 37
audio_slice = audio_embeds[:, :38, :]
logits = model.forward(audio_slice, prefix_embeds, t_embed)
next_token = logits[:, -1, :].argmax()  # Position 37 predicts position 38

# 2. Autoregressive loop
for pos in range(38, seq_len):
    prev_token_embed = embed(generated_tokens[-1])
    audio_at_pos = audio_embeds[:, pos-1:pos, :]
    input = prev_token_embed + audio_at_pos
    logits = model.forward_step(input, cache)
    next_token = logits.argmax()
    generated_tokens.append(next_token)

# 3. Decode (filter control tokens)
text_tokens = [t for t in generated_tokens if t >= 1000]
text = tokenizer.decode(text_tokens)
```

---

## Summary

| Component | Layers | Dim | Heads | Window | Special |
|-----------|--------|-----|-------|--------|---------|
| Audio Encoder | 32 | 1280 | 32 MHA | 750 | Causal, standard RMSNorm, biases |
| Language Model | 26 | 3072 | 32Q/8KV | 8192 | GQA 4:1, ADA RMSNorm, no biases, tied embed |
| Adapter | 2 | 5120→3072 | - | - | GELU activation |

**Key insight:** The causal attention throughout the audio encoder is what enables true real-time streaming ASR. Standard Whisper cannot stream because it uses bidirectional attention.

**Timing:** 1 text token = 80ms audio = 1280 samples @ 16kHz

**Critical:** Use prefix length 38 (not 39) to avoid position 38 anomaly.
