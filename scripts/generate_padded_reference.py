#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mistral-common[soxr]>=1.9.0",
#     "torch",
#     "safetensors",
#     "soundfile",
#     "numpy",
# ]
# ///
"""
Generate properly padded reference data for Rust inference testing.

mistral-common LEFT-PADS the audio with silence to align with the streaming
prefix tokens. This script generates:
1. reference_mel_padded.npy - Mel spectrogram from padded audio
2. reference_audio_embeds.npy - Encoder output for validation
3. reference_prefix_info.txt - Prefix token info
"""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.transcription.request import TranscriptionRequest, StreamingMode
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.audio import Audio

MODEL_PATH = "models/voxtral/consolidated.safetensors"
TOKENIZER_PATH = "models/voxtral/tekken.json"
TEST_DATA_DIR = Path("test_data")

def compute_mel(audio_array, sampling_rate=16000):
    """Compute mel spectrogram using mistral-common's approach."""
    from mistral_common.audio import mel_filter_bank

    window_size = 400
    hop_length = 160
    num_mel_bins = 128
    global_log_mel_max = 1.5

    audio_tensor = torch.from_numpy(audio_array).float()
    window = torch.hann_window(window_size)
    stft = torch.stft(
        audio_tensor,
        window_size,
        hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + window_size // 2,
        num_mel_bins=num_mel_bins,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sampling_rate,
    )
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32)
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec_max = torch.tensor(global_log_mel_max)
    log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec

def rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

def rope_freqs(dim, max_seq_len, theta=1_000_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out

def run_encoder(mel, f):
    """Run mel through encoder layers."""
    def load_weight(name):
        return f.get_tensor(name).float()

    mel = mel.unsqueeze(0)  # [1, 128, T]

    # Conv layers
    conv1_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight")
    conv1_b = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias")
    conv2_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight")
    conv2_b = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.bias")

    x = F.gelu(F.conv1d(mel, conv1_w, conv1_b, stride=2, padding=1))
    x = F.gelu(F.conv1d(x, conv2_w, conv2_b, stride=2, padding=1))
    x = x.transpose(1, 2)  # [B, T, C]

    # Encoder layers
    n_heads = 32
    head_dim = 64

    for layer_idx in range(32):
        prefix = f"mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{layer_idx}"

        attn_norm_w = load_weight(f"{prefix}.attention_norm.weight")
        wq = load_weight(f"{prefix}.attention.wq.weight")
        wk = load_weight(f"{prefix}.attention.wk.weight")
        wv = load_weight(f"{prefix}.attention.wv.weight")
        wo = load_weight(f"{prefix}.attention.wo.weight")
        wq_b = load_weight(f"{prefix}.attention.wq.bias")
        wv_b = load_weight(f"{prefix}.attention.wv.bias")
        wo_b = load_weight(f"{prefix}.attention.wo.bias")
        ffn_norm_w = load_weight(f"{prefix}.ffn_norm.weight")
        w1 = load_weight(f"{prefix}.feed_forward.w1.weight")
        w2 = load_weight(f"{prefix}.feed_forward.w2.weight")
        w3 = load_weight(f"{prefix}.feed_forward.w3.weight")
        w2_b = load_weight(f"{prefix}.feed_forward.w2.bias")

        seq_len = x.shape[1]
        batch = x.shape[0]

        # Attention
        residual = x.clone()
        x_normed = rms_norm(x, attn_norm_w)
        q = F.linear(x_normed, wq, wq_b)
        k = F.linear(x_normed, wk)
        v = F.linear(x_normed, wv, wv_b)

        q = q.view(batch, seq_len, n_heads, head_dim)
        k = k.view(batch, seq_len, n_heads, head_dim)
        v = v.view(batch, seq_len, n_heads, head_dim)

        cos, sin = rope_freqs(head_dim, seq_len)
        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        q_t = q_rope.transpose(1, 2)
        k_t = k_rope.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = F.linear(out, wo, wo_b)
        x = attn_out + residual

        # FFN
        residual = x.clone()
        x_normed = rms_norm(x, ffn_norm_w)
        mlp_out = F.linear(F.silu(F.linear(x_normed, w1)) * F.linear(x_normed, w3), w2, w2_b)
        x = mlp_out + residual

    # Final norm
    encoder_norm_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight")
    x = rms_norm(x, encoder_norm_w)

    # Reshape (4x pooling)
    seq_len = x.shape[1]
    new_seq = seq_len // 4
    x = x[:, :new_seq*4, :].reshape(1, new_seq, 5120)

    # Adapter
    adapter_w_in = load_weight("mm_streams_embeddings.embedding_module.audio_language_projection.0.weight")
    adapter_w_out = load_weight("mm_streams_embeddings.embedding_module.audio_language_projection.2.weight")
    x = F.gelu(F.linear(x, adapter_w_in))
    x = F.linear(x, adapter_w_out)

    return x

def main():
    audio_path = "test_data/mary_had_lamb.wav"
    if not Path(audio_path).exists():
        print(f"Error: {audio_path} not found")
        return

    TEST_DATA_DIR.mkdir(exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MistralTokenizer.from_file(TOKENIZER_PATH)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = Audio.from_file(audio_path, strict=False)
    print(f"  Original duration: {audio.duration:.2f}s at {audio.sampling_rate}Hz")
    print(f"  Original samples: {len(audio.audio_array)}")

    # Encode with mistral-common (this pads the audio!)
    print("\nEncoding with mistral-common (applies left-padding)...")
    request = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        language="en",
    )
    tokenized = tokenizer.encode_transcription(request)

    print(f"  Prefix tokens: {tokenized.tokens}")
    print(f"  Prefix length: {len(tokenized.tokens)}")

    padded_audio = tokenized.audios[0]
    print(f"  Padded audio duration: {padded_audio.duration:.2f}s")
    print(f"  Padded audio samples: {len(padded_audio.audio_array)}")
    print(f"  Left padding: {padded_audio.duration - audio.duration:.2f}s")

    # Compute mel from PADDED audio
    print("\nComputing mel from padded audio...")
    mel = compute_mel(padded_audio.audio_array, padded_audio.sampling_rate)
    print(f"  Mel shape: {mel.shape}")

    # Save padded mel
    mel_path = TEST_DATA_DIR / "reference_mel_padded.npy"
    np.save(mel_path, mel.numpy())
    print(f"  Saved: {mel_path}")

    # Run encoder
    print("\nRunning encoder to generate audio embeddings...")
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        audio_embeds = run_encoder(mel, f)
        print(f"  Audio embeds shape: {audio_embeds.shape}")

        # Save audio embeddings
        embeds_path = TEST_DATA_DIR / "reference_audio_embeds_padded.npy"
        np.save(embeds_path, audio_embeds.squeeze(0).numpy())
        print(f"  Saved: {embeds_path}")

    # Save prefix info
    info_path = TEST_DATA_DIR / "reference_prefix_info.txt"
    with open(info_path, "w") as f:
        f.write(f"prefix_tokens={tokenized.tokens}\n")
        f.write(f"prefix_length={len(tokenized.tokens)}\n")
        f.write(f"original_duration_s={audio.duration:.4f}\n")
        f.write(f"padded_duration_s={padded_audio.duration:.4f}\n")
        f.write(f"padded_samples={len(padded_audio.audio_array)}\n")
        f.write(f"mel_frames={mel.shape[1]}\n")
        f.write(f"audio_embed_positions={audio_embeds.shape[1]}\n")
    print(f"  Saved: {info_path}")

    print("\n=== Summary ===")
    print(f"  Original audio: {audio.duration:.2f}s ({len(audio.audio_array)} samples)")
    print(f"  Padded audio: {padded_audio.duration:.2f}s ({len(padded_audio.audio_array)} samples)")
    print(f"  Mel spectrogram: {mel.shape}")
    print(f"  Audio embeddings: {audio_embeds.shape}")
    print(f"  Prefix length: {len(tokenized.tokens)} tokens")
    print(f"\nReference data saved to {TEST_DATA_DIR}/")

if __name__ == "__main__":
    main()
