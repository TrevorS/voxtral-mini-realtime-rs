#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "safetensors",
#     "soundfile",
#     "numpy",
# ]
# ///
"""
Test inference using vLLM's forward pass components.

This tests individual components to compare with Rust implementation.
"""

import sys
sys.path.insert(0, "/home/trevor/Projects/vllm")
sys.path.insert(0, "/home/trevor/Projects/mistral-common/src")

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from safetensors import safe_open

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def load_weight(f, name):
    """Load weight from SafeTensors."""
    return f.get_tensor(name).float()

def compute_mel(audio_path):
    """Compute mel spectrogram using torchaudio."""
    import torchaudio.transforms as T

    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_tensor = resampler(audio_tensor).squeeze(0)
        audio = audio_tensor.numpy()

    audio_tensor = torch.from_numpy(audio).float()
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=128,
        f_min=0.0,
        f_max=8000.0,
        mel_scale='slaney',
        norm='slaney',
    )
    mel = mel_transform(audio_tensor)

    # Normalize (Voxtral style)
    global_log_mel_max = 1.5
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = torch.maximum(mel, torch.tensor(global_log_mel_max - 8.0))
    mel = (mel + 4.0) / 4.0

    return mel

def time_embedding(t, dim=3072, theta=10000.0):
    """Sinusoidal time embedding."""
    half_dim = dim // 2
    inv_freq = torch.exp(
        -np.log(theta) * torch.arange(half_dim).float() / half_dim
    )
    emb = t * inv_freq
    return torch.cat([emb.cos(), emb.sin()], dim=-1)

def main():
    if len(sys.argv) < 2:
        print("Usage: test_vllm_inference.py <audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("Computing mel spectrogram...")
    mel = compute_mel(audio_path)
    print(f"  Mel shape: {mel.shape}")
    print(f"  Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

    # We'll print intermediate outputs to compare with Rust
    print("\n=== Debug outputs for Rust comparison ===")
    print(f"Mel first 10 values: {mel.flatten()[:10].tolist()}")

    # Save mel for Rust to load
    np.save("test_data/reference_mel.npy", mel.numpy())

    # Compute time embedding
    t = torch.tensor([6.0])
    t_embed = time_embedding(t, dim=3072)
    print(f"\nTime embedding (t=6):")
    print(f"  Shape: {t_embed.shape}")
    print(f"  First 5: {t_embed[:5].tolist()}")
    print(f"  Stats: min={t_embed.min():.4f}, max={t_embed.max():.4f}")

    # Compare with Rust t_embed
    # Rust output was: [0.96017027, 0.94953215, 0.93774676, 0.9248504, 0.91087997]
    print(f"\n  (Rust first 5: [0.96017027, 0.94953215, 0.93774676, 0.9248504, 0.91087997])")

    print("\n=== Checking encoder components ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        # Check if encoder final norm exists
        norm_weight = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight")
        print(f"Encoder final norm weight shape: {norm_weight.shape}")
        print(f"  First 5: {norm_weight[:5].tolist()}")

        # Check adapter weights
        adapter_w_in = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight")
        adapter_w_out = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight")
        print(f"\nAdapter w_in shape: {adapter_w_in.shape}")
        print(f"Adapter w_out shape: {adapter_w_out.shape}")

        # Check tok_embeddings for streaming pad token
        tok_emb = load_weight(f, "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        print(f"\nToken embeddings shape: {tok_emb.shape}")

        # Get embedding for token 32 ([STREAMING_PAD])
        streaming_pad_emb = tok_emb[32]
        print(f"[STREAMING_PAD] (token 32) embedding:")
        print(f"  First 10: {streaming_pad_emb[:10].tolist()}")
        print(f"  Stats: min={streaming_pad_emb.min():.4f}, max={streaming_pad_emb.max():.4f}, mean={streaming_pad_emb.mean():.4f}")

        # Check LLM norm
        llm_norm = load_weight(f, "norm.weight")
        print(f"\nLLM final norm shape: {llm_norm.shape}")

if __name__ == "__main__":
    main()
