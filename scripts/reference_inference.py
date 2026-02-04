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
Reference mel spectrogram computation to compare with Rust.

Usage:
    ./scripts/reference_inference.py <audio.wav>
"""

import sys
import torch
import soundfile as sf
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: reference_inference.py <audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("Computing reference mel spectrogram...")

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(audio) / sr:.2f}s")

    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_tensor = resampler(audio_tensor).squeeze(0)
        audio = audio_tensor.numpy()
        sr = 16000
        print(f"  Resampled to 16kHz, new length: {len(audio)}")

    # Compute mel spectrogram (Whisper-style)
    # Parameters from params.json
    n_mels = 128
    hop_length = 160
    n_fft = 400

    # Simple mel spectrogram using torch
    # IMPORTANT: Must match vLLM/mistral-common parameters!
    import torchaudio.transforms as T

    audio_tensor = torch.from_numpy(audio).float()
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,  # Voxtral uses 8000 Hz max
        mel_scale='slaney',  # Must match mistral-common
        norm='slaney',  # Must match mistral-common
    )
    mel = mel_transform(audio_tensor)

    print(f"  Raw mel shape: {mel.shape}")
    print(f"  Raw mel stats: min={mel.min():.6f}, max={mel.max():.6f}, mean={mel.mean():.6f}")
    print(f"  Raw mel [0, 100:105]: {mel[0, 100:105].tolist()}")

    # Log scale (Voxtral-style with global_log_mel_max=1.5)
    global_log_mel_max = 1.5  # From params.json
    mel = torch.clamp(mel, min=1e-10).log10()
    print(f"  After log10: min={mel.min():.4f}, max={mel.max():.4f}")
    # Use global max instead of per-audio max
    mel = torch.maximum(mel, torch.tensor(global_log_mel_max) - 8.0)
    mel = (mel + 4.0) / 4.0

    print(f"  Normalized mel shape: {mel.shape}")
    print(f"  Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

    print("\n=== Reference mel computed ===")
    print(f"First 10 mel values (flattened): {mel.flatten()[:10].tolist()}")

    # Print mel values at position [0, 0:10] (first mel bin, first 10 frames)
    print(f"\nMel bin 0, frames 0-9: {mel[0, :10].tolist()}")

    # Get mel filterbank info
    fb = mel_transform.mel_scale.fb
    print(f"\nMel filterbank shape: {fb.shape}")
    print(f"Filterbank row 64 sum: {fb[64].sum().item():.6f}")
    print(f"Filterbank row 64 first 10: {fb[64, :10].tolist()}")

    # Save mel for Rust to load
    import os
    os.makedirs("test_data", exist_ok=True)
    np.save("test_data/reference_mel.npy", mel.numpy())
    print(f"\nSaved reference mel to test_data/reference_mel.npy")


if __name__ == "__main__":
    main()
