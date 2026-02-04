#!/usr/bin/env python3
"""Compare Rust mel filterbank with mistral-common's implementation."""

import sys
sys.path.insert(0, "../mistral-common/src")

import numpy as np
from mistral_common.audio import mel_filter_bank

# Voxtral parameters
num_frequency_bins = 201  # 400/2 + 1 (n_fft=400)
num_mel_bins = 128
min_frequency = 0.0
max_frequency = 8000.0
sampling_rate = 16000

# Get mistral-common filterbank
mc_filterbank = mel_filter_bank(
    num_frequency_bins=num_frequency_bins,
    num_mel_bins=num_mel_bins,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    sampling_rate=sampling_rate,
)

print(f"Mistral-common filterbank shape: {mc_filterbank.shape}")
print(f"  Min: {mc_filterbank.min():.6f}")
print(f"  Max: {mc_filterbank.max():.6f}")
print(f"  Sum (row 0): {mc_filterbank[:, 0].sum():.6f}")
print(f"  Sum (row 64): {mc_filterbank[:, 64].sum():.6f}")

# Save for Rust comparison
np.save("test_data/mel_filterbank_reference.npy", mc_filterbank)
print(f"\nSaved reference filterbank to test_data/mel_filterbank_reference.npy")

# Print first few values of row 64 for comparison
print(f"\nRow 64 first 10 values: {mc_filterbank[:10, 64]}")

# Now let's also check what happens with a simple test input
# Generate a 1-second test signal
t = np.arange(16000) / 16000
signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz sine wave

# Compute mel spectrogram manually using mistral-common filterbank
import torch
import torchaudio.transforms as T

# Use torchaudio to compute STFT
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=128,
    f_min=0.0,
    f_max=8000.0,
    mel_scale="slaney",
    norm="slaney",
)

signal_tensor = torch.from_numpy(signal)
mel_spec = mel_transform(signal_tensor)
print(f"\nTorchaudio mel spectrogram shape: {mel_spec.shape}")
print(f"  Min: {mel_spec.min():.6f}")
print(f"  Max: {mel_spec.max():.6f}")

# Apply log normalization like vLLM
log_mel = torch.clamp(mel_spec, min=1e-10).log10()
global_log_mel_max = 1.5  # From params.json
log_mel = torch.maximum(log_mel, torch.tensor(global_log_mel_max - 8.0))
log_mel = (log_mel + 4.0) / 4.0

print(f"\nNormalized log mel (Voxtral style):")
print(f"  Min: {log_mel.min():.6f}")
print(f"  Max: {log_mel.max():.6f}")
print(f"  Mean: {log_mel.mean():.6f}")

# Save for comparison
np.save("test_data/torchaudio_mel_440hz.npy", log_mel.numpy())
print(f"\nSaved torchaudio mel to test_data/torchaudio_mel_440hz.npy")
