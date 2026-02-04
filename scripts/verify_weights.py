#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "numpy",
# ]
# ///
"""
Verify weight values to compare with Rust loading.

Usage:
    ./scripts/verify_weights.py
"""

import torch
from safetensors import safe_open
import numpy as np


def main():
    print("Loading weights from SafeTensors...")

    with safe_open("models/voxtral/consolidated.safetensors", framework="pt") as f:
        # Check decoder layer 0 ADA RMSNorm weights
        ada_down = f.get_tensor("layers.0.ada_rms_norm_t_cond.0.weight").float()
        ada_up = f.get_tensor("layers.0.ada_rms_norm_t_cond.2.weight").float()

        print("\nDecoder Layer 0 ADA RMSNorm:")
        print(f"  ada_down shape: {ada_down.shape}")  # Should be [32, 3072]
        print(f"  ada_up shape: {ada_up.shape}")  # Should be [3072, 32]
        print(f"  ada_down first 5: {ada_down.flatten()[:5].tolist()}")
        print(f"  ada_up first 5: {ada_up.flatten()[:5].tolist()}")

        # Check token embeddings
        tok_emb = f.get_tensor(
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
        ).float()
        print(f"\nToken Embeddings:")
        print(f"  Shape: {tok_emb.shape}")  # Should be [131072, 3072]
        print(f"  Token 32 (space) embedding first 10: {tok_emb[32, :10].tolist()}")

        # Check conv layer 1
        conv1_w = f.get_tensor(
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight"
        ).float()
        print(f"\nConv1 weight:")
        print(f"  Shape: {conv1_w.shape}")  # [1280, 128, 3]
        print(f"  First 5 values: {conv1_w.flatten()[:5].tolist()}")

        # Test a simple computation: what does the model output for zeros input after LM head?
        # The LM head is just tok_embeddings.T @ hidden
        # If hidden is zeros, output should be zeros (but after softmax, uniform distribution)

        # Test what token 32's embedding looks like after going through layer 0 norm
        attention_norm = f.get_tensor("layers.0.attention_norm.weight").float()
        print(f"\nLayer 0 attention_norm:")
        print(f"  Shape: {attention_norm.shape}")  # [3072]
        print(f"  First 5: {attention_norm[:5].tolist()}")
        print(f"  Mean: {attention_norm.mean().item():.6f}")

        # What's the bias/offset that could make token 32 most likely?
        # Check if there's any bias that favors token 32
        final_norm = f.get_tensor("norm.weight").float()
        print(f"\nFinal norm:")
        print(f"  Shape: {final_norm.shape}")
        print(f"  First 5: {final_norm[:5].tolist()}")

        # Compute what the LM head output would be for a zeros hidden state
        # LM head is: hidden @ tok_emb.T (Burn uses hidden @ weight.T for linear)
        # For zeros hidden, this gives zeros, but that's before any bias

        # Check if token embeddings have any structure that favors certain tokens
        tok_norms = tok_emb.norm(dim=1)
        print(f"\nToken embedding norms:")
        print(f"  Token 32 norm: {tok_norms[32].item():.4f}")
        print(f"  Min norm: {tok_norms.min().item():.4f} (token {tok_norms.argmin().item()})")
        print(f"  Max norm: {tok_norms.max().item():.4f} (token {tok_norms.argmax().item()})")
        print(f"  Mean norm: {tok_norms.mean().item():.4f}")


if __name__ == "__main__":
    main()
