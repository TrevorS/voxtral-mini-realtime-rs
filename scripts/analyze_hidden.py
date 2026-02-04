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
Analyze what hidden states would need to look like to produce token 32 vs other tokens.
"""

import torch
import numpy as np
from safetensors import safe_open

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def load_weight(f, name):
    return f.get_tensor(name).float()

def main():
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        tok_emb = load_weight(f, "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        final_norm = load_weight(f, "norm.weight")

    print(f"Token embeddings shape: {tok_emb.shape}")
    print(f"Final norm shape: {final_norm.shape}")

    # The Rust hidden states (first position):
    # First 5: [-20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523]
    rust_hidden_sample = torch.tensor([
        -20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523
    ])

    print("\n=== Analysis ===")
    print(f"Rust hidden first 5: {rust_hidden_sample.tolist()}")

    # For reference, what does the [STREAMING_PAD] embedding look like?
    print(f"\n[STREAMING_PAD] (32) embedding first 5: {tok_emb[32, :5].tolist()}")
    print(f"[STREAMING_PAD] embedding norm: {tok_emb[32].norm().item():.4f}")

    # What embedding would the Rust hidden project to after norm + lm_head?
    # We need a full hidden state to compute this, but let's check if there's a pattern

    # Let's see what hidden state would be needed to get token 32 as argmax
    # LM head: logits = hidden @ tok_emb.T
    # For token 32 to win, hidden @ tok_emb[32] > hidden @ tok_emb[k] for all k

    # Let's compute what the logits would be for different hidden states

    # Test 1: What if hidden = token 32 embedding (scaled up)?
    print("\n=== Test: hidden = scaled [STREAMING_PAD] embedding ===")
    for scale in [1.0, 10.0, 50.0, 100.0]:
        test_hidden = tok_emb[32] * scale
        # Apply final norm
        variance = test_hidden.pow(2).mean(-1, keepdim=True)
        test_normed = test_hidden * torch.rsqrt(variance + 1e-5) * final_norm
        # LM head
        logits = torch.matmul(test_normed, tok_emb.T)
        top5 = logits.topk(5)
        print(f"  scale={scale}: top tokens = {top5.indices.tolist()}, logits = {[f'{v:.2f}' for v in top5.values.tolist()]}")

    # Test 2: What about a random hidden state?
    print("\n=== Test: random hidden state ===")
    torch.manual_seed(42)
    test_hidden = torch.randn(3072) * 10  # Similar scale to our Rust output
    variance = test_hidden.pow(2).mean(-1, keepdim=True)
    test_normed = test_hidden * torch.rsqrt(variance + 1e-5) * final_norm
    logits = torch.matmul(test_normed, tok_emb.T)
    top5 = logits.topk(5)
    print(f"  top tokens = {top5.indices.tolist()}")
    print(f"  token 32 rank: {(logits > logits[32]).sum().item()}")
    print(f"  token 32 logit: {logits[32].item():.2f}")

    # Test 3: The key insight - what direction in hidden space leads to token 32?
    # The answer is: the direction of tok_emb[32] * final_norm
    print("\n=== Token 32 'direction' ===")
    token32_direction = tok_emb[32] * final_norm  # This is what gets matched against
    print(f"  Direction norm: {token32_direction.norm().item():.4f}")
    print(f"  Direction first 5: {token32_direction[:5].tolist()}")

    # Compare with Rust hidden first 5
    print(f"\n  Rust hidden first 5: [-20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523]")
    print(f"  Token 32 direction first 5: {token32_direction[:5].tolist()}")

    # These should be aligned if the model predicts token 32!
    # Let's check the cosine similarity
    rust_hidden_full = torch.tensor([
        -20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523
    ] + [0.0] * (3072 - 5))  # Pad with zeros for now

    # We can't compute full cosine sim without full hidden state, but let's check first 5
    print("\n=== Partial alignment check ===")
    rust_h5 = torch.tensor([-20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523])
    t32_d5 = token32_direction[:5]
    dot = (rust_h5 * t32_d5).sum()
    print(f"  Dot product (first 5 dims): {dot.item():.4f}")
    print(f"  Signs match: {[(r > 0) == (t > 0) for r, t in zip(rust_h5.tolist(), t32_d5.tolist())]}")

if __name__ == "__main__":
    main()
