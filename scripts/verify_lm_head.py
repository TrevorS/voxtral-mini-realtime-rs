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
Manually compute what the Rust hidden state would produce as logits.
"""

import torch
import numpy as np
from safetensors import safe_open

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def load_weight(f, name):
    return f.get_tensor(name).float()

def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

def main():
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        tok_emb = load_weight(f, "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        final_norm = load_weight(f, "norm.weight")

    # Rust hidden state values (first position, complete)
    # From output: First 5: [-20.21586, -12.572489, 36.75707, -1.9876776, -0.29208523]
    # We need the full hidden state. Let me create a test with similar statistics.

    # From Rust output:
    # hidden (before lm_head) stats: min=-28.4833, max=120.9833, mean=0.0380

    # Let's create a synthetic hidden state with similar statistics
    torch.manual_seed(42)
    hidden = torch.randn(3072) * 30  # Scale to get similar range
    hidden = hidden - hidden.mean() + 0.038  # Adjust mean

    print(f"Synthetic hidden stats: min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")

    # Apply RMSNorm
    hidden_normed = rms_norm(hidden, final_norm)
    print(f"After RMSNorm: min={hidden_normed.min():.4f}, max={hidden_normed.max():.4f}, mean={hidden_normed.mean():.4f}")

    # LM head (tied embeddings)
    logits = torch.matmul(hidden_normed, tok_emb.T)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    # Check top tokens
    top5 = logits.topk(5)
    print(f"\nTop 5 tokens: {top5.indices.tolist()}")
    print(f"Top 5 logits: {[f'{v:.4f}' for v in top5.values.tolist()]}")

    # Check token 32 specifically
    print(f"\nToken 32 logit: {logits[32].item():.4f}")
    print(f"Token 32 rank: {(logits > logits[32]).sum().item()}")

    # Now let's check what hidden state would give token 32 as top
    print("\n=== Finding hidden states that produce token 32 ===")

    # The key insight: for token 32 to win, after RMSNorm, the hidden should align with tok_emb[32]
    # Let's check what tok_emb[32] looks like
    print(f"Token 32 embedding: norm={tok_emb[32].norm():.4f}")

    # If we want token 32 to win, we need:
    # hidden_normed @ tok_emb[32] > hidden_normed @ tok_emb[k] for all k
    #
    # One way: set hidden such that after RMSNorm, it's exactly tok_emb[32]
    # RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    # We want: x * weight / sqrt(mean(x^2) + eps) = tok_emb[32]
    # So: x = tok_emb[32] * sqrt(mean(x^2) + eps) / weight

    # This is circular, but we can iterate or use the fact that if we set
    # x = tok_emb[32] / weight, then RMSNorm will just scale it uniformly

    test_hidden = tok_emb[32] / final_norm
    # Handle potential division by zero
    test_hidden = torch.where(final_norm.abs() > 1e-6, test_hidden, torch.zeros_like(test_hidden))

    print(f"\nTest hidden (tok_emb[32] / norm_weight):")
    print(f"  Stats: min={test_hidden.min():.4f}, max={test_hidden.max():.4f}, mean={test_hidden.mean():.4f}")

    # Check what this produces
    test_normed = rms_norm(test_hidden, final_norm)
    test_logits = torch.matmul(test_normed, tok_emb.T)
    top5 = test_logits.topk(5)
    print(f"  Top 5 tokens: {top5.indices.tolist()}")
    print(f"  Token 32 logit: {test_logits[32].item():.4f}")

if __name__ == "__main__":
    main()
