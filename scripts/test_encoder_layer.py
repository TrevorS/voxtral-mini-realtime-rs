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
Test a single encoder layer to compare with Rust implementation.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight

def rope_freqs(dim, max_seq_len, theta=1_000_000.0):
    """Compute RoPE frequency tensors."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    """Apply rotary position embeddings."""
    # x: [batch, seq, heads, head_dim]
    x_r = x[..., ::2]  # Real parts
    x_i = x[..., 1::2]  # Imaginary parts

    # Reshape cos/sin for broadcasting
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim/2]
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2)

    # Apply rotation
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos

    # Interleave back
    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out

def load_weight(f, name):
    return f.get_tensor(name).float()

def main():
    print("Testing encoder layer 0 forward pass...")

    # Create test input
    torch.manual_seed(42)
    batch, seq_len, d_model = 1, 10, 1280
    x = torch.randn(batch, seq_len, d_model)

    n_heads = 32
    head_dim = 64

    # Load weights
    prefix = "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0"

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        # Norms
        attn_norm_w = load_weight(f, f"{prefix}.attention_norm.weight")
        ffn_norm_w = load_weight(f, f"{prefix}.ffn_norm.weight")

        # Attention
        wq = load_weight(f, f"{prefix}.attention.wq.weight")
        wk = load_weight(f, f"{prefix}.attention.wk.weight")
        wv = load_weight(f, f"{prefix}.attention.wv.weight")
        wo = load_weight(f, f"{prefix}.attention.wo.weight")
        wq_b = load_weight(f, f"{prefix}.attention.wq.bias")
        wv_b = load_weight(f, f"{prefix}.attention.wv.bias")
        wo_b = load_weight(f, f"{prefix}.attention.wo.bias")

        # MLP
        w1 = load_weight(f, f"{prefix}.feed_forward.w1.weight")
        w2 = load_weight(f, f"{prefix}.feed_forward.w2.weight")
        w3 = load_weight(f, f"{prefix}.feed_forward.w3.weight")
        w2_b = load_weight(f, f"{prefix}.feed_forward.w2.bias")

    print(f"Input shape: {x.shape}")
    print(f"Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

    # === Attention Block ===
    residual = x.clone()

    # Attention norm
    x_normed = rms_norm(x, attn_norm_w)
    print(f"\nAfter attention_norm: min={x_normed.min():.4f}, max={x_normed.max():.4f}, mean={x_normed.mean():.4f}")

    # QKV projections
    q = F.linear(x_normed, wq, wq_b)  # [batch, seq, d_model]
    k = F.linear(x_normed, wk)  # No bias for K
    v = F.linear(x_normed, wv, wv_b)

    # Reshape for multi-head attention
    q = q.view(batch, seq_len, n_heads, head_dim)  # [1, 10, 32, 64]
    k = k.view(batch, seq_len, n_heads, head_dim)
    v = v.view(batch, seq_len, n_heads, head_dim)

    # Apply RoPE
    cos, sin = rope_freqs(head_dim, seq_len)
    q_rope = apply_rope(q, cos, sin)
    k_rope = apply_rope(k, cos, sin)

    # Transpose for attention: [batch, heads, seq, head_dim]
    q_t = q_rope.transpose(1, 2)
    k_t = k_rope.transpose(1, 2)
    v_t = v.transpose(1, 2)

    # Attention scores
    scale = head_dim ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax and value projection
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t)

    # Output projection
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    attn_out = F.linear(out, wo, wo_b)

    # Residual
    x = attn_out + residual
    print(f"After attention+residual: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

    # === MLP Block ===
    residual = x.clone()

    # FFN norm
    x_normed = rms_norm(x, ffn_norm_w)
    print(f"After ffn_norm: min={x_normed.min():.4f}, max={x_normed.max():.4f}, mean={x_normed.mean():.4f}")

    # SwiGLU: w2(silu(w1(x)) * w3(x))
    mlp_out = F.linear(F.silu(F.linear(x_normed, w1)) * F.linear(x_normed, w3), w2, w2_b)

    # Residual
    x = mlp_out + residual
    print(f"After MLP+residual: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

    # Save outputs for Rust comparison
    np.save("test_data/encoder_layer_input.npy", torch.randn(batch, seq_len, d_model).numpy())
    np.save("test_data/encoder_layer_output.npy", x.numpy())

    print("\n=== Summary ===")
    print(f"Input: random tensor [1, 10, 1280]")
    print(f"Output: [1, 10, 1280]")
    print(f"Output first 5 values: {x[0, 0, :5].tolist()}")

if __name__ == "__main__":
    main()
