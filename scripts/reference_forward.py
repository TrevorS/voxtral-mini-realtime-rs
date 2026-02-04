#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "safetensors>=0.4",
#     "numpy>=1.24",
# ]
# ///
"""
Reference forward pass for individual Voxtral components.

Run specific layers/components with known inputs and save outputs for Rust validation.

Usage:
    ./scripts/reference_forward.py rms_norm      # Test RMSNorm
    ./scripts/reference_forward.py rope          # Test RoPE embeddings
    ./scripts/reference_forward.py swiglu        # Test SwiGLU MLP
    ./scripts/reference_forward.py attention     # Test attention layer
    ./scripts/reference_forward.py conv          # Test conv downsampler
    ./scripts/reference_forward.py encoder_layer # Test full encoder layer
    ./scripts/reference_forward.py llm_layer     # Test full LLM layer
    ./scripts/reference_forward.py ada_rms_norm  # Test ADA RMSNorm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = Path("models/voxtral/consolidated.safetensors")
OUTPUT_DIR = Path("test_data")


def save_tensor(name: str, tensor: torch.Tensor, output_dir: Path = OUTPUT_DIR):
    """Save tensor as numpy for Rust to load."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npy"
    np.save(path, tensor.float().cpu().numpy())
    print(f"  Saved: {path} {list(tensor.shape)}")


def load_weight(f, name: str) -> torch.Tensor:
    """Load weight from SafeTensors file."""
    return f.get_tensor(name).float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def rope_freqs(dim: int, max_seq_len: int, theta: float = 1_000_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE frequency tensors."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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


def swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """SwiGLU MLP: w2(silu(w1(x)) * w3(x))"""
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def test_rms_norm():
    """Test RMSNorm with encoder layer 0 weights."""
    print("\n=== Testing RMSNorm ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        weight = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention_norm.weight")

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(1, 10, 1280)  # [batch, seq, dim]

    # Forward pass
    out = rms_norm(x, weight, eps=1e-5)

    # Save for Rust validation
    save_tensor("rms_norm_input", x)
    save_tensor("rms_norm_weight", weight)
    save_tensor("rms_norm_output", out)

    print(f"  Input shape: {list(x.shape)}")
    print(f"  Output shape: {list(out.shape)}")
    print(f"  Output mean: {out.mean().item():.6f}")
    print(f"  Output std: {out.std().item():.6f}")


def test_rope():
    """Test RoPE embeddings."""
    print("\n=== Testing RoPE ===")

    head_dim = 64  # Encoder head dim
    seq_len = 100
    theta = 1_000_000.0

    cos, sin = rope_freqs(head_dim, seq_len, theta)

    # Test input: [batch, seq, heads, head_dim]
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, 32, head_dim)

    out = apply_rope(x, cos, sin)

    save_tensor("rope_cos", cos)
    save_tensor("rope_sin", sin)
    save_tensor("rope_input", x)
    save_tensor("rope_output", out)

    print(f"  Head dim: {head_dim}")
    print(f"  Seq len: {seq_len}")
    print(f"  Theta: {theta}")
    print(f"  Output shape: {list(out.shape)}")


def test_swiglu():
    """Test SwiGLU MLP with encoder layer 0 weights."""
    print("\n=== Testing SwiGLU MLP ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        w1 = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.feed_forward.w1.weight")
        w2 = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.feed_forward.w2.weight")
        w3 = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.feed_forward.w3.weight")

    # Test input
    torch.manual_seed(42)
    x = torch.randn(1, 10, 1280)

    out = swiglu(x, w1, w2, w3)

    save_tensor("swiglu_input", x)
    save_tensor("swiglu_w1", w1)
    save_tensor("swiglu_w2", w2)
    save_tensor("swiglu_w3", w3)
    save_tensor("swiglu_output", out)

    print(f"  Input shape: {list(x.shape)}")
    print(f"  w1 shape: {list(w1.shape)}")
    print(f"  w2 shape: {list(w2.shape)}")
    print(f"  w3 shape: {list(w3.shape)}")
    print(f"  Output shape: {list(out.shape)}")


def test_conv():
    """Test convolutional downsampler."""
    print("\n=== Testing Conv Downsampler ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        conv1_w = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight")
        conv1_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias")
        conv2_w = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight")
        conv2_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.bias")

    print(f"  Conv1: {list(conv1_w.shape)} (out×in×kernel)")
    print(f"  Conv2: {list(conv2_w.shape)} (out×in×kernel)")

    # Test input: mel spectrogram [batch, mel_bins, time]
    torch.manual_seed(42)
    x = torch.randn(1, 128, 100)  # 128 mel bins, 100 frames

    # Apply convolutions with GELU and stride
    # Conv1: stride=2 (based on typical Whisper architecture)
    x1 = F.gelu(F.conv1d(x, conv1_w, conv1_b, stride=2, padding=1))
    print(f"  After conv1: {list(x1.shape)}")

    # Conv2: stride=2
    x2 = F.gelu(F.conv1d(x1, conv2_w, conv2_b, stride=2, padding=1))
    print(f"  After conv2: {list(x2.shape)}")

    save_tensor("conv_input", x)
    save_tensor("conv1_weight", conv1_w)
    save_tensor("conv1_bias", conv1_b)
    save_tensor("conv2_weight", conv2_w)
    save_tensor("conv2_bias", conv2_b)
    save_tensor("conv_output", x2)

    print(f"  Total downsample: {x.shape[-1]} -> {x2.shape[-1]} ({x.shape[-1] / x2.shape[-1]}x)")


def test_attention():
    """Test attention with encoder layer 0 weights."""
    print("\n=== Testing Attention (Encoder) ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        wq = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wq.weight")
        wk = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wk.weight")
        wv = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wv.weight")
        wo = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wo.weight")
        wq_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wq.bias")
        wo_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wo.bias")
        wv_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wv.bias")

    n_heads = 32
    head_dim = 64
    dim = 1280
    seq_len = 10

    print(f"  wq shape: {list(wq.shape)}")
    print(f"  wk shape: {list(wk.shape)}")
    print(f"  wv shape: {list(wv.shape)}")
    print(f"  wo shape: {list(wo.shape)}")

    # Test input
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, dim)

    # Project Q, K, V
    q = F.linear(x, wq, wq_b)
    k = F.linear(x, wk)  # No bias for K
    v = F.linear(x, wv, wv_b)

    # Reshape for multi-head attention
    q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)  # [1, 32, 10, 64]
    k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    cos, sin = rope_freqs(head_dim, seq_len)
    q_rope = apply_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
    k_rope = apply_rope(k.transpose(1, 2), cos, sin).transpose(1, 2)

    # Attention scores
    scale = head_dim ** -0.5
    scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax and value projection
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    # Output projection
    out = out.transpose(1, 2).contiguous().view(1, seq_len, -1)
    out = F.linear(out, wo, wo_b)

    save_tensor("attn_input", x)
    save_tensor("attn_wq", wq)
    save_tensor("attn_wk", wk)
    save_tensor("attn_wv", wv)
    save_tensor("attn_wo", wo)
    save_tensor("attn_output", out)

    print(f"  Input shape: {list(x.shape)}")
    print(f"  Output shape: {list(out.shape)}")


def test_ada_rms_norm():
    """Test ADA RMSNorm from LLM layer."""
    print("\n=== Testing ADA RMSNorm ===")

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        w0 = load_weight(f, "layers.0.ada_rms_norm_t_cond.0.weight")  # [32, 3072]
        w2 = load_weight(f, "layers.0.ada_rms_norm_t_cond.2.weight")  # [3072, 32]

    print(f"  w0 shape: {list(w0.shape)} (Linear: dim -> t_cond_dim)")
    print(f"  w2 shape: {list(w2.shape)} (Linear: t_cond_dim -> dim)")

    # The structure is: Linear(dim, t_cond_dim) -> SiLU -> Linear(t_cond_dim, dim)
    # This produces per-element scaling factors

    dim = 3072
    t_cond_dim = 32
    seq_len = 10

    torch.manual_seed(42)
    x = torch.randn(1, seq_len, dim)
    t_embed = torch.randn(1, 1, dim)  # Temporal embedding

    # Compute adaptive scale
    scale = F.linear(t_embed, w0)  # [1, 1, 32]
    scale = F.silu(scale)
    scale = F.linear(scale, w2)  # [1, 1, 3072]

    # Apply RMSNorm with adaptive scale
    # First compute standard RMSNorm
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + 1e-5)

    # Then apply adaptive scale (no learnable weight - just scale)
    # The adaptive modulation is: (1 + scale) * x_norm
    out = x_norm * (1 + scale)

    save_tensor("ada_rms_norm_input", x)
    save_tensor("ada_rms_norm_t_embed", t_embed)
    save_tensor("ada_rms_norm_w0", w0)
    save_tensor("ada_rms_norm_w2", w2)
    save_tensor("ada_rms_norm_scale", scale)
    save_tensor("ada_rms_norm_output", out)

    print(f"  Input shape: {list(x.shape)}")
    print(f"  T_embed shape: {list(t_embed.shape)}")
    print(f"  Scale shape: {list(scale.shape)}")
    print(f"  Output shape: {list(out.shape)}")


def test_all():
    """Run all tests."""
    test_rms_norm()
    test_rope()
    test_swiglu()
    test_conv()
    test_attention()
    test_ada_rms_norm()

    print(f"\n=== All test data saved to {OUTPUT_DIR}/ ===")
    print("Use these .npy files to validate Rust implementation.")


def main():
    parser = argparse.ArgumentParser(description="Reference forward pass for Voxtral components")
    parser.add_argument("component", nargs="?", default="all",
                        choices=["all", "rms_norm", "rope", "swiglu", "conv", "attention", "ada_rms_norm"],
                        help="Component to test")

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run: ./scripts/download_model.py")
        return

    dispatch = {
        "all": test_all,
        "rms_norm": test_rms_norm,
        "rope": test_rope,
        "swiglu": test_swiglu,
        "conv": test_conv,
        "attention": test_attention,
        "ada_rms_norm": test_ada_rms_norm,
    }

    dispatch[args.component]()


if __name__ == "__main__":
    main()
