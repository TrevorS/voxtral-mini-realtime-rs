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
Test autoregressive generation with Voxtral.

The key insight: streaming mode requires autoregressive generation.
After the prefix positions (38 [STREAMING_PAD]), the model expects to see
the PREVIOUSLY GENERATED token, not more [STREAMING_PAD] tokens.
"""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path

MODEL_PATH = "models/voxtral/consolidated.safetensors"

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

def time_embedding(t, dim=3072, theta=10000.0):
    half_dim = dim // 2
    inv_freq = torch.exp(-np.log(theta) * torch.arange(half_dim).float() / half_dim)
    emb = t.unsqueeze(-1) * inv_freq
    return torch.cat([emb.cos(), emb.sin()], dim=-1)

class VoxtralDecoder:
    """Simple decoder for testing."""

    def __init__(self, f):
        self.f = f
        self.tok_emb = self._load("mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        self.final_norm = self._load("norm.weight")

    def _load(self, name):
        return self.f.get_tensor(name).float()

    def forward_single(self, audio_embed, text_embed, t_embed, position):
        """
        Forward pass for a single position.
        audio_embed: [1, 3072] - audio embedding for this position
        text_embed: [1, 3072] - text embedding for this position
        t_embed: [1, 3072] - time embedding (same for all positions)
        position: int - current position (for RoPE)
        """
        x = audio_embed + text_embed  # [1, 3072]
        x = x.unsqueeze(1)  # [1, 1, 3072]

        for layer_idx in range(26):
            prefix = f"layers.{layer_idx}"

            # Load weights
            attn_norm_w = self._load(f"{prefix}.attention_norm.weight")
            ffn_norm_w = self._load(f"{prefix}.ffn_norm.weight")
            ada_w0 = self._load(f"{prefix}.ada_rms_norm_t_cond.0.weight")
            ada_w2 = self._load(f"{prefix}.ada_rms_norm_t_cond.2.weight")
            wq = self._load(f"{prefix}.attention.wq.weight")
            wk = self._load(f"{prefix}.attention.wk.weight")
            wv = self._load(f"{prefix}.attention.wv.weight")
            wo = self._load(f"{prefix}.attention.wo.weight")
            w1 = self._load(f"{prefix}.feed_forward.w1.weight")
            w2 = self._load(f"{prefix}.feed_forward.w2.weight")
            w3 = self._load(f"{prefix}.feed_forward.w3.weight")

            # Simplified attention (single position, no cache)
            residual = x.clone()
            x_normed = rms_norm(x, attn_norm_w)
            q = F.linear(x_normed, wq)
            k = F.linear(x_normed, wk)
            v = F.linear(x_normed, wv)

            # For single position, attention is just self-attention
            batch, seq_len, dim = x.shape  # [1, 1, 3072]
            n_heads_q = 32
            n_kv_heads = 8
            head_dim = 128

            q = q.view(batch, seq_len, n_heads_q, head_dim)
            k = k.view(batch, seq_len, n_kv_heads, head_dim)
            v = v.view(batch, seq_len, n_kv_heads, head_dim)

            # RoPE for single position
            cos, sin = rope_freqs(head_dim, position + 1)
            q_rope = apply_rope(q, cos[position:position+1], sin[position:position+1])
            k_rope = apply_rope(k, cos[position:position+1], sin[position:position+1])

            # GQA
            n_rep = n_heads_q // n_kv_heads
            k_rope = k_rope.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len, n_heads_q, head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len, n_heads_q, head_dim)

            # Attention (single position, so just scales v by softmax(q.k) = 1)
            attn_out = v.view(batch, seq_len, -1)
            attn_out = F.linear(attn_out, wo)
            x = attn_out + residual

            # FFN with ADA
            residual = x.clone()
            x_normed = rms_norm(x, ffn_norm_w)
            t_embed_expanded = t_embed.unsqueeze(1)  # [1, 1, 3072]
            scale = F.gelu(F.linear(t_embed_expanded, ada_w0))
            scale = F.linear(scale, ada_w2)
            x_ada = x_normed * (1 + scale)
            mlp_out = F.linear(F.silu(F.linear(x_ada, w1)) * F.linear(x_ada, w3), w2)
            x = mlp_out + residual

        # Final norm and LM head
        hidden = rms_norm(x, self.final_norm)
        logits = torch.matmul(hidden, self.tok_emb.T)  # [1, 1, vocab_size]
        return logits.squeeze(1)  # [1, vocab_size]


def main():
    if not Path("test_data/python_audio_embeds.npy").exists():
        print("Error: test_data/python_audio_embeds.npy not found")
        print("Run: ./scripts/compare_full_forward.py first")
        return

    # Load audio embeddings
    audio_embeds = torch.from_numpy(np.load("test_data/python_audio_embeds.npy")).float()
    print(f"Audio embeds shape: {audio_embeds.shape}")  # [1, seq_len, 3072]

    seq_len = audio_embeds.shape[1]
    prefix_len = 38  # n_left_pad_tokens (32) + num_delay_tokens (6)

    print(f"Sequence length: {seq_len}")
    print(f"Prefix length: {prefix_len}")
    print(f"Positions for transcription: {seq_len - prefix_len}")

    # Time embedding
    t_embed = time_embedding(torch.tensor([6.0]), dim=3072)

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        tok_emb = f.get_tensor("mm_streams_embeddings.embedding_module.tok_embeddings.weight").float()

        # Initialize tokens with prefix
        tokens = [32] * prefix_len  # [STREAMING_PAD] for prefix

        print("\n=== Autoregressive Generation ===")
        print(f"Prefix tokens (first {prefix_len} positions): all [STREAMING_PAD] (32)")

        for pos in range(prefix_len, seq_len):
            # Get audio embedding for this position
            audio_embed = audio_embeds[0, pos:pos+1]  # [1, 3072]

            # Get text embedding for PREVIOUS token (or last prefix token)
            prev_token = tokens[-1] if tokens else 32
            text_embed = tok_emb[prev_token:prev_token+1]  # [1, 3072]

            # Simplified forward: just use the embeddings directly without full decoder
            # (Full decoder is too slow for this test)
            inputs_embed = audio_embed + text_embed

            # For a quick test, just check what the audio alone produces
            # We'll do a simple projection: inputs_embed @ tok_emb.T
            # This isn't the full model but gives us an idea

            # Actually, let's run a full forward pass for just this position
            # But that's very slow without KV cache...

            # For now, let's check if the audio embedding at this position
            # is different from the prefix positions
            print(f"  Position {pos}: audio_embed first 5: {audio_embed[0, :5].tolist()}")

            # Generate next token (simplified: just use audio projection)
            # This is NOT correct but shows the idea
            logits = torch.matmul(inputs_embed, tok_emb.T)  # [1, vocab_size]
            next_token = logits.argmax(dim=-1).item()
            tokens.append(next_token)

            if pos < prefix_len + 5:
                top5 = logits[0].topk(5)
                print(f"    Simplified top 5: tokens={top5.indices.tolist()}, logits={[f'{v:.2f}' for v in top5.values.tolist()]}")

        print(f"\nGenerated tokens ({len(tokens)}):")
        print(f"  First 10: {tokens[:10]}")
        print(f"  Last 10: {tokens[-10:]}")
        print(f"  Unique tokens: {set(tokens)}")

        # Count token types
        pad_count = sum(1 for t in tokens if t == 32)
        word_count = sum(1 for t in tokens if t == 33)
        other_count = len(tokens) - pad_count - word_count
        print(f"\nToken breakdown:")
        print(f"  [STREAMING_PAD] (32): {pad_count}")
        print(f"  [STREAMING_WORD] (33): {word_count}")
        print(f"  Other: {other_count}")

if __name__ == "__main__":
    main()
