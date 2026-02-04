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
Compare full forward pass with Rust implementation.

This loads the reference mel and runs through the same encoder/adapter/decoder
pipeline to generate expected outputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def load_weight(f, name):
    return f.get_tensor(name).float()

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
    """Apply rotary position embeddings. x: [batch, seq, heads, head_dim]"""
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out

def time_embedding(t, dim=3072, theta=10000.0):
    """Sinusoidal time embedding."""
    half_dim = dim // 2
    inv_freq = torch.exp(
        -np.log(theta) * torch.arange(half_dim).float() / half_dim
    )
    emb = t.unsqueeze(-1) * inv_freq
    return torch.cat([emb.cos(), emb.sin()], dim=-1)

def main():
    # Load reference mel
    if not Path("test_data/reference_mel.npy").exists():
        print("Error: test_data/reference_mel.npy not found")
        print("Run: ./scripts/reference_inference.py <audio.wav>")
        return

    mel = torch.from_numpy(np.load("test_data/reference_mel.npy")).float()
    print(f"Mel shape: {mel.shape}")
    print(f"Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

    # Add batch dimension
    mel = mel.unsqueeze(0)  # [1, 128, n_frames]

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        # === CONV LAYERS ===
        conv1_w = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight")
        conv1_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias")
        conv2_w = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight")
        conv2_b = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.bias")

        print("\n=== Conv Layers ===")
        x = F.gelu(F.conv1d(mel, conv1_w, conv1_b, stride=2, padding=1))
        print(f"After conv1: {x.shape}, stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
        x = F.gelu(F.conv1d(x, conv2_w, conv2_b, stride=2, padding=1))
        print(f"After conv2: {x.shape}, stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Transpose for transformer: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        print(f"After transpose: {x.shape}")

        # Save conv output for comparison
        np.save("test_data/python_conv_output.npy", x.numpy())

        # === ENCODER LAYERS (just first layer for debugging) ===
        print("\n=== Encoder Layer 0 ===")
        prefix = "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0"

        # Load weights
        attn_norm_w = load_weight(f, f"{prefix}.attention_norm.weight")
        wq = load_weight(f, f"{prefix}.attention.wq.weight")
        wk = load_weight(f, f"{prefix}.attention.wk.weight")
        wv = load_weight(f, f"{prefix}.attention.wv.weight")
        wo = load_weight(f, f"{prefix}.attention.wo.weight")
        wq_b = load_weight(f, f"{prefix}.attention.wq.bias")
        wv_b = load_weight(f, f"{prefix}.attention.wv.bias")
        wo_b = load_weight(f, f"{prefix}.attention.wo.bias")
        ffn_norm_w = load_weight(f, f"{prefix}.ffn_norm.weight")
        w1 = load_weight(f, f"{prefix}.feed_forward.w1.weight")
        w2 = load_weight(f, f"{prefix}.feed_forward.w2.weight")
        w3 = load_weight(f, f"{prefix}.feed_forward.w3.weight")
        w2_b = load_weight(f, f"{prefix}.feed_forward.w2.bias")

        n_heads = 32
        head_dim = 64
        seq_len = x.shape[1]

        # Attention norm
        residual = x.clone()
        x_normed = rms_norm(x, attn_norm_w)
        print(f"After attn_norm: min={x_normed.min():.4f}, max={x_normed.max():.4f}, mean={x_normed.mean():.4f}")

        # QKV projections
        q = F.linear(x_normed, wq, wq_b)
        k = F.linear(x_normed, wk)
        v = F.linear(x_normed, wv, wv_b)

        # Reshape for MHA
        batch = x.shape[0]
        q = q.view(batch, seq_len, n_heads, head_dim)
        k = k.view(batch, seq_len, n_heads, head_dim)
        v = v.view(batch, seq_len, n_heads, head_dim)

        # RoPE
        cos, sin = rope_freqs(head_dim, seq_len)
        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)

        # Attention
        q_t = q_rope.transpose(1, 2)  # [B, heads, seq, head_dim]
        k_t = k_rope.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = F.linear(out, wo, wo_b)

        # Residual
        x = attn_out + residual
        print(f"After attn+residual: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # FFN
        residual = x.clone()
        x_normed = rms_norm(x, ffn_norm_w)
        mlp_out = F.linear(F.silu(F.linear(x_normed, w1)) * F.linear(x_normed, w3), w2, w2_b)
        x = mlp_out + residual
        print(f"After MLP+residual: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Save layer 0 output
        np.save("test_data/python_encoder_layer0_output.npy", x.numpy())
        print(f"\nSaved encoder layer 0 output first 5 values: {x[0, 0, :5].tolist()}")

        # Run remaining encoder layers
        print("\n=== Running all 32 encoder layers ===")
        # Reset x to conv output
        x = torch.from_numpy(np.load("test_data/python_conv_output.npy")).float()

        for layer_idx in range(32):
            prefix = f"mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{layer_idx}"

            attn_norm_w = load_weight(f, f"{prefix}.attention_norm.weight")
            wq = load_weight(f, f"{prefix}.attention.wq.weight")
            wk = load_weight(f, f"{prefix}.attention.wk.weight")
            wv = load_weight(f, f"{prefix}.attention.wv.weight")
            wo = load_weight(f, f"{prefix}.attention.wo.weight")
            wq_b = load_weight(f, f"{prefix}.attention.wq.bias")
            wv_b = load_weight(f, f"{prefix}.attention.wv.bias")
            wo_b = load_weight(f, f"{prefix}.attention.wo.bias")
            ffn_norm_w = load_weight(f, f"{prefix}.ffn_norm.weight")
            w1 = load_weight(f, f"{prefix}.feed_forward.w1.weight")
            w2 = load_weight(f, f"{prefix}.feed_forward.w2.weight")
            w3 = load_weight(f, f"{prefix}.feed_forward.w3.weight")
            w2_b = load_weight(f, f"{prefix}.feed_forward.w2.bias")

            seq_len = x.shape[1]
            batch = x.shape[0]

            # Attention block
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

            # FFN block
            residual = x.clone()
            x_normed = rms_norm(x, ffn_norm_w)
            mlp_out = F.linear(F.silu(F.linear(x_normed, w1)) * F.linear(x_normed, w3), w2, w2_b)
            x = mlp_out + residual

            if layer_idx % 8 == 7:
                print(f"  After layer {layer_idx}: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Final encoder norm
        encoder_norm_w = load_weight(f, "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight")
        x = rms_norm(x, encoder_norm_w)
        print(f"\nAfter encoder final norm: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Save full encoder output
        np.save("test_data/python_encoder_full_output.npy", x.numpy())
        print(f"Full encoder output first 5 values: {x[0, 0, :5].tolist()}")

        # Reshape (4x pooling)
        seq_len = x.shape[1]
        new_seq = seq_len // 4
        x = x[:, :new_seq*4, :].reshape(1, new_seq, 5120)
        print(f"\nAfter reshape (4x): {x.shape}")

        # Adapter
        adapter_w_in = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight")
        adapter_w_out = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight")

        x = F.gelu(F.linear(x, adapter_w_in))
        x = F.linear(x, adapter_w_out)
        print(f"After adapter: {x.shape}, stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Save audio embeds for comparison
        np.save("test_data/python_audio_embeds.npy", x.numpy())
        print(f"\n=== Python audio embeds first 10: {x[0, 0, :10].tolist()}")

        # === DECODER ===
        print("\n=== DECODER ===")

        # Get text embeddings for [STREAMING_PAD] tokens
        tok_emb = load_weight(f, "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        seq_len = x.shape[1]  # 41
        streaming_pad_emb = tok_emb[32:33].unsqueeze(0).expand(1, seq_len, -1)  # [1, 41, 3072]
        print(f"Text embeds shape: {streaming_pad_emb.shape}")
        print(f"Text embeds stats: min={streaming_pad_emb.min():.4f}, max={streaming_pad_emb.max():.4f}, mean={streaming_pad_emb.mean():.4f}")

        # Sum audio and text embeddings
        inputs_embeds = x + streaming_pad_emb
        print(f"Inputs embeds stats: min={inputs_embeds.min():.4f}, max={inputs_embeds.max():.4f}, mean={inputs_embeds.mean():.4f}")

        # Time embedding
        t_embed = time_embedding(torch.tensor([6.0]), dim=3072)  # [1, 3072]
        t_embed = t_embed.unsqueeze(1)  # [1, 1, 3072]
        print(f"t_embed first 5: {t_embed[0, 0, :5].tolist()}")

        # Run decoder layers
        x = inputs_embeds
        for layer_idx in range(26):
            prefix = f"layers.{layer_idx}"

            # Load weights
            attn_norm_w = load_weight(f, f"{prefix}.attention_norm.weight")
            ffn_norm_w = load_weight(f, f"{prefix}.ffn_norm.weight")
            ada_w0 = load_weight(f, f"{prefix}.ada_rms_norm_t_cond.0.weight")
            ada_w2 = load_weight(f, f"{prefix}.ada_rms_norm_t_cond.2.weight")
            wq = load_weight(f, f"{prefix}.attention.wq.weight")
            wk = load_weight(f, f"{prefix}.attention.wk.weight")
            wv = load_weight(f, f"{prefix}.attention.wv.weight")
            wo = load_weight(f, f"{prefix}.attention.wo.weight")
            w1 = load_weight(f, f"{prefix}.feed_forward.w1.weight")
            w2 = load_weight(f, f"{prefix}.feed_forward.w2.weight")
            w3 = load_weight(f, f"{prefix}.feed_forward.w3.weight")

            # GQA params
            n_heads_q = 32
            n_kv_heads = 8
            head_dim_dec = 128
            seq_len_dec = x.shape[1]
            batch = x.shape[0]

            # Attention block
            residual = x.clone()
            x_normed = rms_norm(x, attn_norm_w)
            q = F.linear(x_normed, wq)
            k = F.linear(x_normed, wk)
            v = F.linear(x_normed, wv)

            q = q.view(batch, seq_len_dec, n_heads_q, head_dim_dec)
            k = k.view(batch, seq_len_dec, n_kv_heads, head_dim_dec)
            v = v.view(batch, seq_len_dec, n_kv_heads, head_dim_dec)

            # RoPE
            cos, sin = rope_freqs(head_dim_dec, seq_len_dec)
            q_rope = apply_rope(q, cos, sin)
            k_rope = apply_rope(k, cos, sin)

            # GQA: repeat KV heads
            n_rep = n_heads_q // n_kv_heads
            k_rope = k_rope.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len_dec, n_heads_q, head_dim_dec)
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len_dec, n_heads_q, head_dim_dec)

            q_t = q_rope.transpose(1, 2)
            k_t = k_rope.transpose(1, 2)
            v_t = v.transpose(1, 2)
            scale = head_dim_dec ** -0.5
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

            mask = torch.triu(torch.ones(seq_len_dec, seq_len_dec), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_t)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len_dec, -1)
            attn_out = F.linear(out, wo)
            x = attn_out + residual

            # FFN block with ADA modulation
            residual = x.clone()
            x_normed = rms_norm(x, ffn_norm_w)

            # ADA modulation: x * (1 + ada_rms_norm_t_cond(t_embed))
            scale = F.gelu(F.linear(t_embed, ada_w0))  # [1, 1, 32]
            scale = F.linear(scale, ada_w2)  # [1, 1, 3072]
            x_ada = x_normed * (1 + scale)

            mlp_out = F.linear(F.silu(F.linear(x_ada, w1)) * F.linear(x_ada, w3), w2)
            x = mlp_out + residual

            if layer_idx % 5 == 4:
                print(f"  After decoder layer {layer_idx}: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        # Final LLM norm
        final_norm_w = load_weight(f, "norm.weight")
        hidden = rms_norm(x, final_norm_w)
        print(f"\nAfter final norm: min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")
        print(f"Hidden first 5: {hidden[0, 0, :5].tolist()}")

        # LM head (tied embeddings)
        logits = torch.matmul(hidden, tok_emb.T)
        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

        # Top tokens
        top5 = logits[0, 0].topk(5)
        print(f"\nPosition 0 top 5 tokens: {top5.indices.tolist()}")
        print(f"Position 0 top 5 logits: {[f'{v:.4f}' for v in top5.values.tolist()]}")
        print(f"Token 32 logit: {logits[0, 0, 32].item():.4f}")

        # Check position 10 too
        top5_10 = logits[0, 10].topk(5)
        print(f"\nPosition 10 top 5 tokens: {top5_10.indices.tolist()}")
        print(f"Position 10 top 5 logits: {[f'{v:.4f}' for v in top5_10.values.tolist()]}")

        # Argmax for all positions
        predicted = logits.argmax(dim=-1)[0].tolist()
        print(f"\nPredicted tokens (all {len(predicted)}): {predicted}")

        # Check positions around the prefix boundary (38 tokens)
        print("\n=== Positions around prefix boundary ===")
        for pos in [35, 36, 37, 38, 39, 40]:
            if pos < len(predicted):
                top5 = logits[0, pos].topk(5)
                print(f"Position {pos}: predicted={predicted[pos]}, top5 tokens={top5.indices.tolist()}, logits={[f'{v:.2f}' for v in top5.values.tolist()]}")

        # Check what top non-control tokens look like
        print("\n=== Position 38 top 20 tokens (excluding control) ===")
        pos38_logits = logits[0, 38]
        sorted_indices = pos38_logits.argsort(descending=True)
        count = 0
        for idx in sorted_indices[:100].tolist():
            if idx >= 1000:  # Skip control tokens (0-999 roughly)
                print(f"  Token {idx}: logit={pos38_logits[idx].item():.4f}")
                count += 1
                if count >= 10:
                    break

if __name__ == "__main__":
    main()
