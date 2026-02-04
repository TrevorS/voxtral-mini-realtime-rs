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
Check token embeddings and decoder weights match expected values.
"""

import torch
import numpy as np
from safetensors import safe_open

MODEL_PATH = "models/voxtral/consolidated.safetensors"

def load_weight(f, name):
    return f.get_tensor(name).float()

def main():
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        # Token embeddings
        tok_emb = load_weight(f, "mm_streams_embeddings.embedding_module.tok_embeddings.weight")
        print(f"Token embeddings shape: {tok_emb.shape}")
        print(f"  Token 32 ([STREAMING_PAD]) first 10: {tok_emb[32, :10].tolist()}")
        print(f"  Token 33 (!) first 10: {tok_emb[33, :10].tolist()}")
        print(f"  Token 1 (<unk>?) first 10: {tok_emb[1, :10].tolist()}")
        print(f"  Token 2 (<s>?) first 10: {tok_emb[2, :10].tolist()}")

        # Check decoder layer 0 weights
        prefix = "layers.0"
        attn_norm = load_weight(f, f"{prefix}.attention_norm.weight")
        ffn_norm = load_weight(f, f"{prefix}.ffn_norm.weight")
        print(f"\nDecoder layer 0 attention_norm first 10: {attn_norm[:10].tolist()}")
        print(f"Decoder layer 0 ffn_norm first 10: {ffn_norm[:10].tolist()}")

        # Check adapter
        adapter_in = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight")
        adapter_out = load_weight(f, "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight")
        print(f"\nAdapter in weight shape: {adapter_in.shape}")
        print(f"Adapter in first row first 10: {adapter_in[0, :10].tolist()}")
        print(f"Adapter out weight shape: {adapter_out.shape}")
        print(f"Adapter out first row first 10: {adapter_out[0, :10].tolist()}")

        # Check final LLM norm
        final_norm = load_weight(f, "norm.weight")
        print(f"\nFinal LLM norm first 10: {final_norm[:10].tolist()}")

        # Let's also check what the LM head would output for a test embedding
        # The LM head uses tied embeddings (tok_emb.T)
        print("\n=== Simple test ===")
        # Create a test hidden state that's similar to token 32 embedding
        test_hidden = tok_emb[32].unsqueeze(0)  # [1, 3072]
        # Apply final norm
        variance = test_hidden.pow(2).mean(-1, keepdim=True)
        test_normed = test_hidden * torch.rsqrt(variance + 1e-5) * final_norm

        # LM head: matmul with tok_emb.T to get logits
        logits = torch.matmul(test_normed, tok_emb.T)  # [1, vocab_size]
        print(f"Logits shape: {logits.shape}")
        print(f"Token 32 logit: {logits[0, 32].item():.4f}")
        print(f"Token 33 logit: {logits[0, 33].item():.4f}")
        print(f"Argmax token: {logits.argmax().item()}")

        # What if hidden = zeros + small noise?
        print("\n=== Test with near-zero hidden state ===")
        test_hidden2 = torch.randn(1, 3072) * 0.001
        variance = test_hidden2.pow(2).mean(-1, keepdim=True)
        test_normed2 = test_hidden2 * torch.rsqrt(variance + 1e-5) * final_norm
        logits2 = torch.matmul(test_normed2, tok_emb.T)
        print(f"Argmax token: {logits2.argmax().item()}")
        top5 = logits2[0].topk(5)
        print(f"Top 5 tokens: {top5.indices.tolist()}")
        print(f"Top 5 logits: {top5.values.tolist()}")

if __name__ == "__main__":
    main()
