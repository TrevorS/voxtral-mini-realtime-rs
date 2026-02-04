#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mistral-common[soxr]>=1.9.0",
#     "torch",
#     "safetensors",
#     "soundfile",
#     "numpy",
# ]
# ///
"""
Test inference using mistral-common's proper audio preprocessing.

This script uses mistral-common to:
1. Pad the audio correctly (left-pad with silence)
2. Generate the proper prefix tokens
3. Then runs our forward pass
"""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.transcription.request import TranscriptionRequest, StreamingMode
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.audio import Audio

MODEL_PATH = "models/voxtral/consolidated.safetensors"
TOKENIZER_PATH = "models/voxtral/tekken.json"

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

def compute_mel(audio_array, sampling_rate=16000):
    """Compute mel spectrogram using vLLM's approach."""
    from mistral_common.audio import mel_filter_bank

    window_size = 400
    hop_length = 160
    num_mel_bins = 128
    global_log_mel_max = 1.5

    audio_tensor = torch.from_numpy(audio_array).float()
    window = torch.hann_window(window_size)
    stft = torch.stft(
        audio_tensor,
        window_size,
        hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + window_size // 2,
        num_mel_bins=num_mel_bins,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sampling_rate,
    )
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32)
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec_max = torch.tensor(global_log_mel_max)
    log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec

def run_encoder(mel, f):
    """Run mel through encoder layers."""
    def load_weight(name):
        return f.get_tensor(name).float()

    # Transpose mel to [batch, channels, time]
    mel = mel.unsqueeze(0)  # [1, 128, T]

    # Conv layers
    conv1_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight")
    conv1_b = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.bias")
    conv2_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.weight")
    conv2_b = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv.bias")

    x = F.gelu(F.conv1d(mel, conv1_w, conv1_b, stride=2, padding=1))
    x = F.gelu(F.conv1d(x, conv2_w, conv2_b, stride=2, padding=1))
    x = x.transpose(1, 2)  # [B, T, C]

    # Encoder layers
    n_heads = 32
    head_dim = 64

    for layer_idx in range(32):
        prefix = f"mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.{layer_idx}"

        attn_norm_w = load_weight(f"{prefix}.attention_norm.weight")
        wq = load_weight(f"{prefix}.attention.wq.weight")
        wk = load_weight(f"{prefix}.attention.wk.weight")
        wv = load_weight(f"{prefix}.attention.wv.weight")
        wo = load_weight(f"{prefix}.attention.wo.weight")
        wq_b = load_weight(f"{prefix}.attention.wq.bias")
        wv_b = load_weight(f"{prefix}.attention.wv.bias")
        wo_b = load_weight(f"{prefix}.attention.wo.bias")
        ffn_norm_w = load_weight(f"{prefix}.ffn_norm.weight")
        w1 = load_weight(f"{prefix}.feed_forward.w1.weight")
        w2 = load_weight(f"{prefix}.feed_forward.w2.weight")
        w3 = load_weight(f"{prefix}.feed_forward.w3.weight")
        w2_b = load_weight(f"{prefix}.feed_forward.w2.bias")

        seq_len = x.shape[1]
        batch = x.shape[0]

        # Attention
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

        # FFN
        residual = x.clone()
        x_normed = rms_norm(x, ffn_norm_w)
        mlp_out = F.linear(F.silu(F.linear(x_normed, w1)) * F.linear(x_normed, w3), w2, w2_b)
        x = mlp_out + residual

    # Final norm
    encoder_norm_w = load_weight("mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm.weight")
    x = rms_norm(x, encoder_norm_w)

    # Reshape (4x pooling)
    seq_len = x.shape[1]
    new_seq = seq_len // 4
    x = x[:, :new_seq*4, :].reshape(1, new_seq, 5120)

    # Adapter
    adapter_w_in = load_weight("mm_streams_embeddings.embedding_module.audio_language_projection.0.weight")
    adapter_w_out = load_weight("mm_streams_embeddings.embedding_module.audio_language_projection.2.weight")
    x = F.gelu(F.linear(x, adapter_w_in))
    x = F.linear(x, adapter_w_out)

    return x

def run_decoder_step(audio_embed, text_embed, t_embed, f, position=0):
    """Run a single decoder step."""
    def load_weight(name):
        return f.get_tensor(name).float()

    tok_emb = load_weight("mm_streams_embeddings.embedding_module.tok_embeddings.weight")

    x = audio_embed + text_embed  # [1, 1, 3072]

    for layer_idx in range(26):
        prefix = f"layers.{layer_idx}"

        attn_norm_w = load_weight(f"{prefix}.attention_norm.weight")
        ffn_norm_w = load_weight(f"{prefix}.ffn_norm.weight")
        ada_w0 = load_weight(f"{prefix}.ada_rms_norm_t_cond.0.weight")
        ada_w2 = load_weight(f"{prefix}.ada_rms_norm_t_cond.2.weight")
        wq = load_weight(f"{prefix}.attention.wq.weight")
        wk = load_weight(f"{prefix}.attention.wk.weight")
        wv = load_weight(f"{prefix}.attention.wv.weight")
        wo = load_weight(f"{prefix}.attention.wo.weight")
        w1 = load_weight(f"{prefix}.feed_forward.w1.weight")
        w2 = load_weight(f"{prefix}.feed_forward.w2.weight")
        w3 = load_weight(f"{prefix}.feed_forward.w3.weight")

        n_heads_q = 32
        n_kv_heads = 8
        head_dim = 128
        batch, seq_len, _ = x.shape

        # Attention
        residual = x.clone()
        x_normed = rms_norm(x, attn_norm_w)
        q = F.linear(x_normed, wq)
        k = F.linear(x_normed, wk)
        v = F.linear(x_normed, wv)

        q = q.view(batch, seq_len, n_heads_q, head_dim)
        k = k.view(batch, seq_len, n_kv_heads, head_dim)
        v = v.view(batch, seq_len, n_kv_heads, head_dim)

        cos, sin = rope_freqs(head_dim, position + seq_len)
        q_rope = apply_rope(q, cos[position:position+seq_len], sin[position:position+seq_len])
        k_rope = apply_rope(k, cos[position:position+seq_len], sin[position:position+seq_len])

        # GQA
        n_rep = n_heads_q // n_kv_heads
        k_rope = k_rope.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len, n_heads_q, head_dim)
        v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch, seq_len, n_heads_q, head_dim)

        q_t = q_rope.transpose(1, 2)
        k_t = k_rope.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = F.linear(out, wo)
        x = attn_out + residual

        # FFN with ADA
        residual = x.clone()
        x_normed = rms_norm(x, ffn_norm_w)
        t_embed_expanded = t_embed.unsqueeze(1).expand(-1, seq_len, -1)
        scale = F.gelu(F.linear(t_embed_expanded, ada_w0))
        scale = F.linear(scale, ada_w2)
        x_ada = x_normed * (1 + scale)
        mlp_out = F.linear(F.silu(F.linear(x_ada, w1)) * F.linear(x_ada, w3), w2)
        x = mlp_out + residual

    # Final norm and LM head
    final_norm_w = load_weight("norm.weight")
    hidden = rms_norm(x, final_norm_w)
    logits = torch.matmul(hidden, tok_emb.T)

    return logits, hidden

def main():
    audio_path = "test_data/mary_had_lamb.wav"
    if not Path(audio_path).exists():
        print(f"Error: {audio_path} not found")
        return

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MistralTokenizer.from_file(TOKENIZER_PATH)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = Audio.from_file(audio_path, strict=False)
    print(f"  Duration: {audio.duration:.2f}s at {audio.sampling_rate}Hz")

    # Encode with mistral-common (this pads the audio!)
    print("\nEncoding with mistral-common (OFFLINE mode for full padding)...")
    request = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        language="en",
    )
    tokenized = tokenizer.encode_transcription(request)

    print(f"  Prefix tokens: {tokenized.tokens}")
    print(f"  Prefix length: {len(tokenized.tokens)}")

    padded_audio = tokenized.audios[0]
    print(f"  Padded audio duration: {padded_audio.duration:.2f}s")
    print(f"  Padded audio samples: {len(padded_audio.audio_array)}")

    # Compute mel from PADDED audio
    print("\nComputing mel from padded audio...")
    mel = compute_mel(padded_audio.audio_array, padded_audio.sampling_rate)
    print(f"  Mel shape: {mel.shape}")

    # Calculate expected positions
    audio_config = tokenizer.instruct_tokenizer.audio_encoder.audio_config
    num_audio_tokens = audio_config.num_audio_tokens(len(padded_audio.audio_array))
    print(f"  Expected audio tokens: {num_audio_tokens}")

    # Run encoder
    print("\nRunning encoder...")
    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        audio_embeds = run_encoder(mel, f)
        print(f"  Audio embeds shape: {audio_embeds.shape}")

        # Get token embeddings
        tok_emb = f.get_tensor("mm_streams_embeddings.embedding_module.tok_embeddings.weight").float()

        seq_len = audio_embeds.shape[1]
        prefix_len = len(tokenized.tokens)

        print(f"\n=== Inference Setup ===")
        print(f"  Audio positions: {seq_len}")
        print(f"  Prefix tokens: {prefix_len}")
        print(f"  Positions for transcription: {seq_len - prefix_len + 1}")

        # Time embedding
        num_delay_tokens = audio_config.num_delay_tokens
        t_embed = time_embedding(torch.tensor([float(num_delay_tokens)]), dim=3072)
        print(f"  t_embed for t={num_delay_tokens}")

        # Run full forward pass with prefix
        print("\n=== Running full forward with prefix tokens ===")
        # Create text embeddings for prefix positions
        prefix_text_embeds = tok_emb[tokenized.tokens].unsqueeze(0)  # [1, prefix_len, 3072]

        # Pad text embeddings to match audio length (using [STREAMING_PAD] for positions beyond prefix)
        # Token 32 is [STREAMING_PAD] per our earlier investigation
        streaming_pad_id = 32
        if seq_len > prefix_len:
            pad_embeds = tok_emb[streaming_pad_id:streaming_pad_id+1].expand(1, seq_len - prefix_len, -1)
            text_embeds = torch.cat([prefix_text_embeds, pad_embeds], dim=1)
        else:
            text_embeds = prefix_text_embeds[:, :seq_len]

        print(f"  Text embeds shape: {text_embeds.shape}")

        # Run decoder
        inputs_embeds = audio_embeds + text_embeds
        logits, hidden = run_decoder_step(audio_embeds, text_embeds, t_embed, f, position=0)

        print(f"  Logits shape: {logits.shape}")

        # Get predictions
        predictions = logits.argmax(dim=-1)[0].tolist()
        print(f"\nPredicted tokens (first 20): {predictions[:20]}")
        print(f"Predicted tokens (around prefix boundary {prefix_len-1}):")
        for pos in range(max(0, prefix_len-3), min(seq_len, prefix_len+5)):
            top5 = logits[0, pos].topk(5)
            print(f"  Position {pos}: pred={predictions[pos]}, top5={top5.indices.tolist()}, logits={[f'{v:.2f}' for v in top5.values.tolist()]}")

        # Decode predictions (skip control tokens)
        text_tokens = [t for t in predictions if t >= 1000]  # Skip control tokens
        if text_tokens:
            decoded = tokenizer.decode(text_tokens)
            print(f"\nDecoded text (non-control tokens): {decoded}")
        else:
            print("\nNo text tokens predicted (all control tokens)")

        # Check what the model predicts at the LAST position (for autoregressive)
        print(f"\n=== Last position analysis ===")
        last_pos = seq_len - 1
        top10 = logits[0, last_pos].topk(10)
        print(f"  Position {last_pos} top 10: tokens={top10.indices.tolist()}")
        print(f"  Logits: {[f'{v:.2f}' for v in top10.values.tolist()]}")

        # === Autoregressive generation ===
        print("\n" + "="*60)
        print("=== AUTOREGRESSIVE GENERATION ===")
        print("="*60)

        # The key insight: in streaming, we generate one token at a time,
        # feeding back the PREVIOUSLY GENERATED token as input.
        #
        # In a causal model:
        #   - Output at position N predicts position N+1
        #   - The text input at position N should be the token we generated at N
        #     (or prefix tokens for the first prefix_len positions)

        # Start with prefix tokens
        generated_tokens = list(tokenized.tokens)  # Copy prefix

        # For each position beyond the prefix, generate autoregressively
        # Note: Without KV cache, this is SLOW (O(n^2) in sequence length)
        # But it proves the concept works

        num_tokens_to_generate = min(30, seq_len - prefix_len)  # Limit for speed
        print(f"\nGenerating {num_tokens_to_generate} tokens autoregressively...")
        print(f"(This is slow without KV cache)\n")

        for step in range(num_tokens_to_generate):
            current_len = prefix_len + step

            # Build text embeddings: prefix + all previously generated tokens
            current_tokens = generated_tokens[:current_len]
            text_embeds = tok_emb[current_tokens].unsqueeze(0)  # [1, current_len, 3072]

            # Run decoder for positions 0..current_len
            audio_slice = audio_embeds[:, :current_len, :]
            logits, hidden = run_decoder_step(audio_slice, text_embeds, t_embed, f, position=0)

            # Get prediction for next token (from last position)
            top5 = logits[0, -1].topk(5)
            next_token = top5.indices[0].item()
            generated_tokens.append(next_token)

            # Show progress with more detail
            if step < 15 or step % 5 == 0:
                logit_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                # Check hidden state norm
                hidden_norm = hidden[0, -1].norm().item()
                print(f"  Step {step}: pos={current_len-1}, input_tok={current_tokens[-1]}, "
                      f"pred={next_token}, top5={top5.indices.tolist()}, "
                      f"logits=[{', '.join(logit_vals)}], hidden_norm={hidden_norm:.2f}")

        print(f"\n=== Generated tokens (beyond prefix) ===")
        transcription_tokens = generated_tokens[prefix_len:]
        print(f"  Raw tokens: {transcription_tokens}")

        # Count token types
        pad_count = sum(1 for t in transcription_tokens if t == 32)
        word_count = sum(1 for t in transcription_tokens if t == 33)
        text_count = sum(1 for t in transcription_tokens if t >= 1000)
        print(f"\n  Token breakdown:")
        print(f"    [STREAMING_PAD] (32): {pad_count}")
        print(f"    [STREAMING_WORD] (33): {word_count}")
        print(f"    Text tokens (>=1000): {text_count}")

        # === Experiment: Truncate input to just prefix (no future padding) ===
        print("\n" + "="*60)
        print("=== EXPERIMENT: Truncate input to prefix only (no future tokens) ===")
        print("="*60)

        # Only use prefix tokens and corresponding audio
        prefix_text_only = tok_emb[tokenized.tokens].unsqueeze(0)  # [1, 39, 3072]
        prefix_audio_only = audio_embeds[:, :prefix_len, :]  # [1, 39, 3072]

        logits_truncated, _ = run_decoder_step(prefix_audio_only, prefix_text_only, t_embed, f, position=0)
        print(f"Truncated input: {prefix_len} positions")
        print(f"Logits shape: {logits_truncated.shape}")

        # Check last few positions
        for pos in range(max(0, prefix_len-5), prefix_len):
            top5 = logits_truncated[0, pos].topk(5)
            logit_vals = [f"{v:.2f}" for v in top5.values.tolist()]
            print(f"  Position {pos}: pred={top5.indices[0].item()}, top5={top5.indices.tolist()}, logits=[{', '.join(logit_vals)}]")

        # Now generate autoregressively from the truncated prefix
        print("\n=== Autoregressive from truncated prefix ===")
        truncated_generated = list(tokenized.tokens)  # Start with prefix

        for step in range(min(25, seq_len - prefix_len)):
            current_len = prefix_len + step

            current_tokens = truncated_generated[:current_len]
            text_embeds = tok_emb[current_tokens].unsqueeze(0)

            audio_slice = audio_embeds[:, :current_len, :]
            logits, _ = run_decoder_step(audio_slice, text_embeds, t_embed, f, position=0)

            top5 = logits[0, -1].topk(5)
            next_token = top5.indices[0].item()
            truncated_generated.append(next_token)

            if step < 12 or step % 5 == 0:
                logit_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                print(f"  Step {step}: pos={current_len-1}, input={current_tokens[-1]}, "
                      f"pred={next_token}, top5={top5.indices.tolist()}, logits=[{', '.join(logit_vals)}]")

        truncated_transcription = truncated_generated[prefix_len:]
        print(f"\nTranscription tokens: {truncated_transcription}")

        text_only = [t for t in truncated_transcription if t >= 1000]
        if text_only:
            try:
                decoded = tokenizer.decode(text_only)
                print(f"Decoded text: '{decoded}'")
            except Exception as e:
                print(f"Could not decode: {e}")

        # === Experiment: Force first token to [STREAMING_WORD] and see what happens ===
        print("\n" + "="*60)
        print("=== EXPERIMENT: Force first token to [STREAMING_WORD] (33) ===")
        print("="*60)

        # Reset generated tokens with forced [STREAMING_WORD] as first post-prefix token
        forced_tokens = list(tokenized.tokens) + [33]  # prefix + forced [STREAMING_WORD]

        for step in range(min(20, seq_len - prefix_len - 1)):
            current_len = prefix_len + 1 + step  # +1 because we already have one forced token

            current_tokens = forced_tokens[:current_len]
            text_embeds = tok_emb[current_tokens].unsqueeze(0)

            audio_slice = audio_embeds[:, :current_len, :]
            logits, _ = run_decoder_step(audio_slice, text_embeds, t_embed, f, position=0)

            top5 = logits[0, -1].topk(5)
            next_token = top5.indices[0].item()
            forced_tokens.append(next_token)

            if step < 10 or step % 5 == 0:
                logit_vals = [f"{v:.2f}" for v in top5.values.tolist()]
                print(f"  Step {step}: pos={current_len-1}, input={current_tokens[-1]}, "
                      f"pred={next_token}, top5={top5.indices.tolist()}")

        forced_transcription = forced_tokens[prefix_len:]
        print(f"\nForced transcription tokens: {forced_transcription}")

        text_only = [t for t in forced_transcription if t >= 1000]
        if text_only:
            try:
                decoded = tokenizer.decode(text_only)
                print(f"Decoded text: '{decoded}'")
            except Exception as e:
                print(f"Could not decode: {e}")

        # Try to decode text tokens
        text_only = [t for t in transcription_tokens if t >= 1000]
        if text_only:
            try:
                decoded = tokenizer.decode(text_only)
                print(f"\n  Decoded text: '{decoded}'")
            except Exception as e:
                print(f"\n  Could not decode: {e}")
        else:
            print("\n  No text tokens to decode")

if __name__ == "__main__":
    main()
