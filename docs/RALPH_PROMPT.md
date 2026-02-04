# Voxtral Mini 4B Realtime - Ralph Development Loop

## Mission

Port Mistral's Voxtral Mini 4B Realtime (streaming ASR model) to Rust using the Burn ML framework. Target: streaming speech-to-text with WASM/browser support.

## Reference Materials

### Local Repos (consult for patterns and inspiration)
- `~/Projects/qwen3-tts-rs/` - Similar Burn-based audio model port (TTS instead of ASR)
  - `docs/QWEN3_TTS_ARCHITECTURE.md` - Architecture deep dive patterns
  - `docs/CONTINUATION.md` - Project status tracking patterns
  - `src/models/` - Layer implementations (attention, rope, swiglu)
- `../mistral-common/` - Mistral's reference implementation (if available)
- `../vllm/` - vLLM with inference support for Voxtral

### Online Resources (use WebSearch/WebFetch)
- vLLM Voxtral implementation (search: "vllm voxtral" OR "mistral voxtral implementation")
- mistral-common Python package (Mistral's official tokenizer/model utilities)
- HuggingFace model card: `mistralai/Voxtral-Mini-4B-Realtime-2602`

## Development Workflow

### For Each Component:

1. **Research Phase**
   - Search sibling repos for reference implementations
   - Use `WebSearch` for vLLM, mistral-common patterns
   - Read relevant model weights with `./scripts/inspect_weights.py`
   - Document findings in `docs/` (architecture notes, deviations)

2. **Tool Creation Phase**
   - If validation tooling is missing, create it in `scripts/`
   - Generate reference data: `./scripts/reference_forward.py <component>`
   - Test tools work before using them

3. **Implementation Phase**
   - Write Rust code in `src/models/layers/` or appropriate location
   - Follow patterns from qwen3-tts-rs where applicable
   - Include unit tests that load reference data from `test_data/`

4. **Validation Phase**
   - Run `cargo test` - all tests must pass
   - Run `cargo clippy -- -D warnings` - must be clean
   - Compare outputs: `./scripts/compare_tensors.py test_data/ rust_output/`

5. **Documentation Phase**
   - Update `docs/CONTINUATION.md` with progress
   - Note any deviations from reference in `docs/VOXTRAL_ARCHITECTURE.md`
   - If blocked, document the blocker clearly

### Component Order

Phase 2 - Core Building Blocks:
- [x] RMSNorm (standard)
- [x] RoPE embeddings (theta=1M)
- [x] SwiGLU MLP
- [x] ADA RMSNorm (t-conditional, used in decoder only)

Phase 3 - Audio Encoder:
- [x] Conv1d downsampler (128→1280, stride 2×2 = 4x downsample)
- [x] Causal self-attention with sliding window (750)
- [x] Full encoder layer (attention + MLP + norms)
- [x] 32-layer encoder stack

Phase 4 - Language Model:
- [x] Token embeddings (vocab=131072)
- [x] GQA attention (32Q/8KV)
- [x] Sliding window (8192)
- [x] 26-layer decoder stack
- [x] LM head (tied embeddings)

Phase 5 - Integration:
- [x] AudioLanguageAdapter projection
- [x] Weight loading from SafeTensors
- [x] KV cache management
- [x] End-to-end forward pass

Phase 6 - Streaming & Testing:
- [x] Streaming inference loop
- [x] Test with real audio (verified: " I spoke in the original phonograph...")
- [ ] WASM build test (future work)

## Tools Available

### Scripts
- `./scripts/inspect_weights.py` - Browse SafeTensors structure
- `./scripts/inspect_weights.py --filter <pattern>` - Filter by name
- `./scripts/inspect_weights.py --dump <pattern>` - Get tensor stats
- `./scripts/reference_forward.py <component>` - Generate test data
- `./scripts/compare_tensors.py` - Validate Rust vs Python

### Claude Code Skills
- `/whisper-test <audio.wav>` - Transcribe audio to verify ASR output quality

### Rust Test Utils
```rust
use crate::test_utils::{load_test_data, assert_tensors_close};
let reference = load_test_data("rms_norm_output")?;
assert_tensors_close(&reference, &actual, 1e-3, 1e-5, "rms_norm");
```

## Quality Gates

Before marking any component complete:
1. `cargo test` passes (all tests)
2. `cargo clippy -- -D warnings` is clean
3. `cargo fmt --check` passes
4. Reference data comparison passes (max_abs_diff < 1e-3)
5. docs/ updated with any findings

## Git Workflow

Use jj (jujutsu) for version control:
- `jj desc -m "message"` to describe current change
- `jj new -m "next phase"` to start new work
- Keep working copy described (never "(no description set)")

## Iteration Strategy

Each Ralph iteration should:
1. Check `docs/CONTINUATION.md` for current status
2. Pick the next unchecked item from Component Order
3. Follow the 5-phase workflow (Research → Tools → Implement → Validate → Document)
4. Update CONTINUATION.md status
5. If blocked, clearly document why and move to next item

## Completion Criteria

Output `<promise>VOXTRAL_COMPLETE</promise>` when:
- All Phase 2-5 components are checked off
- End-to-end forward pass works
- At least one real audio file transcribes correctly (verified with /whisper-test)
- All tests pass, clippy clean

## If Stuck

After 10 iterations on same component:
1. Document what's blocking in docs/BLOCKERS.md
2. List attempted approaches
3. Move to next component
4. Return later with fresh context

---

## Critical Streaming Inference Findings (Feb 2026)

### Position 38 Anomaly

The standard prefix is 39 tokens (BOS + 38 `[STREAMING_PAD]`). However, position 38 (0-indexed) exhibits anomalous behavior when it's the last position:

- **Position 38 hidden state norm diverges**: Layer 25 shows pos 38 norm=452 vs pos 36/37 norm=1000-1100
- **All logits at position 38 are very negative** (-17 to -55 range vs +12 at positions 36-37)
- This causes position 38 to always predict `[STREAMING_PAD]` regardless of audio content

**Root cause**: Position 38 = n_left_pad_tokens(32) + num_delay_tokens(6) is exactly at the trained prefix boundary. The model likely learned special "boundary" behavior.

### Working Solution

Use **prefix length 38** (one less than standard) for generation:
```python
prefix_tokens = [1] + [32] * 37  # BOS + 37 STREAMING_PAD = 38 tokens
```

With this prefix, position 37 correctly predicts `[STREAMING_WORD]` (token 33), and autoregressive generation produces correct transcription.

### Verified Transcription Output

Test audio: `test_data/mary_had_lamb.wav` (15.95s, "First words I spoke in the original phonograph...")

With prefix length 38, autoregressively generates:
```
" I spoke in the original phonograph. A little piece of practical poetry"
```

(Missing "First words" is expected - position 38 corresponds to ~2.1s into the speech, after those words.)

### Token Pattern

Streaming inference produces alternating pattern:
- `[STREAMING_WORD]` (33) = start of word
- Text tokens (≥1000) = word content
- `[STREAMING_PAD]` (32) = pause between words

Example: `[WORD] " I" " spoke" " in" " the" [PAD]x8 [WORD] " original" ...`

### Implementation Notes for Rust

1. Use prefix length 38, not 39
2. Implement autoregressive generation with KV cache
3. Feed previous generated token as next input
4. `[STREAMING_WORD]` triggers text generation, `[STREAMING_PAD]` continues silence
5. Time embedding `t=6.0` (num_delay_tokens) is correct

---

**Current Status:** Check `docs/CONTINUATION.md` for where we left off.

**Start:** Pick up from the first unchecked item in Phase 2.
