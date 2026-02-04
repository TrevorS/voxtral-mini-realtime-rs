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
- [ ] RMSNorm (standard)
- [ ] RoPE embeddings (theta=1M)
- [ ] SwiGLU MLP
- [ ] ADA RMSNorm (t-conditional, used in BOTH encoder and LLM)

Phase 3 - Audio Encoder:
- [ ] Conv1d downsampler (128→1280, stride 2×2 = 4x downsample)
- [ ] Causal self-attention with sliding window (750)
- [ ] Full encoder layer (attention + MLP + norms)
- [ ] 32-layer encoder stack

Phase 4 - Language Model:
- [ ] Token embeddings (vocab=131072)
- [ ] GQA attention (32Q/8KV)
- [ ] Sliding window (8192)
- [ ] 26-layer decoder stack
- [ ] LM head (tied embeddings)

Phase 5 - Integration:
- [ ] AudioLanguageAdapter projection
- [ ] Weight loading from SafeTensors
- [ ] KV cache management
- [ ] End-to-end forward pass

Phase 6 - Streaming & Testing:
- [ ] Streaming inference loop
- [ ] Test with real audio (use `/whisper-test` skill to validate transcriptions)
- [ ] WASM build test

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

**Current Status:** Check `docs/CONTINUATION.md` for where we left off.

**Start:** Pick up from the first unchecked item in Phase 2.
