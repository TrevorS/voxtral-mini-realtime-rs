.PHONY: build build-release build-wasm lint lint-wasm fmt test bench bench-audio bench-q4 bench-e2e profile-chrome profile-flamegraph eval-wer-fleurs eval-wer-libri clean

# Build
build:
	cargo build --features "wgpu,cli,hub"

build-release:
	cargo build --release --features "wgpu,cli,hub"

build-wasm:
	wasm-pack build --target web --no-default-features --features wasm

# Lint & Format
lint:
	cargo clippy --features "wgpu,cli,hub" -- -D warnings

lint-wasm:
	cargo clippy --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

# Test
test:
	cargo test --features "wgpu,cli,hub"

# Benchmarks
bench-audio:
	cargo bench --bench audio

bench-q4:
	cargo bench --bench q4_ops --features wgpu

bench: bench-audio bench-q4

bench-e2e:
	cargo run --release --features "wgpu,cli" --bin e2e-bench -- \
		--audio test_data/mary_had_lamb.wav \
		--gguf models/voxtral-q4.gguf \
		--tokenizer models/voxtral/tekken.json

# Profiling
profile-chrome:
	cargo run --profile profiling --features "wgpu,cli,profiling" --bin voxtral-transcribe -- \
		--audio test_data/mary_had_lamb.wav \
		--gguf models/voxtral-q4.gguf \
		--tokenizer models/voxtral/tekken.json
	@echo "Trace written to trace.json — open in chrome://tracing or https://ui.perfetto.dev"

profile-flamegraph:
	cargo flamegraph --features "wgpu,cli" --bin voxtral-transcribe -- \
		--audio test_data/mary_had_lamb.wav \
		--gguf models/voxtral-q4.gguf \
		--tokenizer models/voxtral/tekken.json

# WER Evaluation
eval-wer-fleurs:
	uv run --script scripts/eval_wer.py -- \
		--dataset fleurs \
		--gguf models/voxtral-q4.gguf \
		--tokenizer models/voxtral/tekken.json \
		--delay 6

eval-wer-libri:
	uv run --script scripts/eval_wer.py -- \
		--dataset librispeech-clean \
		--gguf models/voxtral-q4.gguf \
		--tokenizer models/voxtral/tekken.json \
		--delay 6

# Cleanup
clean:
	cargo clean
	rm -rf pkg/ trace.json flamegraph.svg perf.data perf.data.old
