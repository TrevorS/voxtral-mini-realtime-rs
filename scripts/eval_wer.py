#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets<3",
#     "jiwer",
#     "librosa",
#     "soundfile",
# ]
# ///
"""
WER evaluation for Voxtral Mini 4B Realtime.

Evaluates word error rate against standard ASR datasets by running the
compiled voxtral-transcribe binary in batch mode (model loads once).

Primary: FLEURS English (en_us) test set — 350 sentences, ~12 min total.
  Mistral benchmark: 4.90% WER at 480ms delay.

Secondary: LibriSpeech test-clean — 2,620 utterances, ~5 hours.

Usage:
    uv run --script scripts/eval_wer.py -- \
        --dataset fleurs --gguf models/voxtral-q4.gguf \
        --tokenizer models/voxtral/tekken.json --delay 6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
import soundfile as sf

# Work around root-owned default HF cache on some systems
if "HF_DATASETS_CACHE" not in os.environ:
    default_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    if default_cache.exists() and not os.access(default_cache, os.W_OK):
        os.environ["HF_DATASETS_CACHE"] = str(Path(tempfile.gettempdir()) / "hf_datasets")

BINARY = "target/release/voxtral-transcribe"


@dataclass
class UtteranceResult:
    id: str
    reference: str
    hypothesis: str
    wer: float
    audio_duration_secs: float


@dataclass
class EvalReport:
    dataset: str
    total_utterances: int
    successful: int
    failed: int
    aggregate_wer: float
    aggregate_cer: float
    total_audio_secs: float
    total_wall_secs: float
    rtf: float
    delay_tokens: int
    utterances: list[UtteranceResult] = field(default_factory=list)


def ensure_binary() -> str:
    """Find or build the release binary."""
    if Path(BINARY).exists():
        return BINARY
    print("Release binary not found, building...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "wgpu,cli"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return BINARY


def normalize_text(text: str) -> str:
    """Basic text normalization for WER comparison."""
    return jiwer.RemoveMultipleSpaces()(
        jiwer.Strip()(jiwer.ToLowerCase()(jiwer.RemovePunctuation()(text)))
    )


def load_fleurs(split: str = "test") -> list[tuple[str, any, str]]:
    """Load FLEURS English test set. Returns [(id, audio_dict, reference_text)]."""
    from datasets import load_dataset

    print(f"Loading FLEURS en_us {split} set...")
    ds = load_dataset("google/fleurs", "en_us", split=split, trust_remote_code=True)
    items = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        items.append((f"fleurs_{i}", audio, row["transcription"]))
    print(f"  Loaded {len(items)} utterances")
    return items


def load_librispeech(split: str = "test.clean") -> list[tuple[str, any, str]]:
    """Load LibriSpeech test-clean. Returns [(id, audio_dict, reference_text)]."""
    from datasets import load_dataset

    split_map = {
        "test-clean": "test.clean",
        "test.clean": "test.clean",
        "test-other": "test.other",
        "test.other": "test.other",
    }
    hf_split = split_map.get(split, split)

    print(f"Loading LibriSpeech {hf_split}...")
    ds = load_dataset("openslr/librispeech_asr", split=hf_split, trust_remote_code=True)
    items = []
    for row in ds:
        audio = row["audio"]
        items.append((row["id"], audio, row["text"]))
    print(f"  Loaded {len(items)} utterances")
    return items


def evaluate(
    items: list[tuple[str, any, str]],
    binary: str,
    gguf: str | None,
    model: str | None,
    tokenizer: str | None,
    delay: int,
    dataset_name: str,
) -> EvalReport:
    tmpdir = Path(tempfile.mkdtemp(prefix="voxtral_wer_"))

    try:
        return _evaluate_batch(items, binary, gguf, model, tokenizer, delay, dataset_name, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _evaluate_batch(
    items: list[tuple[str, any, str]],
    binary: str,
    gguf: str | None,
    model: str | None,
    tokenizer: str | None,
    delay: int,
    dataset_name: str,
    tmpdir: Path,
) -> EvalReport:
    # Phase 1: Write all audio to temp WAV files
    print(f"Preparing {len(items)} audio files...")
    wav_paths: list[Path] = []
    durations: list[float] = []
    for i, (uid, audio, _ref_text) in enumerate(items):
        wav_path = tmpdir / f"{i:05d}.wav"
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        sf.write(str(wav_path), audio_array, sr)
        durations.append(len(audio_array) / sr)
        wav_paths.append(wav_path)

    total_audio_secs = sum(durations)
    print(f"  {total_audio_secs:.0f}s of audio ({total_audio_secs / 60:.1f} min)")

    # Phase 2: Write audio list file
    list_file = tmpdir / "audio_list.txt"
    list_file.write_text("\n".join(str(p) for p in wav_paths) + "\n")

    # Phase 3: Run binary once with --audio-list (model loads once)
    cmd = [binary, "--audio-list", str(list_file), "--delay", str(delay)]
    if gguf:
        cmd.extend(["--gguf", gguf])
        if tokenizer:
            cmd.extend(["--tokenizer", tokenizer])
    elif model:
        cmd.extend(["--model", model])
        if tokenizer:
            cmd.extend(["--tokenizer", tokenizer])

    # Timeout: 2x audio duration + 5 min for model loading, minimum 10 min
    timeout_secs = max(600, int(total_audio_secs * 2) + 300)
    print(f"Running batch transcription ({len(items)} files, model loads once, "
          f"timeout={timeout_secs}s)...")
    wall_start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_secs,
    )
    wall_secs = time.time() - wall_start

    if result.returncode != 0:
        print(f"Transcription failed:\n{result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    # Phase 4: Parse output — one transcription per line, matching input order.
    # stdout has exactly one line per audio file (some may be empty). Tracing → stderr.
    # Remove only the single trailing newline from the last println!, keep empty lines.
    raw_output = result.stdout
    if raw_output.endswith("\n"):
        raw_output = raw_output[:-1]
    hyp_lines = raw_output.split("\n") if raw_output else []

    if len(hyp_lines) != len(items):
        print(
            f"WARNING: Expected {len(items)} transcriptions, got {len(hyp_lines)}",
            file=sys.stderr,
        )

    # Phase 5: Compute WER per utterance
    results: list[UtteranceResult] = []
    references: list[str] = []
    hypotheses: list[str] = []
    failed = 0

    for i, (uid, _audio, ref_text) in enumerate(items):
        hyp = hyp_lines[i].strip() if i < len(hyp_lines) else ""

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp)

        if ref_norm:
            utt_wer = jiwer.wer(ref_norm, hyp_norm)
        else:
            utt_wer = 0.0 if not hyp_norm else 1.0

        results.append(
            UtteranceResult(
                id=str(uid),
                reference=ref_text,
                hypothesis=hyp,
                wer=utt_wer,
                audio_duration_secs=durations[i],
            )
        )
        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        status = f"WER={utt_wer:.0%}" if utt_wer > 0 else "OK"
        print(f"  [{i + 1}/{len(items)}] {uid} ({durations[i]:.1f}s) {status}")

    # Aggregate WER/CER
    if references:
        agg_wer = jiwer.wer(references, hypotheses)
        agg_cer = jiwer.cer(references, hypotheses)
    else:
        agg_wer = 0.0
        agg_cer = 0.0

    rtf = wall_secs / total_audio_secs if total_audio_secs > 0 else 0.0

    return EvalReport(
        dataset=dataset_name,
        total_utterances=len(items),
        successful=len(results),
        failed=failed,
        aggregate_wer=agg_wer,
        aggregate_cer=agg_cer,
        total_audio_secs=total_audio_secs,
        total_wall_secs=wall_secs,
        rtf=rtf,
        delay_tokens=delay,
        utterances=results,
    )


def print_report(report: EvalReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"WER Evaluation Report: {report.dataset}")
    print(f"{'=' * 60}")
    print(f"  Utterances:  {report.successful}/{report.total_utterances} "
          f"({report.failed} failed)")
    print(f"  WER:         {report.aggregate_wer:.2%}")
    print(f"  CER:         {report.aggregate_cer:.2%}")
    print(f"  Audio:       {report.total_audio_secs:.1f}s "
          f"({report.total_audio_secs / 60:.1f} min)")
    print(f"  Wall time:   {report.total_wall_secs:.1f}s "
          f"({report.total_wall_secs / 60:.1f} min)")
    print(f"  RTF:         {report.rtf:.2f}x")
    print(f"  Delay:       {report.delay_tokens} tokens "
          f"({report.delay_tokens * 80}ms)")
    print(f"{'=' * 60}")


def save_report(report: EvalReport, path: str) -> None:
    data = {
        "dataset": report.dataset,
        "total_utterances": report.total_utterances,
        "successful": report.successful,
        "failed": report.failed,
        "aggregate_wer": report.aggregate_wer,
        "aggregate_cer": report.aggregate_cer,
        "total_audio_secs": report.total_audio_secs,
        "total_wall_secs": report.total_wall_secs,
        "rtf": report.rtf,
        "delay_tokens": report.delay_tokens,
        "utterances": [
            {
                "id": u.id,
                "reference": u.reference,
                "hypothesis": u.hypothesis,
                "wer": u.wer,
                "audio_duration_secs": u.audio_duration_secs,
            }
            for u in report.utterances
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WER evaluation for Voxtral")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fleurs", "librispeech-clean", "librispeech-other"],
        help="Dataset to evaluate against",
    )
    parser.add_argument("--gguf", help="Path to Q4 GGUF model file")
    parser.add_argument("--model", help="Path to f32 model directory")
    parser.add_argument("--tokenizer", help="Path to tokenizer JSON")
    parser.add_argument("--delay", type=int, default=6, help="Delay tokens (default: 6)")
    parser.add_argument(
        "--output", help="JSON output path (default: wer_{dataset}.json)"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of utterances (for testing)"
    )

    args = parser.parse_args()

    if not args.gguf and not args.model:
        parser.error("Either --gguf or --model is required")

    binary = ensure_binary()

    # Load dataset
    if args.dataset == "fleurs":
        items = load_fleurs()
    elif args.dataset in ("librispeech-clean", "librispeech-other"):
        split = "test.clean" if args.dataset == "librispeech-clean" else "test.other"
        items = load_librispeech(split)
    else:
        parser.error(f"Unknown dataset: {args.dataset}")

    if args.limit:
        items = items[: args.limit]
        print(f"  Limited to {len(items)} utterances")

    # Run evaluation
    report = evaluate(
        items,
        binary=binary,
        gguf=args.gguf,
        model=args.model,
        tokenizer=args.tokenizer,
        delay=args.delay,
        dataset_name=args.dataset,
    )

    print_report(report)

    output_path = args.output or f"wer_{args.dataset}.json"
    save_report(report, output_path)


if __name__ == "__main__":
    main()
