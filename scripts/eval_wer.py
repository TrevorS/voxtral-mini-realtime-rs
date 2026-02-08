#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "jiwer",
#     "soundfile",
# ]
# ///
"""
WER evaluation for Voxtral Mini 4B Realtime.

Evaluates word error rate against standard ASR datasets by running the
compiled voxtral-transcribe binary on each utterance.

Primary: FLEURS English (en_us) test set — 350 sentences, ~12 min total.
  Mistral benchmark: 4.90% WER at 480ms delay.

Secondary: LibriSpeech test-clean — 2,620 utterances, ~5 hours.

Usage:
    uv run --script scripts/eval_wer.py -- \\
        --dataset fleurs --gguf models/voxtral-q4.gguf \\
        --tokenizer models/voxtral/tekken.json --delay 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
import soundfile as sf


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


def build_transcribe_cmd(
    audio_path: str,
    gguf: str | None,
    model: str | None,
    tokenizer: str | None,
    delay: int,
) -> list[str]:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--features",
        "wgpu,cli",
        "--bin",
        "voxtral-transcribe",
        "--",
        "--audio",
        audio_path,
        "--delay",
        str(delay),
    ]
    if gguf:
        cmd.extend(["--gguf", gguf])
        if tokenizer:
            cmd.extend(["--tokenizer", tokenizer])
    elif model:
        cmd.extend(["--model", model])
        if tokenizer:
            cmd.extend(["--tokenizer", tokenizer])
    return cmd


def transcribe(
    audio_path: str,
    gguf: str | None,
    model: str | None,
    tokenizer: str | None,
    delay: int,
) -> str | None:
    cmd = build_transcribe_cmd(audio_path, gguf, model, tokenizer, delay)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()[:200]}", file=sys.stderr)
            return None
        # The transcription is the last non-empty line of stdout
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None
    except subprocess.TimeoutExpired:
        print("  ERROR: transcription timed out", file=sys.stderr)
        return None


def normalize_text(text: str) -> str:
    """Basic text normalization for WER comparison."""
    return jiwer.RemoveMultipleSpaces()(
        jiwer.Strip()(jiwer.ToLowerCase()(jiwer.RemovePunctuation()(text)))
    )


def load_fleurs(split: str = "test") -> list[tuple[str, any, str]]:
    """Load FLEURS English test set. Returns [(id, audio_array, reference_text)]."""
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

    # Map user-friendly name to HF split name
    split_map = {
        "test-clean": "test.clean",
        "test.clean": "test.clean",
        "test-other": "test.other",
        "test.other": "test.other",
    }
    hf_split = split_map.get(split, split)

    print(f"Loading LibriSpeech {hf_split}...")
    ds = load_dataset(
        "openslr/librispeech_asr", split=hf_split, trust_remote_code=True
    )
    items = []
    for row in ds:
        audio = row["audio"]
        items.append((row["id"], audio, row["text"]))
    print(f"  Loaded {len(items)} utterances")
    return items


def evaluate(
    items: list[tuple[str, any, str]],
    gguf: str | None,
    model: str | None,
    tokenizer: str | None,
    delay: int,
    dataset_name: str,
) -> EvalReport:
    results: list[UtteranceResult] = []
    references: list[str] = []
    hypotheses: list[str] = []
    failed = 0
    total_audio_secs = 0.0

    wall_start = time.time()

    for i, (uid, audio, ref_text) in enumerate(items):
        # Write audio to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_array = audio["array"]
            sr = audio["sampling_rate"]
            sf.write(tmp_path, audio_array, sr)
            audio_dur = len(audio_array) / sr

        total_audio_secs += audio_dur

        print(
            f"  [{i + 1}/{len(items)}] {uid} ({audio_dur:.1f}s)...",
            end="",
            flush=True,
        )

        hyp = transcribe(tmp_path, gguf, model, tokenizer, delay)
        Path(tmp_path).unlink(missing_ok=True)

        if hyp is None:
            failed += 1
            print(" FAILED")
            continue

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
                audio_duration_secs=audio_dur,
            )
        )
        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        print(f" WER={utt_wer:.1%}")

    wall_secs = time.time() - wall_start

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
