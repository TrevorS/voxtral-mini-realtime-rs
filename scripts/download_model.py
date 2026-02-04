#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.20",
# ]
# ///
"""
Download Voxtral Mini 4B Realtime model weights from HuggingFace.

Usage:
    ./scripts/download_model.py [--output-dir models/voxtral]

The script downloads:
    - consolidated.safetensors (8.86 GB) - Model weights in BF16
    - params.json (1.34 KB) - Architecture configuration
    - tekken.json (14.9 MB) - Tokenizer vocabulary

Requires HF_TOKEN environment variable for gated model access.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

# Files to download
MODEL_FILES = [
    "consolidated.safetensors",  # 8.86 GB - model weights
    "params.json",               # 1.34 KB - architecture config
    "tekken.json",               # 14.9 MB - tokenizer
]


def download_model(output_dir: Path, use_snapshot: bool = False) -> None:
    """Download Voxtral model files to output directory."""

    # HF token is optional (model is not gated)
    token = os.environ.get("HF_TOKEN")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {output_dir}")
    print()

    if use_snapshot:
        # Download entire repo (useful if there are more files)
        print("Using snapshot download (full repo)...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=output_dir,
            token=token,
            local_dir_use_symlinks=False,
        )
    else:
        # Download specific files
        for filename in MODEL_FILES:
            print(f"Downloading {filename}...")
            try:
                local_path = hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=filename,
                    local_dir=output_dir,
                    token=token,
                    local_dir_use_symlinks=False,
                )

                # Get file size
                size = Path(local_path).stat().st_size
                if size > 1_000_000_000:
                    size_str = f"{size / 1_000_000_000:.2f} GB"
                elif size > 1_000_000:
                    size_str = f"{size / 1_000_000:.2f} MB"
                else:
                    size_str = f"{size / 1_000:.2f} KB"

                print(f"  ✓ {filename} ({size_str})")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")
                sys.exit(1)

    print()
    print("Download complete!")
    print()
    print("Model files:")
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            if size > 1_000_000_000:
                size_str = f"{size / 1_000_000_000:.2f} GB"
            elif size > 1_000_000:
                size_str = f"{size / 1_000_000:.2f} MB"
            else:
                size_str = f"{size / 1_000:.2f} KB"
            print(f"  {f.name}: {size_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Voxtral Mini 4B Realtime model from HuggingFace"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("models/voxtral"),
        help="Output directory for model files (default: models/voxtral)"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Download entire repo instead of specific files"
    )

    args = parser.parse_args()

    download_model(args.output_dir, use_snapshot=args.snapshot)


if __name__ == "__main__":
    main()
