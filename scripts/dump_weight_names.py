#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "safetensors>=0.4",
#     "torch>=2.0",
# ]
# ///
"""Dump all weight names with full paths."""

from pathlib import Path
from safetensors import safe_open

MODEL_PATH = Path("models/voxtral/consolidated.safetensors")

with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())

    # Group by prefix
    by_prefix = {}
    for k in keys:
        parts = k.split(".")
        if parts[0] == "mm_streams_embeddings":
            prefix = ".".join(parts[:3])  # e.g., mm_streams_embeddings.embedding_modules.audio
        elif parts[0] == "layers":
            prefix = f"layers.{parts[1]}"  # e.g., layers.0
        else:
            prefix = parts[0]

        if prefix not in by_prefix:
            by_prefix[prefix] = []
        by_prefix[prefix].append(k)

    print("=" * 80)
    print("WEIGHT NAME STRUCTURE")
    print("=" * 80)

    for prefix in sorted(by_prefix.keys()):
        names = by_prefix[prefix]
        print(f"\n{prefix} ({len(names)} weights):")
        # Show first layer's structure as example
        if prefix.startswith("layers.0") or prefix.endswith(".audio") or not prefix.startswith("layers"):
            for name in sorted(names)[:20]:
                tensor = f.get_tensor(name)
                shape = "×".join(str(d) for d in tensor.shape)
                print(f"  {name}: [{shape}]")
            if len(names) > 20:
                print(f"  ... and {len(names) - 20} more")
