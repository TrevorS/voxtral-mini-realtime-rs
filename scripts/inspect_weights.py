#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "safetensors>=0.4",
#     "rich>=13.0",
#     "torch>=2.0",
# ]
# ///
"""
Inspect Voxtral SafeTensors weights.

Usage:
    ./scripts/inspect_weights.py                    # Full summary
    ./scripts/inspect_weights.py --filter encoder   # Filter by name pattern
    ./scripts/inspect_weights.py --layer 0          # Show specific layer
    ./scripts/inspect_weights.py --shape 1280       # Find tensors with dim 1280
    ./scripts/inspect_weights.py --dump embed       # Dump tensor stats matching pattern
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from safetensors import safe_open

MODEL_PATH = Path("models/voxtral/consolidated.safetensors")


def format_shape(shape: tuple) -> str:
    return "×".join(str(d) for d in shape)


def format_size(num_params: int) -> str:
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    return str(num_params)


def categorize_weight(name: str) -> str:
    """Categorize weight by component."""
    if name.startswith("layers."):
        return "LLM Decoder"
    if name.startswith("mm_streams_embeddings.embedding_module.whisper_encoder"):
        return "Audio Encoder"
    if name.startswith("mm_streams_embeddings.embedding_module.audio_language_projection"):
        return "Adapter"
    if name.startswith("mm_streams_embeddings.embedding_module.tok_embeddings"):
        return "Token Embeddings"
    if name == "norm.weight":
        return "Final Norm"
    return "Other"


def inspect_weights(
    model_path: Path,
    filter_pattern: str | None = None,
    layer_num: int | None = None,
    shape_filter: int | None = None,
    dump_pattern: str | None = None,
    show_all: bool = False,
):
    console = Console()

    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        console.print("Run: ./scripts/download_model.py")
        return

    console.print(f"[bold]Loading:[/bold] {model_path}")
    console.print()

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Build weight info
        weights = []
        for key in sorted(keys):
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            num_params = tensor.numel()
            category = categorize_weight(key)
            weights.append({
                "name": key,
                "shape": shape,
                "dtype": dtype,
                "params": num_params,
                "category": category,
            })

        # Apply filters
        if filter_pattern:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            weights = [w for w in weights if pattern.search(w["name"])]

        if layer_num is not None:
            layer_pattern = f".layers.{layer_num}."
            weights = [w for w in weights if layer_pattern in w["name"]]

        if shape_filter:
            weights = [w for w in weights if shape_filter in w["shape"]]

        # Dump tensor stats
        if dump_pattern:
            pattern = re.compile(dump_pattern, re.IGNORECASE)
            for w in weights:
                if pattern.search(w["name"]):
                    tensor = f.get_tensor(w["name"])
                    console.print(f"\n[bold]{w['name']}[/bold]")
                    console.print(f"  Shape: {format_shape(w['shape'])}")
                    console.print(f"  Dtype: {w['dtype']}")
                    console.print(f"  Min: {tensor.min().item():.6f}")
                    console.print(f"  Max: {tensor.max().item():.6f}")
                    console.print(f"  Mean: {tensor.float().mean().item():.6f}")
                    console.print(f"  Std: {tensor.float().std().item():.6f}")
            return

        # Summary by category
        if not filter_pattern and layer_num is None and shape_filter is None:
            console.print("[bold]Summary by Component[/bold]")
            by_category = defaultdict(lambda: {"count": 0, "params": 0})
            for w in weights:
                by_category[w["category"]]["count"] += 1
                by_category[w["category"]]["params"] += w["params"]

            table = Table()
            table.add_column("Component")
            table.add_column("Tensors", justify="right")
            table.add_column("Parameters", justify="right")

            total_params = 0
            for cat in ["Audio Encoder", "LLM Decoder", "Token Embeddings", "Adapter", "Final Norm", "Other"]:
                if cat in by_category:
                    info = by_category[cat]
                    table.add_row(cat, str(info["count"]), format_size(info["params"]))
                    total_params += info["params"]

            table.add_row("[bold]Total[/bold]", f"[bold]{len(weights)}[/bold]", f"[bold]{format_size(total_params)}[/bold]")
            console.print(table)
            console.print()

            # Show layer structure
            console.print("[bold]Layer Structure[/bold]")

            # Encoder layers
            encoder_layers = set()
            for w in weights:
                if match := re.search(r"encoder\.layers\.(\d+)\.", w["name"]):
                    encoder_layers.add(int(match.group(1)))
            if encoder_layers:
                console.print(f"  Encoder: {len(encoder_layers)} layers (0-{max(encoder_layers)})")

            # LLM layers
            llm_layers = set()
            for w in weights:
                if match := re.search(r"model\.layers\.(\d+)\.", w["name"]):
                    llm_layers.add(int(match.group(1)))
            if llm_layers:
                console.print(f"  LLM: {len(llm_layers)} layers (0-{max(llm_layers)})")

            console.print()

        # Weight table (limited unless --all)
        display_weights = weights if show_all else weights[:50]

        table = Table(title=f"Weights ({len(weights)} total)" if not show_all else "All Weights")
        table.add_column("Name", style="cyan", max_width=60)
        table.add_column("Shape", justify="right")
        table.add_column("Dtype")
        table.add_column("Params", justify="right")

        for w in display_weights:
            table.add_row(
                w["name"],
                format_shape(w["shape"]),
                w["dtype"],
                format_size(w["params"]),
            )

        if not show_all and len(weights) > 50:
            table.add_row("...", "...", "...", "...")
            table.add_row(f"[dim]({len(weights) - 50} more)[/dim]", "", "", "")

        console.print(table)

        # Unique shapes
        if not filter_pattern and layer_num is None:
            console.print()
            console.print("[bold]Unique Shapes[/bold]")
            shapes = defaultdict(int)
            for w in weights:
                shapes[w["shape"]] += 1
            for shape, count in sorted(shapes.items(), key=lambda x: -x[1])[:20]:
                console.print(f"  {format_shape(shape)}: {count}×")


def main():
    parser = argparse.ArgumentParser(description="Inspect Voxtral SafeTensors weights")
    parser.add_argument("--model", "-m", type=Path, default=MODEL_PATH, help="Model path")
    parser.add_argument("--filter", "-f", type=str, help="Filter weights by regex pattern")
    parser.add_argument("--layer", "-l", type=int, help="Show specific layer number")
    parser.add_argument("--shape", "-s", type=int, help="Find tensors containing this dimension")
    parser.add_argument("--dump", "-d", type=str, help="Dump tensor stats for matching pattern")
    parser.add_argument("--all", "-a", action="store_true", help="Show all weights (not just first 50)")

    args = parser.parse_args()
    inspect_weights(
        args.model,
        filter_pattern=args.filter,
        layer_num=args.layer,
        shape_filter=args.shape,
        dump_pattern=args.dump,
        show_all=args.all,
    )


if __name__ == "__main__":
    main()
