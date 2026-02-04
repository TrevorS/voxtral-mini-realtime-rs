#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "rich>=13.0",
# ]
# ///
"""
Compare Rust output tensors against Python reference.

Usage:
    ./scripts/compare_tensors.py test_data/rms_norm_output.npy rust_output/rms_norm.npy
    ./scripts/compare_tensors.py --dir test_data/ rust_output/  # Compare all matching files
"""

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table


def compare_tensors(ref_path: Path, test_path: Path, rtol: float = 1e-3, atol: float = 1e-5) -> dict:
    """Compare two tensor files."""
    ref = np.load(ref_path)
    test = np.load(test_path)

    if ref.shape != test.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: {ref.shape} vs {test.shape}",
            "ref_shape": ref.shape,
            "test_shape": test.shape,
        }

    # Compute differences
    abs_diff = np.abs(ref - test)
    rel_diff = abs_diff / (np.abs(ref) + 1e-10)

    max_abs_diff = abs_diff.max()
    max_rel_diff = rel_diff.max()
    mean_abs_diff = abs_diff.mean()

    # Check if within tolerance
    close = np.allclose(ref, test, rtol=rtol, atol=atol)

    return {
        "match": close,
        "max_abs_diff": float(max_abs_diff),
        "max_rel_diff": float(max_rel_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "shape": ref.shape,
        "ref_mean": float(ref.mean()),
        "test_mean": float(test.mean()),
        "ref_std": float(ref.std()),
        "test_std": float(test.std()),
    }


def compare_single(ref_path: Path, test_path: Path, rtol: float, atol: float):
    """Compare a single pair of files."""
    console = Console()

    if not ref_path.exists():
        console.print(f"[red]Reference not found: {ref_path}[/red]")
        return False

    if not test_path.exists():
        console.print(f"[red]Test file not found: {test_path}[/red]")
        return False

    result = compare_tensors(ref_path, test_path, rtol, atol)

    if "error" in result:
        console.print(f"[red]ERROR: {result['error']}[/red]")
        return False

    status = "[green]PASS[/green]" if result["match"] else "[red]FAIL[/red]"
    console.print(f"{status} {ref_path.name}")
    console.print(f"  Shape: {result['shape']}")
    console.print(f"  Max abs diff: {result['max_abs_diff']:.2e}")
    console.print(f"  Max rel diff: {result['max_rel_diff']:.2e}")
    console.print(f"  Mean abs diff: {result['mean_abs_diff']:.2e}")
    console.print(f"  Ref mean/std: {result['ref_mean']:.4f} / {result['ref_std']:.4f}")
    console.print(f"  Test mean/std: {result['test_mean']:.4f} / {result['test_std']:.4f}")

    return result["match"]


def compare_directories(ref_dir: Path, test_dir: Path, rtol: float, atol: float):
    """Compare all matching .npy files in two directories."""
    console = Console()

    ref_files = sorted(ref_dir.glob("*.npy"))
    if not ref_files:
        console.print(f"[yellow]No .npy files in {ref_dir}[/yellow]")
        return

    results = []
    for ref_path in ref_files:
        test_path = test_dir / ref_path.name
        if test_path.exists():
            result = compare_tensors(ref_path, test_path, rtol, atol)
            result["name"] = ref_path.stem
            results.append(result)
        else:
            results.append({
                "name": ref_path.stem,
                "match": False,
                "error": "Test file not found",
            })

    # Summary table
    table = Table(title="Tensor Comparison Results")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Max Abs Diff")
    table.add_column("Max Rel Diff")
    table.add_column("Shape")

    passed = 0
    for r in results:
        if "error" in r:
            table.add_row(r["name"], f"[red]{r['error']}[/red]", "-", "-", "-")
        else:
            status = "[green]PASS[/green]" if r["match"] else "[red]FAIL[/red]"
            table.add_row(
                r["name"],
                status,
                f"{r['max_abs_diff']:.2e}",
                f"{r['max_rel_diff']:.2e}",
                str(r["shape"]),
            )
            if r["match"]:
                passed += 1

    console.print(table)
    console.print(f"\n{passed}/{len(results)} passed")


def main():
    parser = argparse.ArgumentParser(description="Compare tensor files")
    parser.add_argument("ref", type=Path, help="Reference file or directory")
    parser.add_argument("test", type=Path, help="Test file or directory")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--dir", action="store_true", help="Compare directories")

    args = parser.parse_args()

    if args.dir or args.ref.is_dir():
        compare_directories(args.ref, args.test, args.rtol, args.atol)
    else:
        compare_single(args.ref, args.test, args.rtol, args.atol)


if __name__ == "__main__":
    main()
