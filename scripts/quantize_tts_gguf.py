#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "safetensors",
#     "torch",
#     "numpy",
# ]
# ///
"""Quantize Voxtral 4B TTS weights from SafeTensors to GGUF v3.

Supports Q4_0 and Q8_0 quantization (selectable via --quant-type).

Reads consolidated.safetensors, quantizes backbone + FM linear layers,
keeps codec/norms/small tensors as F32, pre-fuses codec weight norms, and writes
a valid GGUF v3 file consumable by the Rust reader in src/gguf/reader.rs.

Usage:
    uv run --with safetensors --with torch --with numpy scripts/quantize_tts_gguf.py \\
        models/voxtral-tts/ -o models/voxtral-tts-q8.gguf --quant-type q8_0

    uv run --with safetensors --with torch --with numpy scripts/quantize_tts_gguf.py \\
        models/voxtral-tts/ -o models/voxtral-tts-q4.gguf --quant-type q4_0
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" LE
GGUF_VERSION = 3
ALIGNMENT = 32

# Q4_0 block format: 2 bytes (f16 scale) + 16 bytes (packed nibbles) = 18 bytes per 32 elements
Q4_BLOCK_SIZE = 32
Q4_BLOCK_BYTES = 18

# Q8_0 block format: 2 bytes (f16 scale) + 32 bytes (signed int8) = 34 bytes per 32 elements
Q8_BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 34

# GGML dtype codes matching src/gguf/reader.rs GgmlDtype
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_Q4_0 = 2
DTYPE_Q8_0 = 8

# ---------------------------------------------------------------------------
# Quantization strategy
# ---------------------------------------------------------------------------

# Patterns for tensors that should be quantized (large linear layers).
QUANT_PATTERNS: list[str] = [
    "layers.",  # backbone + FM transformer layers (attention + ffn)
    "mm_audio_embeddings.tok_embeddings.weight",
    "acoustic_transformer.llm_projection.weight",
    "acoustic_transformer.time_projection.weight",
    "acoustic_transformer.semantic_codebook_output.weight",
]

# Patterns for tensors that must stay F32 (small or precision-sensitive).
F32_PATTERNS: list[str] = [
    "norm.weight",        # RMSNorm gammas (attention_norm, ffn_norm, etc.)
    "q_norm.weight",      # QK-norm
    "k_norm.weight",      # QK-norm
    "attention_scale",    # LayerScale
    "ffn_scale",          # LayerScale
    "audio_codebook_embeddings",
    "input_projection.weight",
    "acoustic_codebook_output.weight",
    "audio_tokenizer.",   # all codec tensors
    "semantic_codebook.",  # quantizer codebook
]

# Weight norm tensor suffixes (codec convolutions).
WEIGHT_NORM_G_SUFFIX = ".parametrizations.weight.original0"
WEIGHT_NORM_V_SUFFIX = ".parametrizations.weight.original1"


def should_quantize(name: str) -> bool:
    """Return True if this tensor should be quantized."""
    # F32 patterns take priority — check exclusions first.
    for pat in F32_PATTERNS:
        if pat in name:
            return False
    for pat in QUANT_PATTERNS:
        if pat in name:
            return True
    return False


# ---------------------------------------------------------------------------
# Weight norm fusion
# ---------------------------------------------------------------------------

def fuse_weight_norm(
    g: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Fuse weight normalization: weight = g * v / ||v||.

    Args:
        g: Magnitude [C_out, 1, 1]
        v: Direction [C_out, C_in, K]

    Returns:
        Fused weight [C_out, C_in, K]
    """
    c_out = v.shape[0]
    v_flat = v.reshape(c_out, -1)
    v_norm = torch.norm(v_flat, dim=1, keepdim=True).unsqueeze(-1)  # [C_out, 1, 1]
    return g * v / v_norm


def clean_weight_norm_name(name: str) -> str:
    """Strip weight norm parametrization suffix to get the clean tensor name.

    Example:
        audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0
        → audio_tokenizer.decoder_blocks.0.conv.weight
    """
    for suffix in (WEIGHT_NORM_G_SUFFIX, WEIGHT_NORM_V_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".weight"
    return name


# ---------------------------------------------------------------------------
# Q4_0 quantization (matches src/gguf/tensor.rs dequant + tests.rs quantize)
# ---------------------------------------------------------------------------

def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize a flat f32 array to Q4_0 format.

    Block layout (18 bytes per 32 elements):
      - bytes 0-1: f16 scale `d` (little-endian)
      - bytes 2-17: 16 packed bytes, each byte = quants[i] | (quants[i+16] << 4)
    """
    data = data.astype(np.float32).ravel()
    n = len(data)

    # Pad to multiple of 32 if needed.
    remainder = n % Q4_BLOCK_SIZE
    if remainder != 0:
        pad = Q4_BLOCK_SIZE - remainder
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
        n = len(data)

    n_blocks = n // Q4_BLOCK_SIZE
    output = bytearray(n_blocks * Q4_BLOCK_BYTES)

    for b in range(n_blocks):
        block = data[b * Q4_BLOCK_SIZE : (b + 1) * Q4_BLOCK_SIZE]
        amax = float(np.max(np.abs(block)))
        d = amax / 7.0
        inv_d = 1.0 / d if d != 0.0 else 0.0

        # Scale as f16 LE
        d_f16 = np.float16(d)
        offset = b * Q4_BLOCK_BYTES
        struct.pack_into("<e", output, offset, float(d_f16))

        # Quantize and pack nibbles
        for i in range(16):
            v0 = float(block[i])
            v1 = float(block[i + 16])
            q0 = min(15, int(v0 * inv_d + 8.5))
            q1 = min(15, int(v1 * inv_d + 8.5))
            # Clamp negative (shouldn't happen with +8.5, but safety)
            q0 = max(0, q0)
            q1 = max(0, q1)
            output[offset + 2 + i] = q0 | (q1 << 4)

    return bytes(output)


def q4_byte_size(num_elements: int) -> int:
    """Compute Q4_0 byte size for a given element count (after padding to 32)."""
    n = num_elements
    remainder = n % Q4_BLOCK_SIZE
    if remainder != 0:
        n += Q4_BLOCK_SIZE - remainder
    return (n // Q4_BLOCK_SIZE) * Q4_BLOCK_BYTES


# ---------------------------------------------------------------------------
# Q8_0 quantization
# ---------------------------------------------------------------------------

def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize a flat f32 array to Q8_0 format.

    Block layout (34 bytes per 32 elements):
      - bytes 0-1: f16 scale `d` (little-endian)
      - bytes 2-33: 32 signed int8 values

    Dequantization: value = scale * int8_value
    """
    data = data.astype(np.float32).ravel()
    n = len(data)

    # Pad to multiple of 32 if needed.
    remainder = n % Q8_BLOCK_SIZE
    if remainder != 0:
        pad = Q8_BLOCK_SIZE - remainder
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
        n = len(data)

    n_blocks = n // Q8_BLOCK_SIZE
    output = bytearray(n_blocks * Q8_BLOCK_BYTES)

    for b in range(n_blocks):
        block = data[b * Q8_BLOCK_SIZE : (b + 1) * Q8_BLOCK_SIZE]
        amax = float(np.max(np.abs(block)))
        d = amax / 127.0
        inv_d = 1.0 / d if d != 0.0 else 0.0

        # Scale as f16 LE
        d_f16 = np.float16(d)
        offset = b * Q8_BLOCK_BYTES
        struct.pack_into("<e", output, offset, float(d_f16))

        # Quantize to signed int8
        for i in range(Q8_BLOCK_SIZE):
            v = float(block[i])
            q = int(round(v * inv_d))
            q = max(-128, min(127, q))
            # Store as unsigned byte (two's complement)
            output[offset + 2 + i] = q & 0xFF

    return bytes(output)


def q8_byte_size(num_elements: int) -> int:
    """Compute Q8_0 byte size for a given element count (after padding to 32)."""
    n = num_elements
    remainder = n % Q8_BLOCK_SIZE
    if remainder != 0:
        n += Q8_BLOCK_SIZE - remainder
    return (n // Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES


# ---------------------------------------------------------------------------
# GGUF v3 writer helpers
# ---------------------------------------------------------------------------

def write_gguf_string(buf: bytearray, s: str) -> None:
    """Write a GGUF string: u64 length + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def write_string_kv(buf: bytearray, key: str, value: str) -> None:
    """Write a string-type metadata KV pair."""
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", 8))  # value_type = STRING
    write_gguf_string(buf, value)


def align_offset(offset: int) -> int:
    """Round up to next 32-byte boundary."""
    return ((offset + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def load_and_prepare_tensors(
    model_dir: Path,
    quant_type: str,
) -> list[tuple[str, int, np.ndarray, list[int]]]:
    """Load SafeTensors, fuse weight norms, decide dtype for each tensor.

    Returns list of (name, ggml_dtype, data_bytes_as_ndarray_or_bytes, shape).
    """
    st_path = model_dir / "consolidated.safetensors"
    if not st_path.exists():
        print(f"Error: {st_path} not found", file=sys.stderr)
        sys.exit(1)

    # Select quantization parameters
    if quant_type == "q4_0":
        quantize_fn = quantize_q4_0
        dtype_code = DTYPE_Q4_0
        block_size = Q4_BLOCK_SIZE
        quant_label = "Q4_0"
    elif quant_type == "q8_0":
        quantize_fn = quantize_q8_0
        dtype_code = DTYPE_Q8_0
        block_size = Q8_BLOCK_SIZE
        quant_label = "Q8_0"
    else:
        print(f"Error: unknown quant type '{quant_type}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {st_path} ...")
    print(f"Quantization: {quant_label}")
    state_dict = load_file(str(st_path), device="cpu")

    # -----------------------------------------------------------------------
    # Phase 1: Fuse weight norms (codec convolutions)
    # -----------------------------------------------------------------------
    # Collect (g, v) pairs by their clean prefix.
    wn_g: dict[str, torch.Tensor] = {}
    wn_v: dict[str, torch.Tensor] = {}
    regular_keys: list[str] = []

    for name in state_dict:
        if name.endswith(WEIGHT_NORM_G_SUFFIX):
            prefix = name[: -len(WEIGHT_NORM_G_SUFFIX)]
            wn_g[prefix] = state_dict[name]
        elif name.endswith(WEIGHT_NORM_V_SUFFIX):
            prefix = name[: -len(WEIGHT_NORM_V_SUFFIX)]
            wn_v[prefix] = state_dict[name]
        else:
            regular_keys.append(name)

    # Fuse and add as clean names.
    fused: dict[str, torch.Tensor] = {}
    for prefix in sorted(wn_g.keys()):
        if prefix not in wn_v:
            print(f"  WARNING: weight norm g without v for {prefix}", file=sys.stderr)
            continue
        g = wn_g[prefix]
        v = wn_v[prefix]
        fused_w = fuse_weight_norm(g, v)
        clean_name = prefix + ".weight"
        fused[clean_name] = fused_w
        print(f"  fused weight norm: {clean_name} {list(fused_w.shape)}")

    # -----------------------------------------------------------------------
    # Phase 2: Build output tensor list
    # -----------------------------------------------------------------------
    results: list[tuple[str, int, bytes, list[int]]] = []

    # Process regular tensors.
    for name in sorted(regular_keys):
        tensor = state_dict[name]
        shape = list(tensor.shape)

        # Convert BF16 → F32 (numpy doesn't support BF16).
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        arr = tensor.numpy()

        if should_quantize(name):
            data = quantize_fn(arr)
            # Adjust shape if padding was needed.
            n_elem = int(np.prod(shape))
            if n_elem % block_size != 0:
                # Pad last dimension to make total elements divisible by block_size.
                pad_needed = block_size - (n_elem % block_size)
                shape[-1] += pad_needed
            results.append((name, dtype_code, data, shape))
        else:
            data = arr.astype(np.float32).tobytes()
            results.append((name, DTYPE_F32, data, shape))

    # Process fused weight-norm tensors.
    for name in sorted(fused.keys()):
        tensor = fused[name]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        arr = tensor.numpy().astype(np.float32)
        data = arr.tobytes()
        shape = list(arr.shape)
        results.append((name, DTYPE_F32, data, shape))

    return results


def write_gguf(
    tensors: list[tuple[str, int, bytes, list[int]]],
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """Write GGUF v3 file from prepared tensors."""
    # Print summary table.
    dtype_names = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q4_0: "Q4_0", DTYPE_Q8_0: "Q8_0"}
    total_data = 0
    print(f"\n{'Tensor':<70} {'Dtype':<6} {'Shape':<25} {'Size':>12}")
    print("-" * 115)
    for name, dtype, data, shape in tensors:
        size = len(data)
        total_data += size
        shape_str = str(shape)
        print(f"{name:<70} {dtype_names[dtype]:<6} {shape_str:<25} {size:>12,}")
    print("-" * 115)
    print(f"{'Total tensors:':<70} {len(tensors):<6} {'':25} {total_data:>12,}")

    if dry_run:
        print("\n[dry-run] Would write GGUF file, skipping.")
        return

    # -----------------------------------------------------------------------
    # Build GGUF binary
    # -----------------------------------------------------------------------
    header = bytearray()

    # Header: magic, version, tensor_count, metadata_kv_count
    header.extend(struct.pack("<I", GGUF_MAGIC))
    header.extend(struct.pack("<I", GGUF_VERSION))
    header.extend(struct.pack("<Q", len(tensors)))
    header.extend(struct.pack("<Q", 1))  # 1 metadata KV

    # Metadata KV: general.architecture = "voxtral-tts"
    write_string_kv(header, "general.architecture", "voxtral-tts")

    # Tensor index
    # First pass: compute data offsets (relative to start of data section).
    data_offset = 0
    tensor_offsets: list[int] = []
    for _name, dtype, data, _shape in tensors:
        # Each tensor is aligned to 32 bytes within the data section.
        data_offset = align_offset(data_offset)
        tensor_offsets.append(data_offset)
        data_offset += len(data)

    # Write tensor descriptors.
    for i, (name, dtype, _data, shape) in enumerate(tensors):
        write_gguf_string(header, name)
        # n_dimensions
        header.extend(struct.pack("<I", len(shape)))
        # Dimensions — REVERSED from PyTorch convention for GGUF
        for dim in reversed(shape):
            header.extend(struct.pack("<Q", dim))
        # dtype
        header.extend(struct.pack("<I", dtype))
        # offset (relative to data section start)
        header.extend(struct.pack("<Q", tensor_offsets[i]))

    # Alignment padding between header+index and data section.
    header_len = len(header)
    padded_header_len = align_offset(header_len)
    header.extend(b"\x00" * (padded_header_len - header_len))

    # Write file.
    print(f"\nWriting {output_path} ...")
    with open(output_path, "wb") as f:
        f.write(header)
        for i, (_name, _dtype, data, _shape) in enumerate(tensors):
            # Seek to aligned position (relative to data section start + header).
            target_pos = padded_header_len + tensor_offsets[i]
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b"\x00" * (target_pos - current_pos))
            f.write(data)

    file_size = output_path.stat().st_size
    print(f"Done! File size: {file_size:,} bytes ({file_size / (1024**3):.2f} GiB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral 4B TTS to GGUF v3 with Q4_0 or Q8_0"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing consolidated.safetensors",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GGUF path (default: voxtral-tts-{quant_type}.gguf)",
    )
    parser.add_argument(
        "--quant-type",
        choices=["q4_0", "q8_0"],
        default="q8_0",
        help="Quantization type (default: q8_0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tensor list without writing",
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"Error: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = Path(f"voxtral-tts-{args.quant_type.replace('_', '')}.gguf")

    tensors = load_and_prepare_tensors(args.model_dir, args.quant_type)
    write_gguf(tensors, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
