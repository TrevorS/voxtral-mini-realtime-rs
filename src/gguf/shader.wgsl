// Q4_0 Dequantization + Matrix Multiplication Compute Shader
//
// Performs a fused dequant-matmul for Q4_0 quantized weight tensors on GPU.
// Computes: output[B, M, N] = input[B, M, K] × weights[N, K]^T
// where weights are stored in GGML Q4_0 block format.
//
// == Q4_0 Block Format (GGML standard, interleaved) ==
//
// Each block encodes 32 weights into 18 bytes:
//
//   Bytes 0-1:  f16 scale `d` — rescales the quantized ints
//   Bytes 2-17: 16 bytes of packed 4-bit quantized values
//
// Nibble packing within the 16 data bytes:
//   - Each byte holds two 4-bit values (nibbles)
//   - Lower nibble (bits 0-3) → elements 0-15
//   - Upper nibble (bits 4-7) → elements 16-31
//   - Each nibble stores (value + 8) & 0xF, so dequantized value = (nibble - 8) * d
//
// The weight tensor [N, K] (out_features, in_features) is flattened row-major
// and divided into consecutive blocks of 32 elements. Weight at position (n, k)
// has flat index (n * K + k), which falls in block (flat_index / 32) at element
// (flat_index % 32). This matches PyTorch/GGUF convention where each row is one
// output neuron's weights, giving contiguous memory access per output thread.
//
// == Memory Layout ==
//
// The raw Q4_0 bytes are uploaded as-is and bound as array<u32>. Since blocks
// are 18 bytes (not a multiple of 4), we use byte-level addressing within the
// u32 array. Block `b` starts at byte offset `b * 18`.

// -- Bindings --
// All bindings use read_write to avoid wgpu validation errors when cubecl's
// memory sub-allocator places multiple bindings in the same underlying buffer.
// This matches burn-cubecl's internal approach of marking all bindings ReadWrite.
//
// weights: raw Q4_0 bytes viewed as u32 array. Byte `i` is in weights[i/4],
// shifted by (i%4)*8 bits.
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
// input: f32 activation tensor, shape [B, M, K], row-major
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
// output: f32 result tensor, shape [B, M, N], row-major
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// info: dimension metadata [B, M, K, N, num_blocks_per_row]
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

// ---------------------------------------------------------------------------
// read_byte: Read a single byte from the weights buffer at the given byte offset.
// ---------------------------------------------------------------------------
fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    return (weights[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// ---------------------------------------------------------------------------
// read_f16_scale: Read the f16 scale factor at the start of a Q4_0 block.
// Returns the scale as f32.
// ---------------------------------------------------------------------------
fn read_f16_scale(block_byte_offset: u32) -> f32 {
    // f16 occupies 2 bytes in little-endian order
    let lo = read_byte(block_byte_offset);
    let hi = read_byte(block_byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

// ---------------------------------------------------------------------------
// dequant: Reconstruct a single f32 weight from Q4_0 packed representation.
//
// `global_flat_idx` is the position of the weight in the flattened [N, K] tensor.
// ---------------------------------------------------------------------------
fn dequant(global_flat_idx: u32) -> f32 {
    let block_idx = global_flat_idx / 32u;
    let elem_idx  = global_flat_idx % 32u;

    // Each block is 18 bytes: 2 bytes scale + 16 bytes nibbles
    let block_start = block_idx * 18u;

    // Read the f16 scale factor
    let d = read_f16_scale(block_start);

    // Map element index to the byte containing its nibble.
    // Elements 0-15 use the lower nibble, elements 16-31 use the upper nibble
    // of the same byte position.
    let local = elem_idx % 16u;

    // Nibble data starts at byte offset 2 within the block
    let data_byte = read_byte(block_start + 2u + local);

    // Lower nibble (bits 0-3) for elements 0-15,
    // upper nibble (bits 4-7) for elements 16-31
    let nibble = select(data_byte & 0xFu, (data_byte >> 4u) & 0xFu, elem_idx >= 16u);

    // Dequantize: nibbles store (value + 8), so subtract 8 and scale.
    return (f32(nibble) - 8.0) * d;
}

// ---------------------------------------------------------------------------
// Main kernel entry point.
//
// Thread mapping (naive: one thread per output element):
//   gid.x  → n  (column in output / weight matrix)
//   gid.y  → flattened (b * M + m), i.e. batch and row combined
//
// Each thread accumulates the dot product over the K dimension:
//   output[b, m, n] = Σ_k  input[b, m, k] * dequant(weights[n, k])
// ---------------------------------------------------------------------------
@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];
    // info[4] = num_blocks_per_row (reserved for future tiled kernels)

    let n = gid.x;
    let bm = gid.y;   // flattened batch * M + row
    let m = bm % M;
    let b = bm / M;

    // Early exit for out-of-bounds threads (workgroup may overshoot dimensions).
    if (n >= N || b >= B) {
        return;
    }

    var acc: f32 = 0.0;

    // Base offset into the input tensor for this (b, m) slice.
    let input_base = b * M * K + m * K;

    // Accumulate dot product over the K (inner) dimension.
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        // Weight position in the flattened [N, K] tensor (row n, column k).
        let weight_flat = n * K + k;
        acc = acc + dequant(weight_flat) * input[input_base + k];
    }

    // Store result at output[b, m, n].
    output[b * M * N + m * N + n] = acc;
}
