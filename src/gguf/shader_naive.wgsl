// Q4_0 Dequantization + Matrix Multiplication Compute Shader (naive variant)
//
// One-thread-per-output-element kernel, no shared memory.
// Better than the tiled variant for M > 1 (prefill, encoder) where
// each row is independent and the 2D workgroup layout fills the GPU well.
//
// Uses block-level iteration with vectorized u32 reads and vec4 dot products
// (same approach as the tiled kernel, but without shared memory tiling).
//
// Computes: output[B, M, N] = input[B, M, K] × weights[N, K]^T

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

fn read_u32_unaligned(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    if (shift == 0u) {
        return weights[word];
    }
    return (weights[word] >> shift) | (weights[word + 1u] << (32u - shift));
}

fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(block_byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4];

    let n = gid.x;
    let bm = gid.y;
    let m = bm % M;
    let b = bm / M;

    if (n >= N || b >= B) {
        return;
    }

    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;

    // Block-level iteration: read scale once per 32 elements, vectorized dequant
    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let global_block = n * blocks_per_row + blk;
        let block_byte = global_block * 18u;
        let scale = read_f16_scale(block_byte);
        let k_base = blk * 32u;

        // Read 16 data bytes as 4 u32 words, process with vec4 dot products
        let data_start = block_byte + 2u;
        for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
            let packed = read_u32_unaligned(data_start + wi * 4u);
            let b0 = packed & 0xFFu;
            let b1 = (packed >> 8u) & 0xFFu;
            let b2 = (packed >> 16u) & 0xFFu;
            let b3 = (packed >> 24u) & 0xFFu;

            let base_i = wi * 4u;
            let k_off = input_base + k_base;

            // Lower nibbles → elements [base_i..base_i+3]
            let w_lo = (vec4<f32>(
                f32(b0 & 0xFu), f32(b1 & 0xFu),
                f32(b2 & 0xFu), f32(b3 & 0xFu)
            ) - vec4<f32>(8.0)) * scale;
            let in_lo = vec4<f32>(
                input[k_off + base_i],
                input[k_off + base_i + 1u],
                input[k_off + base_i + 2u],
                input[k_off + base_i + 3u]
            );
            acc += dot(w_lo, in_lo);

            // Upper nibbles → elements [16+base_i..16+base_i+3]
            let w_hi = (vec4<f32>(
                f32((b0 >> 4u) & 0xFu), f32((b1 >> 4u) & 0xFu),
                f32((b2 >> 4u) & 0xFu), f32((b3 >> 4u) & 0xFu)
            ) - vec4<f32>(8.0)) * scale;
            let in_hi = vec4<f32>(
                input[k_off + 16u + base_i],
                input[k_off + 16u + base_i + 1u],
                input[k_off + 16u + base_i + 2u],
                input[k_off + 16u + base_i + 3u]
            );
            acc += dot(w_hi, in_hi);
        }
    }

    output[b * M * N + m * N + n] = acc;
}
