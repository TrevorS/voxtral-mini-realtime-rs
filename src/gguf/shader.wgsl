// Q4_0 Dequantization + Matrix Multiplication Compute Shader
//
// Performs a fused dequant-matmul for Q4_0 quantized weight tensors on GPU.
// Computes: output[B, M, N] = input[B, M, K] × weights[N, K]^T
// where weights are stored in GGML Q4_0 block format.
//
// == Optimization strategy ==
//
// Uses workgroup shared memory to tile the K (inner) dimension:
//   1. Threads cooperatively load a TILE_K-sized slice of the input row
//   2. Each thread accumulates against its own weight row using shared input
//   3. Weight reads use vectorized u32 loads (4 bytes at once vs byte-by-byte)
//   4. Accumulation uses vec4 dot products for SIMD efficiency
//
// For M=1 (decode), this eliminates redundant global reads: the input vector
// is loaded once into shared memory and reused by all threads in the workgroup.
//
// == Q4_0 Block Format (GGML standard, interleaved) ==
//
// Each block encodes 32 weights into 18 bytes:
//   Bytes 0-1:  f16 scale `d`
//   Bytes 2-17: 16 bytes of packed 4-bit quantized values
//   Lower nibble (bits 0-3) → elements 0-15
//   Upper nibble (bits 4-7) → elements 16-31
//   Dequantized value = (nibble - 8) * d

// -- Dimensions (templated as compile-time constants) --
// Using constants instead of a storage buffer satisfies Chrome's WGSL
// uniform control flow requirement for workgroupBarrier().
const B: u32 = {{ dim_b }}u;
const M: u32 = {{ dim_m }}u;
const K: u32 = {{ dim_k }}u;
const N: u32 = {{ dim_n }}u;
const blocks_per_row: u32 = {{ blocks_per_row }}u;

// -- Bindings (no info buffer needed — dimensions are constants) --
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Tile size for K-dimension shared memory. Must be a multiple of 32 (Q4 block size).
// 512 = 16 Q4 blocks per tile. Halves barrier syncs vs 256.
// All model K values (1280, 3072, 5120, 9216) are multiples of 32.
const TILE_K: u32 = 512u;

// Shared memory for the input vector tile.
var<workgroup> shared_input: array<f32, 512>;  // TILE_K elements

// ---------------------------------------------------------------------------
// read_u32_unaligned: Read a u32 from the weights buffer at an arbitrary byte
// offset, handling misalignment by combining two adjacent u32 words.
// ---------------------------------------------------------------------------
fn read_u32_unaligned(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    if (shift == 0u) {
        return weights[word];
    }
    return (weights[word] >> shift) | (weights[word + 1u] << (32u - shift));
}

// ---------------------------------------------------------------------------
// read_f16_scale: Read the f16 scale factor at the start of a Q4_0 block.
// ---------------------------------------------------------------------------
fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(block_byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

// ---------------------------------------------------------------------------
// Main kernel entry point.
//
// Thread mapping:
//   gid.x → n  (output column, one thread per output element in N)
//   gid.y → bm (flattened b * M + m, one workgroup row per output row)
// ---------------------------------------------------------------------------
@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = gid.x;
    let bm = gid.y;
    let m = bm % M;
    let b = bm / M;

    // No early return — all threads must reach workgroupBarrier() to satisfy
    // Chrome's WGSL uniform control flow requirement. Out-of-bounds threads
    // participate in barriers but skip computation and output writes.
    let valid = b < B;

    var acc: f32 = 0.0;
    let input_base = select(0u, b * M * K + m * K, valid);
    let wg_size = {{ workgroup_size_x }}u;
    let num_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * TILE_K;

        // -- Cooperative load: all threads load input tile into shared memory --
        for (var k_local: u32 = lid.x; k_local < TILE_K; k_local = k_local + wg_size) {
            let k_global = tile_start + k_local;
            if (valid && k_global < K) {
                shared_input[k_local] = input[input_base + k_global];
            }
        }
        workgroupBarrier();

        // -- Vectorized dequant+accumulate against shared input --
        if (valid && n < N) {
            let tile_end = min(tile_start + TILE_K, K);
            let blocks_in_tile = (tile_end - tile_start) / 32u;
            let block_base = tile_start / 32u;

            for (var blk: u32 = 0u; blk < blocks_in_tile; blk = blk + 1u) {
                let global_block = n * blocks_per_row + block_base + blk;
                let block_byte = global_block * 18u;
                let scale = read_f16_scale(block_byte);
                let k_base = blk * 32u;

                // Read 16 data bytes as 4 u32 words, process with vec4 dot products.
                // Each u32 holds 4 packed bytes; each byte has lo nibble (elem 0-15)
                // and hi nibble (elem 16-31).
                let data_start = block_byte + 2u;
                for (var wi: u32 = 0u; wi < 4u; wi = wi + 1u) {
                    let packed = read_u32_unaligned(data_start + wi * 4u);
                    let b0 = packed & 0xFFu;
                    let b1 = (packed >> 8u) & 0xFFu;
                    let b2 = (packed >> 16u) & 0xFFu;
                    let b3 = (packed >> 24u) & 0xFFu;

                    let base_i = wi * 4u;

                    // Lower nibbles → elements [base_i..base_i+3]
                    let w_lo = (vec4<f32>(
                        f32(b0 & 0xFu), f32(b1 & 0xFu),
                        f32(b2 & 0xFu), f32(b3 & 0xFu)
                    ) - vec4<f32>(8.0)) * scale;
                    let in_lo = vec4<f32>(
                        shared_input[k_base + base_i],
                        shared_input[k_base + base_i + 1u],
                        shared_input[k_base + base_i + 2u],
                        shared_input[k_base + base_i + 3u]
                    );
                    acc += dot(w_lo, in_lo);

                    // Upper nibbles → elements [16+base_i..16+base_i+3]
                    let w_hi = (vec4<f32>(
                        f32((b0 >> 4u) & 0xFu), f32((b1 >> 4u) & 0xFu),
                        f32((b2 >> 4u) & 0xFu), f32((b3 >> 4u) & 0xFu)
                    ) - vec4<f32>(8.0)) * scale;
                    let in_hi = vec4<f32>(
                        shared_input[k_base + 16u + base_i],
                        shared_input[k_base + 16u + base_i + 1u],
                        shared_input[k_base + 16u + base_i + 2u],
                        shared_input[k_base + 16u + base_i + 3u]
                    );
                    acc += dot(w_hi, in_hi);
                }
            }
        }
        workgroupBarrier();
    }

    if (n < N && b < B) {
        output[b * M * N + m * N + n] = acc;
    }
}
