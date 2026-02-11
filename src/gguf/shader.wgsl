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
//   3. Weight reads use block-level dequantization (scale read once per 32 elems)
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
//
// == Memory Layout ==
//
// Raw Q4_0 bytes uploaded as-is, bound as array<u32>. Blocks are 18 bytes
// (not a multiple of 4), so we use byte-level addressing via the u32 array.

// -- Bindings --
@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

// Tile size for K-dimension shared memory. Must be a multiple of 32 (Q4 block size).
// 256 = 8 Q4 blocks per tile. All model K values (1280, 3072, 5120, 9216) are multiples.
const TILE_K: u32 = 256u;

// Shared memory for the input vector tile.
var<workgroup> shared_input: array<f32, 256>;  // TILE_K elements

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
// ---------------------------------------------------------------------------
fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let lo = read_byte(block_byte_offset);
    let hi = read_byte(block_byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

// ---------------------------------------------------------------------------
// Main kernel entry point.
//
// Thread mapping:
//   gid.x → n  (output column, one thread per output element in N)
//   gid.y → bm (flattened b * M + m, one workgroup row per output row)
//
// Each thread accumulates the full dot product over K, reading the input
// from shared memory. K is processed in tiles of TILE_K (256 elements =
// 8 Q4 blocks).
// ---------------------------------------------------------------------------
@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4];

    let n = gid.x;
    let bm = gid.y;
    let m = bm % M;
    let b = bm / M;

    if (b >= B) {
        return;
    }

    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;
    let wg_size = {{ workgroup_size_x }}u;
    let num_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * TILE_K;

        // -- Cooperative load: all threads in workgroup load the input tile --
        // Each thread loads ceil(TILE_K / wg_size) elements in an interleaved pattern.
        for (var k_local: u32 = lid.x; k_local < TILE_K; k_local = k_local + wg_size) {
            let k_global = tile_start + k_local;
            if (k_global < K) {
                shared_input[k_local] = input[input_base + k_global];
            }
        }
        workgroupBarrier();

        // -- Block-level dequant+accumulate against shared input --
        if (n < N) {
            // Number of complete Q4 blocks in this tile
            let tile_end = min(tile_start + TILE_K, K);
            let blocks_in_tile = (tile_end - tile_start) / 32u;
            let block_base = tile_start / 32u;  // first Q4 block index within this row

            for (var blk: u32 = 0u; blk < blocks_in_tile; blk = blk + 1u) {
                let global_block = n * blocks_per_row + block_base + blk;
                let block_byte = global_block * 18u;
                let scale = read_f16_scale(block_byte);
                let k_base = blk * 32u;  // offset within shared_input for this block

                // Process 16 data bytes → 32 dequantized weights.
                // Each byte contains two nibbles: lower → elements 0-15,
                // upper → elements 16-31.
                for (var i: u32 = 0u; i < 16u; i = i + 1u) {
                    let byte_val = read_byte(block_byte + 2u + i);
                    let lo = byte_val & 0xFu;
                    let hi = (byte_val >> 4u) & 0xFu;
                    let w_lo = (f32(lo) - 8.0) * scale;
                    let w_hi = (f32(hi) - 8.0) * scale;
                    acc = acc + w_lo * shared_input[k_base + i]
                              + w_hi * shared_input[k_base + i + 16u];
                }
            }
        }
        workgroupBarrier();
    }

    if (n < N && b < B) {
        output[b * M * N + m * N + n] = acc;
    }
}
