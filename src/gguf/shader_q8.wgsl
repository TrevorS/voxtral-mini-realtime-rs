// Q8_0 Dequantization + Matrix Multiplication Compute Shader (tiled variant)
//
// Q8_0 block layout (34 bytes per 32 elements):
//   - bytes 0-1: f16 scale
//   - bytes 2-33: 32 signed int8 quantized values
// Dequantization: value = scale * i8_value

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

const TILE_K: u32 = 512u;
const Q8_BLOCK_BYTES: u32 = 34u;
var<workgroup> shared_input: array<f32, 512>;

fn read_u32_unaligned(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    if (shift == 0u) { return weights[word]; }
    return (weights[word] >> shift) | (weights[word + 1u] << (32u - shift));
}
fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(block_byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}
fn extract_i8(word: u32, byte_idx: u32) -> f32 {
    let raw = (word >> (byte_idx * 8u)) & 0xFFu;
    return f32(i32(raw) - select(0, 256, raw >= 128u));
}

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let B = info[0]; let M = info[1]; let K = info[2]; let N = info[3];
    let blocks_per_row = info[4];
    let n = gid.x; let bm = gid.y; let m = bm % M; let b = bm / M;
    let valid = b < B;
    var acc: f32 = 0.0;
    let input_base = select(0u, b * M * K + m * K, valid);
    let wg_size = {{ workgroup_size_x }}u;
    let num_tiles = (K + TILE_K - 1u) / TILE_K;

    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * TILE_K;
        for (var k_local: u32 = lid.x; k_local < TILE_K; k_local = k_local + wg_size) {
            let k_global = tile_start + k_local;
            if (valid && k_global < K) { shared_input[k_local] = input[input_base + k_global]; }
        }
        workgroupBarrier();
        if (valid && n < N) {
            let tile_end = min(tile_start + TILE_K, K);
            let blocks_in_tile = (tile_end - tile_start) / 32u;
            let block_base = tile_start / 32u;
            for (var blk: u32 = 0u; blk < blocks_in_tile; blk = blk + 1u) {
                let global_block = n * blocks_per_row + block_base + blk;
                let block_byte = global_block * Q8_BLOCK_BYTES;
                let scale = read_f16_scale(block_byte);
                let k_base = blk * 32u;
                let data_start = block_byte + 2u;
                for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
                    let packed = read_u32_unaligned(data_start + wi * 4u);
                    let base_i = wi * 4u;
                    let w = vec4<f32>(extract_i8(packed,0u),extract_i8(packed,1u),extract_i8(packed,2u),extract_i8(packed,3u)) * scale;
                    let inp = vec4<f32>(shared_input[k_base+base_i],shared_input[k_base+base_i+1u],shared_input[k_base+base_i+2u],shared_input[k_base+base_i+3u]);
                    acc += dot(w, inp);
                }
            }
        }
        workgroupBarrier();
    }
    if (n < N && b < B) { output[b * M * N + m * N + n] = acc; }
}
