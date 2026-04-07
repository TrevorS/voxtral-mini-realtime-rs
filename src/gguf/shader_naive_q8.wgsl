// Q8_0 Dequantization + Matrix Multiplication (naive variant, one thread per element)
// Q8_0: 34 bytes/block = 2 (f16 scale) + 32 (int8 values)

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;
const Q8_BLOCK_BYTES: u32 = 34u;

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

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0]; let M = info[1]; let K = info[2]; let N = info[3];
    let blocks_per_row = info[4];
    let n = gid.x; let bm = gid.y; let m = bm % M; let b = bm / M;
    if (n >= N || b >= B) { return; }
    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;
    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let global_block = n * blocks_per_row + blk;
        let block_byte = global_block * Q8_BLOCK_BYTES;
        let scale = read_f16_scale(block_byte);
        let k_base = blk * 32u;
        let data_start = block_byte + 2u;
        for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
            let packed = read_u32_unaligned(data_start + wi * 4u);
            let base_i = wi * 4u;
            let k_off = input_base + k_base;
            let w = vec4<f32>(extract_i8(packed,0u),extract_i8(packed,1u),extract_i8(packed,2u),extract_i8(packed,3u)) * scale;
            let inp = vec4<f32>(input[k_off+base_i],input[k_off+base_i+1u],input[k_off+base_i+2u],input[k_off+base_i+3u]);
            acc += dot(w, inp);
        }
    }
    output[b * M * N + m * N + n] = acc;
}
