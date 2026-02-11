//! Fused Q4_0 dequant+matmul GPU kernel launch.
//!
//! [`q4_matmul`] launches a WGSL compute shader to perform
//! `output[B, M, N] = input[B, M, K] × weights[N, K]^T` where weights are in
//! Q4_0 format with shape `[N, K]` (out_features, in_features).
//!
//! Two kernel variants are dispatched based on M:
//! - **M ≤ threshold**: Tiled kernel with shared memory (shader.wgsl).
//!   Cooperatively loads the input vector, eliminating redundant global reads.
//!   Dimensions are baked as compile-time constants (not read from a storage
//!   buffer) so that `workgroupBarrier()` satisfies Chrome's WGSL uniform
//!   control flow requirement.
//! - **M > threshold**: Naive kernel (shader_naive.wgsl).
//!   One thread per output element with (16,16) workgroups — better for
//!   multi-row matmuls where the 2D layout fills the GPU efficiently.
//!
//! On WASM/WebGPU, only the naive kernel is used because of a CubeCL bind
//! group layout issue: switching between 3-binding (tiled) and 4-binding
//! (naive) shaders within the same session produces incorrect results.

use burn::backend::wgpu::{
    into_contiguous, AutoCompiler, CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate,
    WgpuRuntime,
};
use burn::backend::Wgpu;
use burn::tensor::{DType, Tensor, TensorPrimitive};
use cubecl::prelude::KernelId;
use cubecl::server::{Bindings, CubeCount};
use cubecl::CubeTask;

use super::tensor::Q4Tensor;

/// M threshold: use tiled kernel when M ≤ this, naive kernel otherwise.
/// On WASM, always use naive to avoid a CubeCL bind group layout issue.
#[cfg(target_family = "wasm")]
const TILED_M_THRESHOLD: usize = 0;
#[cfg(not(target_family = "wasm"))]
const TILED_M_THRESHOLD: usize = 4;

// -- Tiled kernel (shared memory, good for M=1 decode) --

const TILED_WG_X: u32 = 128;

struct Q4MatmulTiledKernel {
    workgroup_size_x: u32,
    dim_b: u32,
    dim_m: u32,
    dim_k: u32,
    dim_n: u32,
    blocks_per_row: u32,
}

impl KernelSource for Q4MatmulTiledKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("shader.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("dim_b", self.dim_b.to_string())
            .register("dim_m", self.dim_m.to_string())
            .register("dim_k", self.dim_k.to_string())
            .register("dim_n", self.dim_n.to_string())
            .register("blocks_per_row", self.blocks_per_row.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.workgroup_size_x)
            .info(self.dim_b)
            .info(self.dim_m)
            .info(self.dim_k)
            .info(self.dim_n)
    }
}

// -- Naive kernel (one thread per element, good for M>1 prefill/encoder) --

const NAIVE_WG_X: u32 = 16;
const NAIVE_WG_Y: u32 = 16;

struct Q4MatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q4MatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("shader_naive.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q4_0 dequant+matmul on GPU.
///
/// Computes `output[B, M, N] = input[B, M, K] × weights[N, K]^T` where weights
/// are stored in Q4_0 block format on the GPU with shape `[N, K]`
/// (out_features, in_features), matching PyTorch/GGUF convention.
/// Dequantization happens inside the compute shader — no intermediate
/// full-precision weight buffer is created.
pub fn q4_matmul(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    // Convert Tensor → CubeTensor and ensure contiguous layout
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    // Extract dimensions
    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    // Allocate output buffer (B × M × N × 4 bytes for f32)
    let output_handle = client.empty(b * m * n * 4);

    // Dispatch: tiled kernel for small M (decode), naive for large M (prefill/encoder)
    if m <= TILED_M_THRESHOLD {
        // Tiled kernel: dimensions baked as shader constants, no info buffer needed.
        let bindings = Bindings::new()
            .with_buffer(weights.handle.clone().binding())
            .with_buffer(cube_input.handle.clone().binding())
            .with_buffer(output_handle.clone().binding());

        let kernel = SourceKernel::new(
            Q4MatmulTiledKernel {
                workgroup_size_x: TILED_WG_X,
                dim_b: b as u32,
                dim_m: m as u32,
                dim_k: k as u32,
                dim_n: n as u32,
                blocks_per_row: blocks_per_row as u32,
            },
            CubeDim::new_1d(TILED_WG_X),
        );
        let wg_x = n.div_ceil(TILED_WG_X as usize) as u32;
        let wg_y = (b * m) as u32;
        client
            .launch(
                Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
                CubeCount::new_2d(wg_x, wg_y),
                bindings,
            )
            .expect("Q4 tiled matmul kernel launch failed");
    } else {
        // Naive kernel: dimensions read from info buffer (no barriers, no UCF issue).
        let info: [u32; 5] = [
            b as u32,
            m as u32,
            k as u32,
            n as u32,
            blocks_per_row as u32,
        ];
        let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
        let info_handle = client.create_from_slice(&info_bytes);

        let bindings = Bindings::new()
            .with_buffer(weights.handle.clone().binding())
            .with_buffer(cube_input.handle.clone().binding())
            .with_buffer(output_handle.clone().binding())
            .with_buffer(info_handle.binding());

        let kernel = SourceKernel::new(
            Q4MatmulNaiveKernel {
                workgroup_size_x: NAIVE_WG_X,
                workgroup_size_y: NAIVE_WG_Y,
            },
            CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
        );
        let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
        let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
        client
            .launch(
                Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
                CubeCount::new_2d(wg_x, wg_y),
                bindings,
            )
            .expect("Q4 naive matmul kernel launch failed");
    }

    // Wrap output handle in a CubeTensor → Tensor
    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}
