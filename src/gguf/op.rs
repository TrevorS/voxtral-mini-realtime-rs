//! Fused Q4_0 dequant+matmul GPU kernel launch.
//!
//! [`q4_matmul`] launches the WGSL compute shader at `shader.wgsl` to perform
//! `output[B, M, N] = input[B, M, K] × weights[N, K]^T` where weights are in
//! Q4_0 format with shape `[N, K]` (out_features, in_features).

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

/// Workgroup size for the N (output column) dimension.
/// Each thread computes one output element; the workgroup cooperatively
/// loads the input tile into shared memory.
const WG_X: u32 = 64;

/// WGSL kernel source for Q4_0 dequant+matmul with shared memory tiling.
struct Q4MatmulKernel {
    workgroup_size_x: u32,
}

impl KernelSource for Q4MatmulKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("shader.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x)
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

    // Info buffer: [B, M, K, N, blocks_per_row]
    let blocks_per_row = k / 32;
    let info: [u32; 5] = [b as u32, m as u32, k as u32, n as u32, blocks_per_row as u32];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    // Allocate output buffer (B × M × N × 4 bytes for f32)
    let output_handle = client.empty(b * m * n * 4);

    // Create kernel — workgroup_size(WG_X, 1, 1); Y=1 because each row
    // of the output (each bm value) maps to separate workgroups.
    let kernel = SourceKernel::new(
        Q4MatmulKernel {
            workgroup_size_x: WG_X,
        },
        CubeDim::new_1d(WG_X),
    );

    // Bindings must match WGSL @binding order:
    //   @binding(0) = weights (array<u32>, Q4_0 blocks as raw bytes)
    //   @binding(1) = input   (array<f32>, activations)
    //   @binding(2) = output  (array<f32>, result)
    //   @binding(3) = info    (array<u32>, metadata)
    let bindings = Bindings::new()
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    // Workgroup dispatch: x=ceil(N/WG_X), y=B*M (one row per workgroup in Y)
    let wg_count_x = n.div_ceil(WG_X as usize) as u32;
    let wg_count_y = (b * m) as u32;

    // Launch
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_count_x, wg_count_y),
            bindings,
        )
        .expect("Q4 matmul kernel launch failed");

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
