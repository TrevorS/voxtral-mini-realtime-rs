//! Q4_0 quantized weight tensor stored on GPU.
//!
//! [`Q4Tensor`] uploads raw Q4_0 blocks to a GPU storage buffer and provides
//! a [`dequantize`](Q4Tensor::dequantize) method for diagnostics/testing.
//! The primary inference path is [`q4_matmul`](super::op::q4_matmul), which
//! dequantizes on-the-fly inside a fused compute shader.

use anyhow::{ensure, Result};
use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::Runtime;

/// A Q4_0 quantized weight tensor living on GPU.
///
/// The buffer contains raw Q4_0 blocks (18 bytes per block of 32 elements),
/// laid out exactly as in GGUF. The WGSL shader interprets the buffer as
/// `array<f16>` with 9 f16 slots per block.
pub struct Q4Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
    client: ComputeClient<WgpuRuntime>,
    device: WgpuDevice,
}

impl Q4Tensor {
    /// Upload raw Q4_0 bytes to a GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`, matching PyTorch/GGUF
    /// convention. `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes.
    /// The element count `N * K` must be divisible by 32.
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
        // Q4_0 blocks are 18 bytes, so total size may not be a multiple of 4.
        let padded = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            buf
        } else {
            raw_bytes.to_vec()
        };
        let handle = client.create_from_slice(&padded);

        Ok(Self {
            handle,
            shape,
            num_blocks,
            client,
            device: device.clone(),
        })
    }

    /// Logical weight dimensions `[N, K]` = `[out_features, in_features]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Number of Q4_0 blocks in the tensor.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Dequantize the Q4_0 data to a full-precision `Tensor<Wgpu, 2>`.
    ///
    /// This reads the raw bytes back from GPU and dequantizes on CPU.
    /// Intended for diagnostics and testing — the hot path uses
    /// [`q4_matmul`](super::op::q4_matmul) which dequantizes on GPU.
    pub fn dequantize(&self) -> Tensor<Wgpu, 2> {
        let bytes = self.client.read_one(self.handle.clone());
        let raw: &[u8] = &bytes;

        let [n, k] = self.shape;
        let num_elements = n * k;
        let mut output = vec![0.0f32; num_elements];

        for block_idx in 0..self.num_blocks {
            let offset = block_idx * 18;
            let d_bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let base = block_idx * 32;
            for i in 0..16 {
                let byte = raw[offset + 2 + i];
                let lo = (byte & 0x0F) as f32 - 8.0;
                let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                output[base + i] = lo * d;
                output[base + i + 16] = hi * d;
            }
        }

        let tensor_data = TensorData::new(output, [n, k]);
        Tensor::from_data(tensor_data, &self.device)
    }
}
