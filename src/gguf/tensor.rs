//! Quantized weight tensor stored on GPU (Q4_0 and Q8_0).
//!
//! [`Q4Tensor`] uploads raw quantized blocks to a GPU storage buffer and provides
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

use super::reader::GgmlDtype;

/// Block constants for Q4_0: 32 elements per block, 18 bytes per block.
pub const Q4_BLOCK_SIZE: usize = 32;
pub const Q4_BLOCK_BYTES: usize = 18;

/// Block constants for Q8_0: 32 elements per block, 34 bytes per block.
pub const Q8_BLOCK_SIZE: usize = 32;
pub const Q8_BLOCK_BYTES: usize = 34;

/// A quantized weight tensor living on GPU (supports Q4_0 and Q8_0).
///
/// For Q4_0: the buffer contains raw Q4_0 blocks (18 bytes per block of 32 elements).
/// For Q8_0: the buffer contains raw Q8_0 blocks (34 bytes per block of 32 elements).
/// Both are laid out exactly as in GGUF.
pub struct Q4Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
    dtype: GgmlDtype,
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
        Self::from_quantized_bytes(raw_bytes, shape, GgmlDtype::Q4_0, device)
    }

    /// Upload raw Q8_0 bytes to a GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`, matching PyTorch/GGUF
    /// convention. `raw_bytes` must contain exactly `(N * K / 32) * 34` bytes.
    /// The element count `N * K` must be divisible by 32.
    pub fn from_q8_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        Self::from_quantized_bytes(raw_bytes, shape, GgmlDtype::Q8_0, device)
    }

    /// Upload raw quantized bytes to a GPU storage buffer (Q4_0 or Q8_0).
    pub fn from_quantized_bytes(
        raw_bytes: &[u8],
        shape: [usize; 2],
        dtype: GgmlDtype,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        let block_size = match dtype {
            GgmlDtype::Q4_0 => Q4_BLOCK_SIZE,
            GgmlDtype::Q8_0 => Q8_BLOCK_SIZE,
            _ => anyhow::bail!("from_quantized_bytes only supports Q4_0 and Q8_0, got {dtype:?}"),
        };
        let block_bytes = match dtype {
            GgmlDtype::Q4_0 => Q4_BLOCK_BYTES,
            GgmlDtype::Q8_0 => Q8_BLOCK_BYTES,
            _ => unreachable!(),
        };

        ensure!(
            num_elements % block_size == 0,
            "{dtype:?} requires element count divisible by {block_size}, got {num_elements}"
        );
        let num_blocks = num_elements / block_size;
        let expected_bytes = num_blocks * block_bytes;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "{dtype:?} byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
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
            dtype,
            client,
            device: device.clone(),
        })
    }

    /// Logical weight dimensions `[N, K]` = `[out_features, in_features]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Number of quantized blocks in the tensor.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// The quantization dtype (Q4_0 or Q8_0).
    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    /// Bytes per block for this tensor's quantization type.
    pub fn block_bytes(&self) -> usize {
        match self.dtype {
            GgmlDtype::Q4_0 => Q4_BLOCK_BYTES,
            GgmlDtype::Q8_0 => Q8_BLOCK_BYTES,
            _ => unreachable!("Q4Tensor should only hold Q4_0 or Q8_0"),
        }
    }

    /// Read raw quantized bytes from GPU.
    ///
    /// Returns exactly `num_blocks * block_bytes` bytes (may include padding at end
    /// that was added for 4-byte alignment).
    pub fn read_bytes(&self) -> Vec<u8> {
        let raw = self.client.read_one(self.handle.clone());
        let expected = self.num_blocks * self.block_bytes();
        assert!(
            raw.len() >= expected,
            "Q4Tensor::read_bytes: GPU buffer shorter than expected: got {}, expected {expected}",
            raw.len()
        );
        raw[..expected].to_vec()
    }

    /// Dequantize the quantized data to a full-precision `Tensor<Wgpu, 2>`.
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

        match self.dtype {
            GgmlDtype::Q4_0 => {
                for block_idx in 0..self.num_blocks {
                    let offset = block_idx * Q4_BLOCK_BYTES;
                    let d_bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
                    let d = half::f16::from_bits(d_bits).to_f32();

                    let base = block_idx * Q4_BLOCK_SIZE;
                    for i in 0..16 {
                        let byte = raw[offset + 2 + i];
                        let lo = (byte & 0x0F) as f32 - 8.0;
                        let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                        output[base + i] = lo * d;
                        output[base + i + 16] = hi * d;
                    }
                }
            }
            GgmlDtype::Q8_0 => {
                for block_idx in 0..self.num_blocks {
                    let offset = block_idx * Q8_BLOCK_BYTES;
                    let d_bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
                    let d = half::f16::from_bits(d_bits).to_f32();

                    let base = block_idx * Q8_BLOCK_SIZE;
                    for i in 0..Q8_BLOCK_SIZE {
                        let val = raw[offset + 2 + i] as i8;
                        output[base + i] = d * val as f32;
                    }
                }
            }
            _ => unreachable!("Q4Tensor should only hold Q4_0 or Q8_0"),
        }

        let tensor_data = TensorData::new(output, [n, k]);
        Tensor::from_data(tensor_data, &self.device)
    }
}
