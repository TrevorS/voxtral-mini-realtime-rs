//! GGUF quantized inference on GPU.
//!
//! Provides GGUF file reading, Q4_0 GPU tensor storage, and a fused
//! dequant+matmul compute shader launched through Burn's custom kernel API.
//! Named after the container format rather than a specific quant type so
//! future formats (Q4_K_M, Q5_K_M, Q8_0, …) live alongside Q4_0.

pub mod linear;
pub mod loader;
pub mod model;
pub mod op;
pub mod reader;
pub mod tensor;

#[cfg(test)]
mod tests;

pub use linear::Q4Linear;
pub use loader::{Q4ModelLoader, Q4ModelParts};
pub use model::{
    Q4AdaRmsNorm, Q4Adapter, Q4Attention, Q4AudioEncoder, Q4DecoderLayer, Q4EncoderLayer,
    Q4FeedForward, Q4LanguageModel, Q4VoxtralModel,
};
pub use op::q4_matmul;
pub use reader::{GgmlDtype, GgufReader, GgufTensorInfo, ShardedCursor};
pub use tensor::Q4Tensor;
