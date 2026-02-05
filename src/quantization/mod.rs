//! Quantization support for Voxtral model.
//!
//! This module provides utilities for quantizing the Voxtral model to reduce
//! memory footprint for WASM/browser deployment. Supports INT8 and INT4
//! quantization with configurable strategies.

use burn::module::{Module, Quantizer};
use burn::prelude::Backend;
use burn::tensor::quantization::{
    BlockSize, Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

use crate::models::decoder::LanguageModel;
use crate::models::encoder::AudioEncoder;
use crate::models::voxtral::VoxtralModel;

/// Quantization precision levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPrecision {
    /// 8-bit integer quantization (~50% size reduction)
    Q8,
    /// 4-bit integer quantization (~75% size reduction)
    Q4,
    /// Keep at full precision (BF16/F32)
    Full,
}

impl std::fmt::Display for QuantPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantPrecision::Q8 => write!(f, "Q8"),
            QuantPrecision::Q4 => write!(f, "Q4"),
            QuantPrecision::Full => write!(f, "Full"),
        }
    }
}

/// Configuration for model quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Precision for the audio encoder
    pub encoder_precision: QuantPrecision,
    /// Precision for the language model decoder
    pub decoder_precision: QuantPrecision,
    /// Precision for the adapter
    pub adapter_precision: QuantPrecision,
    /// Block size for block-level quantization (0 = per-tensor)
    pub block_size: usize,
}

impl QuantConfig {
    /// Full precision - no quantization (baseline)
    pub fn full() -> Self {
        Self {
            encoder_precision: QuantPrecision::Full,
            decoder_precision: QuantPrecision::Full,
            adapter_precision: QuantPrecision::Full,
            block_size: 0,
        }
    }

    /// Q8 quantization for everything
    pub fn q8_full() -> Self {
        Self {
            encoder_precision: QuantPrecision::Q8,
            decoder_precision: QuantPrecision::Q8,
            adapter_precision: QuantPrecision::Q8,
            block_size: 32,
        }
    }

    /// Q4 quantization for everything
    pub fn q4_full() -> Self {
        Self {
            encoder_precision: QuantPrecision::Q4,
            decoder_precision: QuantPrecision::Q4,
            adapter_precision: QuantPrecision::Q4,
            block_size: 32,
        }
    }

    /// Mixed precision: Q4 decoder, full precision encoder
    /// Recommended for best accuracy/size tradeoff
    pub fn mixed_q4_decoder() -> Self {
        Self {
            encoder_precision: QuantPrecision::Full,
            decoder_precision: QuantPrecision::Q4,
            adapter_precision: QuantPrecision::Full,
            block_size: 32,
        }
    }

    /// Mixed precision: Q8 decoder, full precision encoder
    pub fn mixed_q8_decoder() -> Self {
        Self {
            encoder_precision: QuantPrecision::Full,
            decoder_precision: QuantPrecision::Q8,
            adapter_precision: QuantPrecision::Full,
            block_size: 32,
        }
    }

    /// Q4 decoder and encoder, full precision adapter
    pub fn q4_model() -> Self {
        Self {
            encoder_precision: QuantPrecision::Q4,
            decoder_precision: QuantPrecision::Q4,
            adapter_precision: QuantPrecision::Full,
            block_size: 32,
        }
    }

    /// All presets available for testing.
    ///
    /// Q8 presets work on NdArray backend. Q4 requires backend support
    /// (not yet available in NdArray or Wgpu).
    /// Encoder uses per-tensor quantization to handle Conv1d layers safely.
    pub fn all_presets() -> Vec<(&'static str, Self)> {
        vec![
            ("full", Self::full()),
            ("q8-full", Self::q8_full()),
            ("mixed-q8-decoder", Self::mixed_q8_decoder()),
            ("mixed-q4-decoder", Self::mixed_q4_decoder()), // Q4 not yet supported
            ("q4-full", Self::q4_full()),                   // Q4 not yet supported
            ("q4-model", Self::q4_model()),                 // Q4 not yet supported
        ]
    }

    /// Presets that work with current NdArray backend.
    pub fn working_presets() -> Vec<(&'static str, Self)> {
        vec![
            ("full", Self::full()),
            ("q8-full", Self::q8_full()),
            ("mixed-q8-decoder", Self::mixed_q8_decoder()),
        ]
    }

    /// Estimate the resulting model size in bytes
    pub fn estimate_size(&self, original_size_bytes: u64) -> u64 {
        // Rough component breakdown for Voxtral:
        // Encoder: ~17% of params
        // Decoder: ~75% of params
        // Adapter + Embeddings: ~8% of params

        let encoder_ratio = 0.17;
        let decoder_ratio = 0.75;
        let adapter_ratio = 0.08;

        let encoder_multiplier = self.encoder_precision.size_multiplier();
        let decoder_multiplier = self.decoder_precision.size_multiplier();
        let adapter_multiplier = self.adapter_precision.size_multiplier();

        let encoder_size = (original_size_bytes as f64 * encoder_ratio * encoder_multiplier) as u64;
        let decoder_size = (original_size_bytes as f64 * decoder_ratio * decoder_multiplier) as u64;
        let adapter_size = (original_size_bytes as f64 * adapter_ratio * adapter_multiplier) as u64;

        encoder_size + decoder_size + adapter_size
    }

    /// Get a description of this config
    pub fn description(&self) -> String {
        format!(
            "encoder={}, decoder={}, adapter={}, block={}",
            self.encoder_precision,
            self.decoder_precision,
            self.adapter_precision,
            if self.block_size > 0 {
                self.block_size.to_string()
            } else {
                "tensor".to_string()
            }
        )
    }
}

impl QuantPrecision {
    /// Size multiplier relative to FP16/BF16
    pub fn size_multiplier(&self) -> f64 {
        match self {
            QuantPrecision::Full => 1.0,
            QuantPrecision::Q8 => 0.5,  // 8-bit vs 16-bit
            QuantPrecision::Q4 => 0.25, // 4-bit vs 16-bit
        }
    }

    /// Create a QuantScheme for this precision.
    ///
    /// `block_size` controls granularity:
    /// - 0 = per-tensor quantization (safe for all layer types including Conv1d)
    /// - >0 = per-block quantization (only safe for 2D tensors like Linear weights)
    pub fn to_scheme(&self, block_size: usize) -> Option<QuantScheme> {
        let level = if block_size > 0 {
            QuantLevel::Block(BlockSize::new([block_size as u8]))
        } else {
            QuantLevel::Tensor
        };

        match self {
            QuantPrecision::Full => None,
            QuantPrecision::Q8 => Some(QuantScheme {
                level,
                mode: QuantMode::Symmetric,
                value: QuantValue::Q8S,
                store: QuantStore::Native,
                param: QuantParam::F32,
            }),
            QuantPrecision::Q4 => Some(QuantScheme {
                level,
                mode: QuantMode::Symmetric,
                value: QuantValue::Q4S,
                store: QuantStore::PackedU32(0),
                param: QuantParam::F32,
            }),
        }
    }
}

/// Create a quantizer for the given precision
pub fn create_quantizer(precision: QuantPrecision, block_size: usize) -> Option<Quantizer> {
    precision.to_scheme(block_size).map(|scheme| Quantizer {
        calibration: Calibration::MinMax,
        scheme,
    })
}

/// Quantize an audio encoder with the given precision.
///
/// Uses per-tensor quantization (block_size=0) because the encoder's Conv1d
/// downsampler has kernel_size=1, which causes block quantization to produce
/// a zero-sized dimension: [1280, 128, 1] / [1, 1, 32] = [1280, 128, 0].
pub fn quantize_encoder<B: Backend>(
    encoder: AudioEncoder<B>,
    precision: QuantPrecision,
    _block_size: usize,
) -> AudioEncoder<B> {
    // Force per-tensor quantization for encoder (Conv1d-safe)
    match create_quantizer(precision, 0) {
        Some(mut quantizer) => encoder.quantize_weights(&mut quantizer),
        None => encoder,
    }
}

/// Quantize a language model decoder with the given precision
pub fn quantize_decoder<B: Backend>(
    decoder: LanguageModel<B>,
    precision: QuantPrecision,
    block_size: usize,
) -> LanguageModel<B> {
    match create_quantizer(precision, block_size) {
        Some(mut quantizer) => decoder.quantize_weights(&mut quantizer),
        None => decoder,
    }
}

/// Quantize a Voxtral model according to the config
pub fn quantize_model<B: Backend>(model: VoxtralModel<B>, config: &QuantConfig) -> VoxtralModel<B> {
    // Destructure the model to get individual components
    let (encoder, decoder, adapter, downsample_factor) = model.into_parts();

    // Quantize each component according to config
    let encoder = quantize_encoder(encoder, config.encoder_precision, config.block_size);
    let decoder = quantize_decoder(decoder, config.decoder_precision, config.block_size);

    // Adapter quantization
    let adapter = match create_quantizer(config.adapter_precision, config.block_size) {
        Some(mut quantizer) => adapter.quantize_weights(&mut quantizer),
        None => adapter,
    };

    // Reconstruct the model
    VoxtralModel::new(encoder, decoder, adapter, downsample_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_presets() {
        let presets = QuantConfig::all_presets();
        assert_eq!(presets.len(), 6);

        for (name, config) in presets {
            println!("{}: {}", name, config.description());
        }
    }

    #[test]
    fn test_working_presets() {
        let presets = QuantConfig::working_presets();
        assert_eq!(presets.len(), 3);
        assert_eq!(presets[0].0, "full");
        assert_eq!(presets[1].0, "q8-full");
        assert_eq!(presets[2].0, "mixed-q8-decoder");
    }

    #[test]
    fn test_size_estimation() {
        let original = 8_860_000_000u64; // 8.86 GB

        let full = QuantConfig::full();
        let q8 = QuantConfig::q8_full();
        let q4 = QuantConfig::q4_full();
        let mixed = QuantConfig::mixed_q4_decoder();

        println!("Original: {:.2} GB", original as f64 / 1e9);
        println!("Full: {:.2} GB", full.estimate_size(original) as f64 / 1e9);
        println!("Q8: {:.2} GB", q8.estimate_size(original) as f64 / 1e9);
        println!("Q4: {:.2} GB", q4.estimate_size(original) as f64 / 1e9);
        println!(
            "Mixed Q4 decoder: {:.2} GB",
            mixed.estimate_size(original) as f64 / 1e9
        );

        // Q4 should be roughly 25% of original
        assert!(q4.estimate_size(original) < original / 3);
    }
}
