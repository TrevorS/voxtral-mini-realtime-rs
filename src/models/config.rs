//! Model configuration for Voxtral Mini 4B Realtime.
//!
//! Parses the `params.json` configuration file from the HuggingFace model.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level Voxtral configuration.
///
/// Combines audio encoder and language model configurations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VoxtralConfig {
    /// Audio encoder configuration
    pub audio_encoder: AudioEncoderConfig,
    /// Language model configuration
    pub language_model: LanguageModelConfig,
    /// Audio-to-LLM adapter configuration
    pub adapter: AdapterConfig,
    /// Audio input specifications
    pub audio: AudioInputConfig,
    /// ADA RMSNorm T-conditioning dimension (0 = disabled)
    pub ada_rms_norm_t_cond_dim: usize,
}

impl VoxtralConfig {
    /// Load configuration from a `params.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;
        Self::from_json(&content)
    }

    /// Parse configuration from JSON string.
    ///
    /// Handles the nested Voxtral params.json format:
    /// - Top-level: LLM config
    /// - `multimodal.whisper_model_args.encoder_args`: Audio encoder config
    /// - `multimodal.whisper_model_args.encoder_args.audio_encoding_args`: Audio input specs
    pub fn from_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).context("Failed to parse JSON")?;

        // Parse LLM config from top level
        let language_model = LanguageModelConfig::from_json_value(&v)?;

        // Parse audio encoder from nested path
        let encoder_args = v
            .get("multimodal")
            .and_then(|m| m.get("whisper_model_args"))
            .and_then(|w| w.get("encoder_args"));

        let audio_encoder = if let Some(enc) = encoder_args {
            AudioEncoderConfig::from_json_value(enc)?
        } else {
            AudioEncoderConfig::default()
        };

        // Parse audio input specs from nested path
        let audio_encoding_args = encoder_args.and_then(|e| e.get("audio_encoding_args"));

        let audio = if let Some(aud) = audio_encoding_args {
            AudioInputConfig::from_json_value(aud)?
        } else {
            AudioInputConfig::default()
        };

        // Parse downsample factor
        let downsample_factor = v
            .get("multimodal")
            .and_then(|m| m.get("whisper_model_args"))
            .and_then(|w| w.get("downsample_args"))
            .and_then(|d| d.get("downsample_factor"))
            .and_then(|f| f.as_u64())
            .unwrap_or(4) as usize;

        // Build adapter config based on encoder output and LLM input
        let adapter = AdapterConfig {
            input_dim: audio_encoder.dim * downsample_factor, // 1280 * 4 = 5120
            hidden_dim: audio_encoder.dim * downsample_factor,
            output_dim: language_model.dim,
        };

        // ADA RMSNorm conditioning
        let ada_rms_norm_t_cond_dim = if v
            .get("ada_rms_norm_t_cond")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            v.get("ada_rms_norm_t_cond_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(32) as usize
        } else {
            0
        };

        Ok(Self {
            audio_encoder,
            language_model,
            adapter,
            audio,
            ada_rms_norm_t_cond_dim,
        })
    }

    /// Load from a model directory (looks for `params.json`).
    pub fn from_model_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let path = dir.as_ref().join("params.json");
        Self::from_file(path)
    }

    /// Whether ADA RMSNorm (T-conditional) is enabled.
    pub fn has_ada_rms_norm(&self) -> bool {
        self.ada_rms_norm_t_cond_dim > 0
    }
}

/// Audio encoder configuration (causal Whisper-style).
///
/// ~0.6B parameters with 32 transformer layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEncoderConfig {
    /// Hidden dimension
    #[serde(default = "default_audio_dim")]
    pub dim: usize,
    /// Number of transformer layers
    #[serde(default = "default_audio_layers")]
    pub n_layers: usize,
    /// Number of query heads (MHA, not GQA)
    #[serde(default = "default_audio_heads")]
    pub n_heads: usize,
    /// Number of KV heads (same as query heads for MHA)
    #[serde(default = "default_audio_heads")]
    pub n_kv_heads: usize,
    /// Head dimension
    #[serde(default = "default_audio_head_dim")]
    pub head_dim: usize,
    /// FFN hidden dimension
    #[serde(default = "default_audio_hidden_dim")]
    pub hidden_dim: usize,
    /// Sliding window size for attention
    #[serde(default = "default_audio_sliding_window")]
    pub sliding_window: usize,
    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// RMS norm epsilon
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Whether attention uses biases
    #[serde(default = "default_true")]
    pub use_biases: bool,
    /// Whether attention is causal
    #[serde(default = "default_true")]
    pub causal: bool,
    /// FFN type (swiglu)
    #[serde(default = "default_ffn_type")]
    pub ffn_type: String,
    /// Norm type (rms_norm)
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
}

impl AudioEncoderConfig {
    fn from_json_value(v: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            dim: v["dim"].as_u64().unwrap_or(1280) as usize,
            n_layers: v["n_layers"].as_u64().unwrap_or(32) as usize,
            n_heads: v["n_heads"].as_u64().unwrap_or(32) as usize,
            n_kv_heads: v["n_kv_heads"].as_u64().unwrap_or(32) as usize,
            head_dim: v["head_dim"].as_u64().unwrap_or(64) as usize,
            hidden_dim: v["hidden_dim"].as_u64().unwrap_or(5120) as usize,
            sliding_window: v["sliding_window"].as_u64().unwrap_or(750) as usize,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5),
            use_biases: v["use_biases"].as_bool().unwrap_or(true),
            causal: v["causal"].as_bool().unwrap_or(true),
            ffn_type: v["ffn_type"].as_str().unwrap_or("swiglu").to_string(),
            norm_type: v["norm_type"].as_str().unwrap_or("rms_norm").to_string(),
        })
    }
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            dim: default_audio_dim(),
            n_layers: default_audio_layers(),
            n_heads: default_audio_heads(),
            n_kv_heads: default_audio_heads(),
            head_dim: default_audio_head_dim(),
            hidden_dim: default_audio_hidden_dim(),
            sliding_window: default_audio_sliding_window(),
            rope_theta: default_rope_theta(),
            norm_eps: default_norm_eps(),
            use_biases: true,
            causal: true,
            ffn_type: default_ffn_type(),
            norm_type: default_norm_type(),
        }
    }
}

/// Language model configuration (Ministral-3B based).
///
/// ~3.4B parameters with 26 transformer layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelConfig {
    /// Hidden dimension
    #[serde(default = "default_llm_dim")]
    pub dim: usize,
    /// Number of transformer layers
    #[serde(default = "default_llm_layers")]
    pub n_layers: usize,
    /// Number of query heads
    #[serde(default = "default_llm_q_heads")]
    pub n_heads: usize,
    /// Number of KV heads (GQA)
    #[serde(default = "default_llm_kv_heads")]
    pub n_kv_heads: usize,
    /// Head dimension
    #[serde(default = "default_llm_head_dim")]
    pub head_dim: usize,
    /// FFN hidden dimension
    #[serde(default = "default_llm_hidden_dim")]
    pub hidden_dim: usize,
    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Sliding window size for attention
    #[serde(default = "default_llm_sliding_window")]
    pub sliding_window: usize,
    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// RMS norm epsilon
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Whether embeddings are tied with LM head
    #[serde(default = "default_true")]
    pub tied_embeddings: bool,
    /// Whether attention uses biases
    #[serde(default = "default_false")]
    pub use_biases: bool,
    /// Whether attention is causal
    #[serde(default = "default_true")]
    pub causal: bool,
}

impl LanguageModelConfig {
    fn from_json_value(v: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            dim: v["dim"].as_u64().unwrap_or(3072) as usize,
            n_layers: v["n_layers"].as_u64().unwrap_or(26) as usize,
            n_heads: v["n_heads"].as_u64().unwrap_or(32) as usize,
            n_kv_heads: v["n_kv_heads"].as_u64().unwrap_or(8) as usize,
            head_dim: v["head_dim"].as_u64().unwrap_or(128) as usize,
            hidden_dim: v["hidden_dim"].as_u64().unwrap_or(9216) as usize,
            vocab_size: v["vocab_size"].as_u64().unwrap_or(131072) as usize,
            sliding_window: v["sliding_window"].as_u64().unwrap_or(8192) as usize,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            norm_eps: v["norm_eps"].as_f64().unwrap_or(1e-5),
            tied_embeddings: v["tied_embeddings"].as_bool().unwrap_or(true),
            use_biases: v["use_biases"].as_bool().unwrap_or(false),
            causal: v["causal"].as_bool().unwrap_or(true),
        })
    }

    /// GQA group size (queries per KV head).
    pub fn gqa_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

impl Default for LanguageModelConfig {
    fn default() -> Self {
        Self {
            dim: default_llm_dim(),
            n_layers: default_llm_layers(),
            n_heads: default_llm_q_heads(),
            n_kv_heads: default_llm_kv_heads(),
            head_dim: default_llm_head_dim(),
            hidden_dim: default_llm_hidden_dim(),
            vocab_size: default_vocab_size(),
            sliding_window: default_llm_sliding_window(),
            rope_theta: default_rope_theta(),
            norm_eps: default_norm_eps(),
            tied_embeddings: true,
            use_biases: false,
            causal: true,
        }
    }
}

/// Audio-to-LLM adapter configuration.
///
/// Projects audio encoder output to LLM input dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// Input dimension (audio encoder output after reshape)
    #[serde(default = "default_adapter_input")]
    pub input_dim: usize,
    /// Hidden dimension
    #[serde(default = "default_adapter_hidden")]
    pub hidden_dim: usize,
    /// Output dimension (LLM hidden size)
    #[serde(default = "default_llm_dim")]
    pub output_dim: usize,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            input_dim: default_adapter_input(),
            hidden_dim: default_adapter_hidden(),
            output_dim: default_llm_dim(),
        }
    }
}

/// Audio input specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputConfig {
    /// Sample rate in Hz
    #[serde(default = "default_sample_rate")]
    pub sampling_rate: u32,
    /// Number of mel filterbank bins
    #[serde(default = "default_n_mels")]
    pub num_mel_bins: usize,
    /// FFT hop length in samples
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    /// FFT window size in samples
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    /// Global log mel maximum for normalization
    #[serde(default = "default_log_mel_max")]
    pub global_log_mel_max: f32,
    /// Frame rate in Hz (after downsampling)
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f32,
    /// Transcription format
    #[serde(default = "default_transcription_format")]
    pub transcription_format: String,
}

impl AudioInputConfig {
    fn from_json_value(v: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            sampling_rate: v["sampling_rate"].as_u64().unwrap_or(16000) as u32,
            num_mel_bins: v["num_mel_bins"].as_u64().unwrap_or(128) as usize,
            hop_length: v["hop_length"].as_u64().unwrap_or(160) as usize,
            window_size: v["window_size"].as_u64().unwrap_or(400) as usize,
            global_log_mel_max: v["global_log_mel_max"].as_f64().unwrap_or(1.5) as f32,
            frame_rate: v["frame_rate"].as_f64().unwrap_or(12.5) as f32,
            transcription_format: v["transcription_format"]
                .as_str()
                .unwrap_or("streaming")
                .to_string(),
        })
    }

    /// Milliseconds per text token (Voxtral uses 1 token = 80ms).
    pub fn ms_per_token(&self) -> f32 {
        1000.0 / self.frame_rate
    }

    /// Audio samples per text token.
    pub fn samples_per_token(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate) as usize
    }

    /// Raw mel frame rate (before downsampling).
    pub fn raw_frame_rate(&self) -> f32 {
        self.sampling_rate as f32 / self.hop_length as f32
    }
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        Self {
            sampling_rate: default_sample_rate(),
            num_mel_bins: default_n_mels(),
            hop_length: default_hop_length(),
            window_size: default_window_size(),
            global_log_mel_max: default_log_mel_max(),
            frame_rate: default_frame_rate(),
            transcription_format: default_transcription_format(),
        }
    }
}

// Default value functions for serde
fn default_audio_dim() -> usize {
    1280
}
fn default_audio_layers() -> usize {
    32
}
fn default_audio_heads() -> usize {
    32
}
fn default_audio_head_dim() -> usize {
    64
}
fn default_audio_hidden_dim() -> usize {
    5120
}
fn default_audio_sliding_window() -> usize {
    750
}

fn default_llm_dim() -> usize {
    3072
}
fn default_llm_layers() -> usize {
    26
}
fn default_llm_q_heads() -> usize {
    32
}
fn default_llm_kv_heads() -> usize {
    8
}
fn default_llm_head_dim() -> usize {
    128
}
fn default_llm_hidden_dim() -> usize {
    9216
}
fn default_vocab_size() -> usize {
    131072
}
fn default_llm_sliding_window() -> usize {
    8192
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_norm_eps() -> f64 {
    1e-5
}

fn default_adapter_input() -> usize {
    5120
}
fn default_adapter_hidden() -> usize {
    5120
}

fn default_sample_rate() -> u32 {
    16000
}
fn default_n_mels() -> usize {
    128
}
fn default_hop_length() -> usize {
    160
}
fn default_window_size() -> usize {
    400
}
fn default_log_mel_max() -> f32 {
    1.5
}
fn default_frame_rate() -> f32 {
    12.5
} // After 4x downsample: 100 / 8 = 12.5 Hz

fn default_true() -> bool {
    true
}
fn default_false() -> bool {
    false
}
fn default_ffn_type() -> String {
    "swiglu".to_string()
}
fn default_norm_type() -> String {
    "rms_norm".to_string()
}
fn default_transcription_format() -> String {
    "streaming".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_encoder_defaults() {
        let config = AudioEncoderConfig::default();
        assert_eq!(config.dim, 1280);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 32);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.hidden_dim, 5120);
        assert_eq!(config.sliding_window, 750);
        assert!(config.use_biases);
        assert!(config.causal);
    }

    #[test]
    fn test_language_model_defaults() {
        let config = LanguageModelConfig::default();
        assert_eq!(config.dim, 3072);
        assert_eq!(config.n_layers, 26);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.hidden_dim, 9216);
        assert_eq!(config.vocab_size, 131072);
        assert_eq!(config.sliding_window, 8192);
        assert_eq!(config.gqa_groups(), 4);
        assert!(!config.use_biases);
        assert!(config.causal);
    }

    #[test]
    fn test_adapter_defaults() {
        let config = AdapterConfig::default();
        assert_eq!(config.input_dim, 5120);
        assert_eq!(config.hidden_dim, 5120);
        assert_eq!(config.output_dim, 3072);
    }

    #[test]
    fn test_audio_input_defaults() {
        let config = AudioInputConfig::default();
        assert_eq!(config.sampling_rate, 16000);
        assert_eq!(config.num_mel_bins, 128);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.window_size, 400);
        assert!((config.global_log_mel_max - 1.5).abs() < 1e-6);
        assert!((config.frame_rate - 12.5).abs() < 1e-6);
        assert!((config.ms_per_token() - 80.0).abs() < 1e-6);
        assert_eq!(config.samples_per_token(), 1280);
        assert!((config.raw_frame_rate() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_voxtral_config_defaults() {
        let config = VoxtralConfig::default();
        assert_eq!(config.audio_encoder.dim, 1280);
        assert_eq!(config.language_model.dim, 3072);
        assert_eq!(config.adapter.output_dim, 3072);
        assert_eq!(config.audio.sampling_rate, 16000);
        assert_eq!(config.ada_rms_norm_t_cond_dim, 0);
        assert!(!config.has_ada_rms_norm());
    }

    #[test]
    fn test_gqa_calculation() {
        let config = LanguageModelConfig {
            n_heads: 32,
            n_kv_heads: 8,
            ..Default::default()
        };
        assert_eq!(config.gqa_groups(), 4);

        let config_mha = LanguageModelConfig {
            n_heads: 32,
            n_kv_heads: 32,
            ..Default::default()
        };
        assert_eq!(config_mha.gqa_groups(), 1);
    }

    #[test]
    fn test_parse_actual_params_json() {
        let json = r#"{
          "dim": 3072,
          "n_layers": 26,
          "head_dim": 128,
          "hidden_dim": 9216,
          "n_heads": 32,
          "n_kv_heads": 8,
          "use_biases": false,
          "causal": true,
          "rope_theta": 1000000.0,
          "norm_eps": 1e-05,
          "vocab_size": 131072,
          "model_parallel": 1,
          "tied_embeddings": true,
          "sliding_window": 8192,
          "model_max_length": 131072,
          "multimodal": {
            "whisper_model_args": {
              "encoder_args": {
                "audio_encoding_args": {
                  "sampling_rate": 16000,
                  "frame_rate": 12.5,
                  "num_mel_bins": 128,
                  "hop_length": 160,
                  "window_size": 400,
                  "chunk_length_s": null,
                  "global_log_mel_max": 1.5,
                  "transcription_format": "streaming"
                },
                "dim": 1280,
                "n_layers": 32,
                "head_dim": 64,
                "hidden_dim": 5120,
                "n_heads": 32,
                "vocab_size": 131072,
                "n_kv_heads": 32,
                "use_biases": true,
                "use_cache": false,
                "rope_theta": 1000000.0,
                "causal": true,
                "norm_eps": 1e-05,
                "pos_embed": "rope",
                "max_source_positions": null,
                "ffn_type": "swiglu",
                "norm_type": "rms_norm",
                "sliding_window": 750
              },
              "downsample_args": {
                "downsample_factor": 4
              }
            }
          },
          "ada_rms_norm_t_cond": true,
          "ada_rms_norm_t_cond_dim": 32
        }"#;

        let config = VoxtralConfig::from_json(json).unwrap();

        // LLM config
        assert_eq!(config.language_model.dim, 3072);
        assert_eq!(config.language_model.n_layers, 26);
        assert_eq!(config.language_model.n_heads, 32);
        assert_eq!(config.language_model.n_kv_heads, 8);
        assert_eq!(config.language_model.head_dim, 128);
        assert_eq!(config.language_model.hidden_dim, 9216);
        assert_eq!(config.language_model.vocab_size, 131072);
        assert_eq!(config.language_model.sliding_window, 8192);
        assert!(!config.language_model.use_biases);
        assert!(config.language_model.tied_embeddings);

        // Audio encoder config
        assert_eq!(config.audio_encoder.dim, 1280);
        assert_eq!(config.audio_encoder.n_layers, 32);
        assert_eq!(config.audio_encoder.n_heads, 32);
        assert_eq!(config.audio_encoder.n_kv_heads, 32);
        assert_eq!(config.audio_encoder.head_dim, 64);
        assert_eq!(config.audio_encoder.hidden_dim, 5120);
        assert_eq!(config.audio_encoder.sliding_window, 750);
        assert!(config.audio_encoder.use_biases);
        assert!(config.audio_encoder.causal);
        assert_eq!(config.audio_encoder.ffn_type, "swiglu");
        assert_eq!(config.audio_encoder.norm_type, "rms_norm");

        // Audio input config
        assert_eq!(config.audio.sampling_rate, 16000);
        assert_eq!(config.audio.num_mel_bins, 128);
        assert_eq!(config.audio.hop_length, 160);
        assert_eq!(config.audio.window_size, 400);
        assert!((config.audio.global_log_mel_max - 1.5).abs() < 1e-6);
        assert!((config.audio.frame_rate - 12.5).abs() < 1e-6);
        assert_eq!(config.audio.transcription_format, "streaming");

        // Adapter config (derived)
        assert_eq!(config.adapter.input_dim, 5120); // 1280 * 4
        assert_eq!(config.adapter.output_dim, 3072);

        // ADA RMSNorm
        assert!(config.has_ada_rms_norm());
        assert_eq!(config.ada_rms_norm_t_cond_dim, 32);
    }

    #[test]
    fn test_load_from_model_dir() {
        // This test requires the model to be downloaded
        let model_dir = std::path::Path::new("models/voxtral");
        if !model_dir.exists() {
            eprintln!("Skipping test_load_from_model_dir: models/voxtral not found");
            eprintln!("Run ./scripts/download_model.py to download the model");
            return;
        }

        let config = VoxtralConfig::from_model_dir(model_dir).unwrap();

        // Verify key values match expected
        assert_eq!(config.language_model.dim, 3072);
        assert_eq!(config.language_model.n_layers, 26);
        assert_eq!(config.audio_encoder.dim, 1280);
        assert_eq!(config.audio_encoder.n_layers, 32);
        assert!(config.has_ada_rms_norm());
        assert_eq!(config.ada_rms_norm_t_cond_dim, 32);

        println!("Loaded config from models/voxtral:");
        println!(
            "  LLM: {} layers, {} dim, {} heads (GQA {})",
            config.language_model.n_layers,
            config.language_model.dim,
            config.language_model.n_heads,
            config.language_model.gqa_groups()
        );
        println!(
            "  Encoder: {} layers, {} dim, {} heads",
            config.audio_encoder.n_layers, config.audio_encoder.dim, config.audio_encoder.n_heads
        );
        println!(
            "  Audio: {}Hz, {} mel bins, {:.1}Hz frame rate",
            config.audio.sampling_rate, config.audio.num_mel_bins, config.audio.frame_rate
        );
        println!(
            "  ADA RMSNorm: {} (dim={})",
            config.has_ada_rms_norm(),
            config.ada_rms_norm_t_cond_dim
        );
    }
}
