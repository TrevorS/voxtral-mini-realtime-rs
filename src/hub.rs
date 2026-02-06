//! HuggingFace Hub integration for downloading Voxtral models.
//!
//! This module provides utilities for downloading model weights from HuggingFace Hub.
//! Requires the `hub` feature to be enabled.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Model file paths for a downloaded Voxtral model.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Path to the model directory
    pub model_dir: PathBuf,
    /// Path to consolidated.safetensors
    pub weights: PathBuf,
    /// Path to params.json
    pub config: PathBuf,
    /// Path to tekken.json
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// Create paths from a model directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Self {
        let model_dir = dir.as_ref().to_path_buf();
        Self {
            weights: model_dir.join("consolidated.safetensors"),
            config: model_dir.join("params.json"),
            tokenizer: model_dir.join("tekken.json"),
            model_dir,
        }
    }

    /// Check if all required files exist.
    pub fn validate(&self) -> Result<()> {
        if !self.weights.exists() {
            anyhow::bail!("Weights not found: {}", self.weights.display());
        }
        if !self.config.exists() {
            anyhow::bail!("Config not found: {}", self.config.display());
        }
        if !self.tokenizer.exists() {
            anyhow::bail!("Tokenizer not found: {}", self.tokenizer.display());
        }
        Ok(())
    }

    /// Download model from HuggingFace Hub.
    ///
    /// Downloads to the specified directory, or to `~/.cache/voxtral` if not specified.
    #[cfg(feature = "hub")]
    pub fn download(model_id: &str, cache_dir: Option<&Path>) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let api = Api::new().context("Failed to create HuggingFace API")?;
        let repo = api.model(model_id.to_string());

        let cache = cache_dir.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("voxtral")
        });

        std::fs::create_dir_all(&cache)?;

        // Download each file
        let weights = repo
            .get("consolidated.safetensors")
            .context("Failed to download weights")?;
        let config = repo
            .get("params.json")
            .context("Failed to download config")?;
        let tokenizer = repo
            .get("tekken.json")
            .context("Failed to download tokenizer")?;

        // Copy to cache directory
        let paths = Self::from_dir(&cache);
        std::fs::copy(&weights, &paths.weights)?;
        std::fs::copy(&config, &paths.config)?;
        std::fs::copy(&tokenizer, &paths.tokenizer)?;

        Ok(paths)
    }
}

/// Default Voxtral model ID on HuggingFace.
pub const VOXTRAL_MINI_4B_REALTIME: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";
