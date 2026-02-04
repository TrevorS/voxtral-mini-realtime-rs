//! Tokenizer for Voxtral (Tekken format).
//!
//! Voxtral uses a Tekken tokenizer with 131,072 vocabulary size.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Tekken tokenizer wrapper for Voxtral.
pub struct VoxtralTokenizer {
    inner: Tokenizer,
}

impl VoxtralTokenizer {
    /// Load tokenizer from a `tekken.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .with_context(|| format!("Failed to load tokenizer from {}", path.display()))?;
        Ok(Self { inner })
    }

    /// Load tokenizer from a model directory.
    pub fn from_model_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let path = dir.as_ref().join("tekken.json");
        Self::from_file(path)
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to encode text")?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to decode tokens")
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_struct() {
        // Basic struct test - actual tokenizer tests require the tekken.json file
        assert_eq!(
            std::mem::size_of::<VoxtralTokenizer>(),
            std::mem::size_of::<Tokenizer>()
        );
    }
}
