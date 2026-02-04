//! Tokenizer for Voxtral (Tekken format).
//!
//! Voxtral uses Mistral's Tekken tokenizer with 131,072 vocabulary size.
//! This is a custom format that stores tokens as base64-encoded bytes.

use anyhow::{Context, Result};
use base64::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Tekken tokenizer configuration from JSON.
#[derive(Debug, Deserialize)]
struct TekkenConfig {
    #[allow(dead_code)]
    pattern: String,
    #[allow(dead_code)]
    num_vocab_tokens: usize,
    default_vocab_size: usize,
    #[allow(dead_code)]
    default_num_special_tokens: usize,
    #[allow(dead_code)]
    version: String,
}

/// Single vocabulary entry.
#[derive(Debug, Deserialize)]
struct VocabEntry {
    rank: u32,
    #[serde(default)]
    token_bytes: Option<String>,
    #[serde(default)]
    token_str: Option<String>,
    #[serde(default)]
    is_control: bool,
}

/// Tekken JSON structure.
#[derive(Debug, Deserialize)]
struct TekkenJson {
    config: TekkenConfig,
    vocab: Vec<VocabEntry>,
}

/// Tekken tokenizer wrapper for Voxtral.
///
/// This implements decoding only - for ASR we receive token IDs from
/// the model and decode them to text.
pub struct VoxtralTokenizer {
    /// Token ID -> decoded bytes
    id_to_bytes: HashMap<u32, Vec<u8>>,
    /// Token ID -> string representation (for special tokens)
    id_to_str: HashMap<u32, String>,
    /// Vocabulary size
    vocab_size: usize,
}

impl VoxtralTokenizer {
    /// Load tokenizer from a `tekken.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open tokenizer file: {}", path.display()))?;
        let reader = BufReader::new(file);

        let tekken: TekkenJson = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse tekken.json: {}", path.display()))?;

        let vocab_size = tekken.config.default_vocab_size;
        let mut id_to_bytes = HashMap::with_capacity(tekken.vocab.len());
        let mut id_to_str = HashMap::with_capacity(tekken.vocab.len());

        for entry in tekken.vocab {
            let id = entry.rank;

            // For control/special tokens, just store the string representation
            if entry.is_control {
                if let Some(s) = &entry.token_str {
                    id_to_str.insert(id, s.clone());
                }
                continue;
            }

            // Decode token bytes from base64 if present
            if let Some(b64) = &entry.token_bytes {
                if let Ok(bytes) = BASE64_STANDARD.decode(b64) {
                    id_to_bytes.insert(id, bytes);
                    continue;
                }
            }

            // Fall back to UTF-8 encoding of token_str if present
            if let Some(s) = &entry.token_str {
                id_to_bytes.insert(id, s.as_bytes().to_vec());
            }
        }

        Ok(Self {
            id_to_bytes,
            id_to_str,
            vocab_size,
        })
    }

    /// Load tokenizer from a model directory.
    pub fn from_model_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let path = dir.as_ref().join("tekken.json");
        Self::from_file(path)
    }

    /// Decode token IDs to text.
    ///
    /// Concatenates the bytes for each token and decodes as UTF-8.
    /// Special/control tokens are skipped.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();

        for &id in ids {
            // Skip special/control tokens
            if self.id_to_str.contains_key(&id) {
                continue;
            }

            if let Some(token_bytes) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
            // Unknown tokens are silently skipped
        }

        // Decode accumulated bytes as UTF-8 (lossy for invalid sequences)
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> Option<String> {
        if let Some(s) = self.id_to_str.get(&id) {
            return Some(s.clone());
        }
        if let Some(bytes) = self.id_to_bytes.get(&id) {
            return Some(String::from_utf8_lossy(bytes).into_owned());
        }
        None
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tokenizer_path() -> PathBuf {
        PathBuf::from("models/voxtral/tekken.json")
    }

    #[test]
    fn test_load_tokenizer() {
        let path = tokenizer_path();
        if !path.exists() {
            println!("Skipping: tokenizer not downloaded");
            return;
        }

        let tokenizer = VoxtralTokenizer::from_file(&path).unwrap();
        assert_eq!(tokenizer.vocab_size(), 131072);
        println!(
            "Loaded tokenizer with {} vocab entries",
            tokenizer.id_to_bytes.len() + tokenizer.id_to_str.len()
        );
    }

    #[test]
    fn test_decode_simple() {
        let path = tokenizer_path();
        if !path.exists() {
            println!("Skipping: tokenizer not downloaded");
            return;
        }

        let tokenizer = VoxtralTokenizer::from_file(&path).unwrap();

        // Token 72 should be "H", token 101 should be "e", etc.
        // Let's check what some common ASCII tokens decode to
        for id in 65..=90 {
            // A-Z in ASCII
            if let Some(s) = tokenizer.decode_token(id) {
                println!("Token {}: {:?}", id, s);
            }
        }
    }
}
