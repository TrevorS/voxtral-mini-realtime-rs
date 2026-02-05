//! WASM bindings for Voxtral using wgpu (WebGPU) backend.
//!
//! This module provides JavaScript-callable APIs using the WGPU backend,
//! which enables GPU acceleration in browsers with WebGPU support.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;
use crate::models::loader::VoxtralModelLoader;
use crate::models::time_embedding::TimeEmbedding;
use crate::models::voxtral::VoxtralModel;
use crate::tokenizer::VoxtralTokenizer;

type Backend = Wgpu<f32, i32>;

/// Initialize panic hook for better error messages in browser console.
#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Voxtral transcription model for browser use with WebGPU acceleration.
///
/// This class wraps the full Voxtral model and provides a simple API
/// for transcribing audio in the browser using GPU acceleration.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct VoxtralGpu {
    model: Option<VoxtralModel<Backend>>,
    tokenizer: Option<VoxtralTokenizer>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: TimeEmbedding,
    device: WgpuDevice,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl VoxtralGpu {
    /// Create a new VoxtralGpu instance.
    ///
    /// Call `loadModel` to load weights before transcribing.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            model: None,
            tokenizer: None,
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            time_embed: TimeEmbedding::new(3072),
            device: WgpuDevice::default(),
        }
    }

    /// Load model weights from a SafeTensors byte array.
    ///
    /// # Arguments
    /// * `model_bytes` - The model weights as a Uint8Array (from consolidated.safetensors)
    /// * `tokenizer_json` - The tokenizer configuration as a string (from tekken.json)
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModel))]
    pub fn load_model(&mut self, model_bytes: &[u8], tokenizer_json: &str) -> Result<(), String> {
        // Load tokenizer from JSON string
        self.tokenizer = Some(
            VoxtralTokenizer::from_json(tokenizer_json)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        );

        // Load model from bytes
        let loader = VoxtralModelLoader::from_bytes(model_bytes.to_vec())
            .map_err(|e| format!("Failed to create model loader: {}", e))?;

        self.model = Some(
            loader
                .load::<Backend>(&self.device)
                .map_err(|e| format!("Failed to load model: {}", e))?,
        );

        Ok(())
    }

    /// Check if the model is loaded and ready for transcription.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = isReady))]
    pub fn is_ready(&self) -> bool {
        self.model.is_some() && self.tokenizer.is_some()
    }

    /// Transcribe audio to text.
    ///
    /// # Arguments
    /// * `audio` - Audio samples as a Float32Array (must be 16kHz mono)
    ///
    /// # Returns
    /// The transcribed text.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        let model = self
            .model
            .as_ref()
            .ok_or("Model not loaded. Call loadModel first.")?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or("Tokenizer not loaded. Call loadModel first.")?;

        // Create audio buffer (assumes 16kHz input)
        let audio_buf = AudioBuffer {
            samples: audio.to_vec(),
            sample_rate: 16000,
        };

        // Apply padding for streaming alignment
        let padded = pad_audio(&audio_buf, &self.pad_config);

        // Extract mel spectrogram
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

        if n_frames == 0 {
            return Ok(String::new());
        }

        // Transpose to [n_mels, n_frames] and create tensor
        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &self.device,
        );

        // Run encoder
        let audio_embeds = model.encode_audio(mel_tensor);

        // Create time embedding (t=6 for 480ms delay)
        let t_embed = self.time_embed.embed::<Backend>(6.0, &self.device);

        // Run transcription
        let generated_tokens = self.transcribe_with_cache(model, audio_embeds, t_embed);

        // Filter control tokens and decode
        let text_tokens: Vec<u32> = generated_tokens
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();

        tokenizer
            .decode(&text_tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e))
    }

    /// Internal: Run transcription with KV cache optimization.
    fn transcribe_with_cache(
        &self,
        model: &VoxtralModel<Backend>,
        audio_embeds: Tensor<Backend, 3>,
        t_embed: Tensor<Backend, 3>,
    ) -> Vec<i32> {
        use burn::prelude::ElementConversion;
        use burn::tensor::Int;

        let seq_len = audio_embeds.dims()[1];
        let d_model = audio_embeds.dims()[2];

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        // Check if audio is long enough
        if seq_len < PREFIX_LEN {
            return Vec::new();
        }

        // Create KV cache
        let mut decoder_cache = model.create_decoder_cache();

        // Build prefix: BOS + 37 STREAMING_PAD
        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        // Embed prefix tokens
        let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
            &self.device,
        );
        let prefix_text_embeds = model.decoder().embed_tokens(prefix_tensor);

        // Slice audio embeddings for prefix
        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);

        // Combine and run forward
        let prefix_inputs = prefix_audio + prefix_text_embeds;
        let hidden = model.decoder().forward_hidden_with_cache(
            prefix_inputs,
            t_embed.clone(),
            &mut decoder_cache,
        );
        let logits = model.decoder().lm_head(hidden);

        // Get first prediction
        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = first_pred.into_scalar().elem();

        let mut generated = prefix.clone();
        generated.push(first_token);

        // Autoregressive generation with cache
        for pos in PREFIX_LEN + 1..seq_len {
            let new_token = generated[pos - 1];
            let token_tensor = Tensor::<Backend, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![new_token], [1, 1]),
                &self.device,
            );
            let text_embed = model.decoder().embed_tokens(token_tensor);

            let audio_pos = audio_embeds
                .clone()
                .slice([0..1, (pos - 1)..pos, 0..d_model]);

            let input = audio_pos + text_embed;
            let hidden = model.decoder().forward_hidden_with_cache(
                input,
                t_embed.clone(),
                &mut decoder_cache,
            );
            let logits = model.decoder().lm_head(hidden);

            let pred = logits.argmax(2);
            let next_token: i32 = pred.into_scalar().elem();
            generated.push(next_token);
        }

        generated.into_iter().skip(PREFIX_LEN).collect()
    }

    /// Get the expected sample rate for input audio.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getSampleRate))]
    pub fn get_sample_rate(&self) -> u32 {
        16000
    }
}

impl Default for VoxtralGpu {
    fn default() -> Self {
        Self::new()
    }
}
