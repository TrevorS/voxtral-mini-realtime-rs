//! WASM bindings for Voxtral using wgpu (WebGPU) backend.
//!
//! This module provides JavaScript-callable APIs using the Wgpu backend,
//! which enables GPU-accelerated inference in browsers with WebGPU support.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(target_family = "wasm")]
use js_sys::Uint8Array;

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

/// Initialize the WebGPU device asynchronously.
///
/// **Must** be called (and awaited) before creating `Voxtral` or `VoxtralPhased`.
/// WebGPU adapter/device creation is inherently async; without this
/// pre-initialization the first tensor operation triggers a synchronous
/// `block_on()` that panics in the browser.
#[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = initWgpuDevice))]
pub async fn init_wgpu_device() {
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::{init_setup_async, RuntimeOptions, WgpuDevice};

    let device = WgpuDevice::default();
    init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
}

/// Voxtral transcription model for browser use.
///
/// This class wraps the full Voxtral model and provides a simple API
/// for transcribing audio in the browser.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Voxtral {
    model: Option<VoxtralModel<Backend>>,
    tokenizer: Option<VoxtralTokenizer>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: TimeEmbedding,
    device: WgpuDevice,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Voxtral {
    /// Create a new Voxtral instance.
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

impl Default for Voxtral {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for loading large models in chunks.
///
/// This allows loading 8GB+ models by allocating WASM memory first,
/// then copying chunks in, bypassing JS ArrayBuffer size limits.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct ModelLoader {
    buffer: Vec<u8>,
    offset: usize,
    total_size: usize,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl ModelLoader {
    /// Create a new loader with pre-allocated buffer for the model.
    ///
    /// # Arguments
    /// * `total_size` - Total size of the model in bytes
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new(total_size: usize) -> Result<ModelLoader, String> {
        // Pre-allocate the full buffer in WASM memory
        let buffer = vec![0u8; total_size];
        Ok(ModelLoader {
            buffer,
            offset: 0,
            total_size,
        })
    }

    /// Write a chunk of model data at the current offset.
    ///
    /// # Arguments
    /// * `chunk` - A Uint8Array chunk to write
    ///
    /// # Returns
    /// The current offset after writing (for progress tracking)
    #[cfg(target_family = "wasm")]
    #[wasm_bindgen(js_name = writeChunk)]
    pub fn write_chunk(&mut self, chunk: &Uint8Array) -> Result<usize, String> {
        let chunk_len = chunk.length() as usize;
        if self.offset + chunk_len > self.total_size {
            return Err(format!(
                "Chunk would exceed buffer: offset={}, chunk={}, total={}",
                self.offset, chunk_len, self.total_size
            ));
        }

        // Copy directly from JS Uint8Array into our WASM Vec
        chunk.copy_to(&mut self.buffer[self.offset..self.offset + chunk_len]);
        self.offset += chunk_len;

        Ok(self.offset)
    }

    /// Check if the buffer is fully loaded.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = isComplete))]
    pub fn is_complete(&self) -> bool {
        self.offset >= self.total_size
    }

    /// Get the current loading progress as a percentage.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getProgress))]
    pub fn get_progress(&self) -> f64 {
        (self.offset as f64 / self.total_size as f64) * 100.0
    }

    /// Finalize loading and return the buffer for model construction.
    /// This consumes the loader.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = finalize))]
    pub fn finalize(self) -> Result<Vec<u8>, String> {
        if self.offset < self.total_size {
            return Err(format!(
                "Buffer incomplete: {} of {} bytes loaded",
                self.offset, self.total_size
            ));
        }
        Ok(self.buffer)
    }
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Voxtral {
    /// Load model from a completed ModelLoader (for chunked loading).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModelFromLoader))]
    pub fn load_model_from_loader(
        &mut self,
        loader: ModelLoader,
        tokenizer_json: &str,
    ) -> Result<(), String> {
        let bytes = loader.finalize()?;
        self.load_model_internal(bytes, tokenizer_json)
    }
}

impl Voxtral {
    /// Internal method to load from owned bytes.
    fn load_model_internal(
        &mut self,
        model_bytes: Vec<u8>,
        tokenizer_json: &str,
    ) -> Result<(), String> {
        // Load tokenizer from JSON string
        self.tokenizer = Some(
            VoxtralTokenizer::from_json(tokenizer_json)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        );

        // Load model from bytes
        let loader = VoxtralModelLoader::from_bytes(model_bytes)
            .map_err(|e| format!("Failed to create model loader: {}", e))?;

        self.model = Some(
            loader
                .load::<Backend>(&self.device)
                .map_err(|e| format!("Failed to load model: {}", e))?,
        );

        Ok(())
    }
}

/// Lightweight mel spectrogram extractor for hybrid client-server deployment.
///
/// This class runs in the browser with minimal memory (~50KB) and produces
/// mel spectrograms that can be sent to a server for full model inference.
///
/// # Usage
/// 1. Create `MelClient` in browser
/// 2. Capture audio via Web Audio API
/// 3. Call `extractMel` to get mel spectrogram
/// 4. Send mel bytes to server via WebSocket/HTTP
/// 5. Server runs full Voxtral model and returns text
///
/// This approach works within wasm32's 4GB limit since only the mel
/// extraction code is loaded (~50KB), not the 8GB model.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct MelClient {
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl MelClient {
    /// Create a new mel spectrogram client.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
        }
    }

    /// Extract mel spectrogram from audio.
    ///
    /// # Arguments
    /// * `audio` - Audio samples as Float32Array (must be 16kHz mono)
    ///
    /// # Returns
    /// Mel spectrogram as Float32Array, shape [n_mels * n_frames].
    /// First 2 values are n_mels and n_frames for reconstruction.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = extractMel))]
    pub fn extract_mel(&self, audio: &[f32]) -> Vec<f32> {
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
        let n_mels = if n_frames > 0 { mel[0].len() } else { 128 };

        // Pack as [n_mels, n_frames, mel_data...]
        let mut result = Vec::with_capacity(2 + n_mels * n_frames);
        result.push(n_mels as f32);
        result.push(n_frames as f32);

        // Transpose to [n_mels, n_frames] for the model
        for mel_idx in 0..n_mels {
            for frame_idx in 0..n_frames {
                result.push(mel[frame_idx][mel_idx]);
            }
        }

        result
    }

    /// Get mel spectrogram dimensions for a given audio length.
    ///
    /// # Arguments
    /// * `audio_samples` - Number of audio samples at 16kHz
    ///
    /// # Returns
    /// Object with n_mels and n_frames
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getMelDimensions))]
    pub fn get_mel_dimensions(&self, audio_samples: usize) -> MelDimensions {
        // After padding and mel extraction
        let padded_samples = audio_samples + self.pad_config.left_pad_samples();
        let n_frames = (padded_samples / 160) + 1; // hop_length=160
        MelDimensions {
            n_mels: 128,
            n_frames,
        }
    }

    /// Get the expected sample rate for input audio.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getSampleRate))]
    pub fn get_sample_rate(&self) -> u32 {
        16000
    }
}

impl Default for MelClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Mel spectrogram dimensions.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct MelDimensions {
    /// Number of mel frequency bins (always 128 for Voxtral)
    pub n_mels: usize,
    /// Number of time frames
    pub n_frames: usize,
}

// ---------------------------------------------------------------------------
// Phased inference API for wasm32 deployment
// ---------------------------------------------------------------------------

use burn::nn::Embedding;

use crate::models::adapter::{reshape_encoder_output, AudioLanguageAdapter};
use crate::models::decoder::{DecoderParts, LanguageModel, LanguageModelConfig};
use crate::models::encoder::AudioEncoder;
use crate::models::layers::{DecoderLayer, RmsNorm};
use crate::models::voxtral::VoxtralModelConfig;

/// Phased inference for wasm32 browser deployment.
///
/// Loads encoder and decoder separately so they never coexist in memory.
/// Peak memory = max(encoder ~0.7 GB, decoder ~3.5 GB) ≈ 3.6 GB, fitting
/// within wasm32's 4 GiB limit.
///
/// ## Usage from JavaScript
/// ```js
/// const voxtral = new VoxtralPhased();
/// voxtral.loadTokenizer(tokenizerJson);
///
/// // Phase 1: Encode audio
/// voxtral.loadEncoderShard(encoderBytes);
/// voxtral.loadAdapterShard(adapterBytes);
/// voxtral.encodeAudio(audioSamples);
/// voxtral.freeEncoder();
///
/// // Phase 2: Decode tokens
/// voxtral.loadDecoderShard(decoderBytes);
/// const text = voxtral.transcribe();
/// voxtral.freeDecoder();
/// ```
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct VoxtralPhased {
    config: VoxtralModelConfig,
    encoder: Option<AudioEncoder<Backend>>,
    adapter: Option<AudioLanguageAdapter<Backend>>,
    decoder: Option<LanguageModel<Backend>>,
    audio_embeds_data: Option<Vec<f32>>,
    audio_embeds_shape: Option<[usize; 3]>,
    tokenizer: Option<VoxtralTokenizer>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: TimeEmbedding,
    device: WgpuDevice,
    // Streamed decoder accumulators (per-layer loading)
    decoder_embeddings: Option<Embedding<Backend>>,
    decoder_layers: Vec<Option<DecoderLayer<Backend>>>,
    decoder_norm: Option<RmsNorm<Backend>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl VoxtralPhased {
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let config = VoxtralModelConfig::voxtral();
        let n_layers = config.decoder.n_layers;
        Self {
            config,
            encoder: None,
            adapter: None,
            decoder: None,
            audio_embeds_data: None,
            audio_embeds_shape: None,
            tokenizer: None,
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            time_embed: TimeEmbedding::new(3072),
            device: WgpuDevice::default(),
            decoder_embeddings: None,
            decoder_layers: (0..n_layers).map(|_| None).collect(),
            decoder_norm: None,
        }
    }

    /// Load the tokenizer from JSON.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadTokenizer))]
    pub fn load_tokenizer(&mut self, tokenizer_json: &str) -> Result<(), String> {
        self.tokenizer = Some(
            VoxtralTokenizer::from_json(tokenizer_json)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        );
        Ok(())
    }

    /// Load encoder shard from SafeTensors bytes (decompressed .safetensors.gz).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadEncoderShard))]
    pub fn load_encoder_shard(&mut self, bytes: &[u8]) -> Result<(), String> {
        let loader = VoxtralModelLoader::from_bytes(bytes.to_vec())
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        self.encoder = Some(
            loader
                .load_encoder::<Backend>(&self.device)
                .map_err(|e| format!("Failed to load encoder: {}", e))?,
        );
        Ok(())
    }

    /// Load adapter shard from SafeTensors bytes (decompressed .safetensors.gz).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadAdapterShard))]
    pub fn load_adapter_shard(&mut self, bytes: &[u8]) -> Result<(), String> {
        let loader = VoxtralModelLoader::from_bytes(bytes.to_vec())
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        self.adapter = Some(
            loader
                .load_adapter::<Backend>(&self.device)
                .map_err(|e| format!("Failed to load adapter: {}", e))?,
        );
        Ok(())
    }

    /// Run encoder phase: mel extraction + encoder + reshape + adapter.
    ///
    /// Stores the audio_embeds internally. Call freeEncoder() after this
    /// to release encoder memory before loading the decoder.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = encodeAudio))]
    pub async fn encode_audio(&mut self, audio: &[f32]) -> Result<(), String> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or("Encoder not loaded. Call loadEncoderShard first.")?;
        let adapter = self
            .adapter
            .as_ref()
            .ok_or("Adapter not loaded. Call loadAdapterShard first.")?;

        // Create audio buffer and extract mel
        let audio_buf = AudioBuffer {
            samples: audio.to_vec(),
            sample_rate: 16000,
        };
        let padded = pad_audio(&audio_buf, &self.pad_config);
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 128 };

        if n_frames == 0 {
            return Err("Audio too short for mel extraction".to_string());
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

        // Run encoder + reshape + adapter
        let encoder_out = encoder.forward(mel_tensor, 0);
        let reshaped = reshape_encoder_output(encoder_out, self.config.reshape_factor);
        let audio_embeds = adapter.forward(reshaped);

        let shape = audio_embeds.dims();
        let data = audio_embeds
            .into_data_async()
            .await
            .map_err(|e| format!("Failed to read audio_embeds: {}", e))?
            .to_vec::<f32>()
            .map_err(|e| format!("Failed to extract audio_embeds: {}", e))?;

        self.audio_embeds_data = Some(data);
        self.audio_embeds_shape = Some(shape);
        Ok(())
    }

    /// Free encoder and adapter memory after encoding.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = freeEncoder))]
    pub fn free_encoder(&mut self) {
        self.encoder = None;
        self.adapter = None;
    }

    /// Load decoder shard from SafeTensors bytes (decompressed .safetensors.gz).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadDecoderShard))]
    pub fn load_decoder_shard(&mut self, bytes: &[u8]) -> Result<(), String> {
        let loader = VoxtralModelLoader::from_bytes(bytes.to_vec())
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        self.decoder = Some(
            loader
                .load_decoder_with_vocab::<Backend>(&self.device, None)
                .map_err(|e| format!("Failed to load decoder: {}", e))?,
        );
        Ok(())
    }

    // --- Streamed decoder loading (per-layer) ---

    /// Load decoder token embeddings from SafeTensors bytes.
    ///
    /// Part of the streamed decoder loading API. Call this first, then
    /// loadDecoderLayer for each layer, then loadDecoderNorm.
    ///
    /// Parses SafeTensors directly from borrowed bytes to avoid doubling
    /// peak WASM memory with a Vec copy.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadDecoderEmbeddings))]
    pub fn load_decoder_embeddings(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.decoder_embeddings = Some(
            VoxtralModelLoader::tok_embeddings_from_raw::<Backend>(bytes, &self.device)
                .map_err(|e| format!("Failed to load decoder embeddings: {}", e))?,
        );
        Ok(())
    }

    /// Load a single decoder transformer layer from SafeTensors bytes.
    ///
    /// `index` is 0-based and must be < n_layers (26 for Voxtral).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadDecoderLayer))]
    pub fn load_decoder_layer(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        if index >= self.decoder_layers.len() {
            return Err(format!(
                "Layer index {} out of range (model has {} layers)",
                index,
                self.decoder_layers.len()
            ));
        }

        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        let config = LanguageModelConfig::voxtral();
        self.decoder_layers[index] = Some(
            VoxtralModelLoader::decoder_layer_from_st::<Backend>(&st, index, &config, &self.device)
                .map_err(|e| format!("Failed to load decoder layer {}: {}", index, e))?,
        );
        Ok(())
    }

    /// Load decoder final norm from SafeTensors bytes and assemble the decoder.
    ///
    /// This is the last step of streamed decoder loading. After this call,
    /// the full decoder is assembled and ready for transcription.
    /// All accumulated parts (embeddings, layers) are consumed.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadDecoderNorm))]
    pub fn load_decoder_norm(&mut self, bytes: &[u8]) -> Result<(), String> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        let norm = VoxtralModelLoader::final_norm_from_st::<Backend>(&st, &self.device)
            .map_err(|e| format!("Failed to load decoder norm: {}", e))?;

        // Assemble the full decoder from accumulated parts
        let tok_embeddings = self
            .decoder_embeddings
            .take()
            .ok_or("Embeddings not loaded. Call loadDecoderEmbeddings first.")?;

        let mut layers = Vec::with_capacity(self.decoder_layers.len());
        for (i, slot) in self.decoder_layers.iter_mut().enumerate() {
            layers.push(slot.take().ok_or(format!(
                "Layer {} not loaded. Call loadDecoderLayer first.",
                i
            ))?);
        }

        let rope = self.config.decoder.init_rope::<Backend>(&self.device);

        let parts = DecoderParts {
            tok_embeddings,
            layers,
            norm,
            d_model: self.config.decoder.d_model,
        };

        self.decoder = Some(LanguageModel::from_parts(parts, rope));
        Ok(())
    }

    /// Run transcription using stored audio_embeds and loaded decoder.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn transcribe(&self) -> Result<String, String> {
        let decoder = self
            .decoder
            .as_ref()
            .ok_or("Decoder not loaded. Call loadDecoderShard first.")?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or("Tokenizer not loaded. Call loadTokenizer first.")?;
        let embeds_data = self
            .audio_embeds_data
            .as_ref()
            .ok_or("No audio encoded. Call encodeAudio first.")?;
        let embeds_shape = self
            .audio_embeds_shape
            .ok_or("No audio encoded. Call encodeAudio first.")?;

        // Reconstruct audio_embeds tensor
        let audio_embeds: Tensor<Backend, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(embeds_data.clone(), embeds_shape),
            &self.device,
        );

        let t_embed = self.time_embed.embed::<Backend>(6.0, &self.device);

        // Run autoregressive decoding
        let generated = self.decode_with_cache(decoder, audio_embeds, t_embed).await;

        // Filter control tokens and decode
        let text_tokens: Vec<u32> = generated
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();

        tokenizer
            .decode(&text_tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e))
    }

    /// Free decoder memory after transcription.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = freeDecoder))]
    pub fn free_decoder(&mut self) {
        self.decoder = None;
        self.audio_embeds_data = None;
        self.audio_embeds_shape = None;
        // Also clear any streaming accumulator state
        self.decoder_embeddings = None;
        for slot in &mut self.decoder_layers {
            *slot = None;
        }
        self.decoder_norm = None;
    }

    /// Check current state.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getState))]
    pub fn get_state(&self) -> String {
        let mut parts = Vec::new();
        if self.tokenizer.is_some() {
            parts.push("tokenizer");
        }
        if self.encoder.is_some() {
            parts.push("encoder");
        }
        if self.adapter.is_some() {
            parts.push("adapter");
        }
        if self.audio_embeds_data.is_some() {
            parts.push("audio_embeds");
        }
        if self.decoder.is_some() {
            parts.push("decoder");
        }
        if parts.is_empty() {
            "empty".to_string()
        } else {
            parts.join(", ")
        }
    }
}

impl VoxtralPhased {
    /// Autoregressive decode using a standalone LanguageModel.
    async fn decode_with_cache(
        &self,
        decoder: &LanguageModel<Backend>,
        audio_embeds: Tensor<Backend, 3>,
        t_embed: Tensor<Backend, 3>,
    ) -> Vec<i32> {
        use burn::tensor::Int;

        let seq_len = audio_embeds.dims()[1];
        let d_model = audio_embeds.dims()[2];

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Vec::new();
        }

        let mut cache = decoder.create_cache();

        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        let prefix_tensor = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
            &self.device,
        );
        let prefix_text_embeds = decoder.embed_tokens(prefix_tensor);
        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);
        let prefix_inputs = prefix_audio + prefix_text_embeds;

        let hidden = decoder.forward_hidden_with_cache(prefix_inputs, t_embed.clone(), &mut cache);
        let logits = decoder.lm_head(hidden);

        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = first_pred
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<i32>()
            .unwrap()[0];

        let mut generated = prefix;
        generated.push(first_token);

        for pos in PREFIX_LEN + 1..seq_len {
            let new_token = generated[pos - 1];
            let token_tensor = Tensor::<Backend, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![new_token], [1, 1]),
                &self.device,
            );
            let text_embed = decoder.embed_tokens(token_tensor);
            let audio_pos = audio_embeds
                .clone()
                .slice([0..1, (pos - 1)..pos, 0..d_model]);
            let input = audio_pos + text_embed;

            let hidden = decoder.forward_hidden_with_cache(input, t_embed.clone(), &mut cache);
            let logits = decoder.lm_head(hidden);

            let pred = logits.argmax(2);
            let next_token: i32 = pred
                .into_data_async()
                .await
                .unwrap()
                .to_vec::<i32>()
                .unwrap()[0];
            generated.push(next_token);
        }

        generated.into_iter().skip(PREFIX_LEN).collect()
    }
}

impl Default for VoxtralPhased {
    fn default() -> Self {
        Self::new()
    }
}
