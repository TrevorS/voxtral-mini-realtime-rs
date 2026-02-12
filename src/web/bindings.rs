//! WASM bindings for Voxtral using Q4 GGUF weights and wgpu (WebGPU) backend.
//!
//! This module provides JavaScript-callable APIs for GPU-accelerated Q4 inference
//! in browsers with WebGPU support. The Q4 GGUF model is ~2GB, small enough to
//! load entirely in the browser.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor};

use crate::audio::chunk::{chunk_audio, needs_chunking, ChunkConfig};
use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;
use crate::gguf::loader::Q4ModelLoader;
use crate::gguf::model::Q4VoxtralModel;
use crate::models::time_embedding::TimeEmbedding;
use crate::tokenizer::VoxtralTokenizer;

type Backend = Wgpu<f32, i32>;

/// Device initialized by `initWgpuDevice()` — used by `VoxtralQ4` instances.
static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn wasm_log(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    let _ = msg;
}

/// Initialize panic hook for better error messages in browser console.
#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the WebGPU device asynchronously.
///
/// **Must** be called (and awaited) before creating `VoxtralQ4`.
///
/// Manually creates the wgpu device requesting the adapter's full limits
/// (especially `max_compute_invocations_per_workgroup`) instead of relying
/// on `init_setup_async` which may end up with WebGPU spec-defaults (256).
#[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = initWgpuDevice))]
pub async fn init_wgpu_device() {
    use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("No WebGPU adapter found");

    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    wasm_log(&format!(
        "[wgpu] Adapter: {} ({:?}), backend: {:?}",
        info.name, info.device_type, info.backend
    ));
    wasm_log(&format!(
        "[wgpu] Adapter limits: max_compute_invocations_per_workgroup={}, workgroup_size=({},{},{}), max_buffer_size={}",
        adapter_limits.max_compute_invocations_per_workgroup,
        adapter_limits.max_compute_workgroup_size_x,
        adapter_limits.max_compute_workgroup_size_y,
        adapter_limits.max_compute_workgroup_size_z,
        adapter_limits.max_buffer_size,
    ));

    // Request device with the adapter's full limits — not spec defaults
    let features = adapter.features() - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("voxtral-wgpu"),
            required_features: features,
            required_limits: adapter_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create WebGPU device");

    wasm_log(&format!(
        "[wgpu] Device created: max_compute_invocations_per_workgroup={}",
        device.limits().max_compute_invocations_per_workgroup,
    ));

    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: info.backend,
    };

    let wgpu_device = init_device(setup, RuntimeOptions::default());
    WGPU_DEVICE.set(wgpu_device).ok();
}

/// Q4 GGUF Voxtral transcription model for browser use.
///
/// Loads a Q4-quantized GGUF model (split into ≤512 MB shards to stay
/// under the browser's 2 GB `ArrayBuffer` limit) and provides a simple
/// API for transcribing audio via WebGPU.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct VoxtralQ4 {
    model: Option<Q4VoxtralModel>,
    tokenizer: Option<VoxtralTokenizer>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: TimeEmbedding,
    device: WgpuDevice,
    /// Sharded GGUF loading — each shard is kept as a separate Vec to stay
    /// under the WASM32 ~2 GB per-allocation limit.
    shard_bufs: Vec<Vec<u8>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl VoxtralQ4 {
    /// Create a new VoxtralQ4 instance.
    ///
    /// Call `initWgpuDevice()` first, then create this, then load GGUF weights.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let device = WGPU_DEVICE
            .get()
            .cloned()
            .unwrap_or_else(WgpuDevice::default);
        Self {
            model: None,
            tokenizer: None,
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            time_embed: TimeEmbedding::new(3072),
            device,
            shard_bufs: Vec::new(),
        }
    }

    /// Load model weights from a GGUF byte array and tokenizer JSON.
    ///
    /// # Arguments
    /// * `gguf_bytes` - The Q4 GGUF model as a Uint8Array (~2GB)
    /// * `tokenizer_json` - The tokenizer configuration as a string (from tekken.json)
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModel))]
    pub fn load_model(&mut self, gguf_bytes: &[u8], tokenizer_json: &str) -> Result<(), String> {
        // Load tokenizer from JSON string
        self.tokenizer = Some(
            VoxtralTokenizer::from_json(tokenizer_json)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        );

        // Load Q4 model from GGUF bytes
        let mut loader = Q4ModelLoader::from_bytes(gguf_bytes)
            .map_err(|e| format!("Failed to parse GGUF: {}", e))?;

        self.model = Some(
            loader
                .load(&self.device)
                .map_err(|e| format!("Failed to load Q4 model: {}", e))?,
        );

        Ok(())
    }

    /// Append a GGUF shard to the internal buffer.
    ///
    /// Call this once per shard (in order), then call `loadModelFromShards`
    /// to parse and load the assembled GGUF.  Each shard should be ≤512 MB
    /// so it fits in a single browser `ArrayBuffer`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = appendModelShard))]
    pub fn append_model_shard(&mut self, shard: &[u8]) {
        self.shard_bufs.push(shard.to_vec());
    }

    /// Parse the accumulated shards as a GGUF file and load the model.
    ///
    /// Must be called after all shards have been appended via `appendModelShard`.
    /// Uses two-phase loading: all Q4 tensors are loaded first, then the GGUF
    /// reader is dropped (freeing ~2.5 GB of shard data), and finally the
    /// token embeddings are dequantized to f32 (~1.5 GiB).
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModelFromShards))]
    pub fn load_model_from_shards(&mut self, tokenizer_json: &str) -> Result<(), String> {
        if self.shard_bufs.is_empty() {
            return Err("No shards appended. Call appendModelShard first.".into());
        }

        // Load tokenizer
        self.tokenizer = Some(
            VoxtralTokenizer::from_json(tokenizer_json)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        );

        // Phase 1: Load all Q4 tensors from GGUF (tok_embeddings stay as raw Q4 bytes)
        let shards = std::mem::take(&mut self.shard_bufs);
        let parts = {
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| format!("Failed to parse GGUF: {}", e))?;
            loader
                .load_deferred(&self.device)
                .map_err(|e| format!("Failed to load Q4 model: {}", e))?
            // loader (and its 2.5 GB shard data) dropped here
        };

        // Phase 2: Create Q4 tok_embeddings on GPU (~216 MB) with CPU copy for embed lookups
        self.model = Some(
            parts
                .finalize(&self.device)
                .map_err(|e| format!("Failed to finalize model: {}", e))?,
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
    /// Long audio is automatically chunked to stay within WebGPU's shared
    /// memory limits (max 1200 mel frames per chunk, matching the CLI).
    ///
    /// # Arguments
    /// * `audio` - Audio samples as a Float32Array (must be 16kHz mono)
    ///
    /// # Returns
    /// The transcribed text.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        let model = self
            .model
            .as_ref()
            .ok_or("Model not loaded. Call loadModel first.")?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or("Tokenizer not loaded. Call loadModel first.")?;

        // Normalize peak amplitude — Q4 can't resolve subtle mel features from
        // quiet audio, so we lift to 0.95 before mel computation.
        let mut audio_buf = AudioBuffer {
            samples: audio.to_vec(),
            sample_rate: 16000,
        };
        audio_buf.peak_normalize(0.95);

        // Chunk long audio to stay within WebGPU shared memory limits.
        let chunk_config = ChunkConfig::voxtral().with_max_frames(1200);
        let sample_chunks = if needs_chunking(audio_buf.samples.len(), &chunk_config) {
            chunk_audio(&audio_buf.samples, &chunk_config)
        } else {
            vec![crate::audio::AudioChunk {
                samples: audio_buf.samples.clone(),
                start_sample: 0,
                end_sample: audio_buf.samples.len(),
                index: 0,
                is_last: true,
            }]
        };

        let t_embed = self.time_embed.embed::<Backend>(6.0, &self.device);
        let mut texts = Vec::new();

        for chunk in &sample_chunks {
            let chunk_audio = AudioBuffer::new(chunk.samples.clone(), audio_buf.sample_rate);
            let mel_tensor = self.audio_to_mel(&chunk_audio)?;

            let audio_embeds = model.encode_audio(mel_tensor);
            let generated_tokens = self
                .decode_with_cache_async(model, audio_embeds, t_embed.clone())
                .await?;

            let text_tokens: Vec<u32> = generated_tokens
                .iter()
                .filter(|&&t| t >= 1000)
                .map(|&t| t as u32)
                .collect();

            let text = tokenizer
                .decode(&text_tokens)
                .map_err(|e| format!("Failed to decode tokens: {}", e))?;

            if !text.trim().is_empty() {
                texts.push(text.trim().to_string());
            }
        }

        Ok(texts.join(" "))
    }

    /// Convert an audio buffer to a mel spectrogram tensor.
    fn audio_to_mel(&self, audio: &AudioBuffer) -> Result<Tensor<Backend, 3>, String> {
        let padded = pad_audio(audio, &self.pad_config);
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

        if n_frames == 0 {
            return Err("Audio too short to produce mel frames".to_string());
        }

        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &self.device,
        ))
    }

    /// Get the expected sample rate for input audio.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getSampleRate))]
    pub fn get_sample_rate(&self) -> u32 {
        16000
    }
}

impl VoxtralQ4 {
    /// Async autoregressive decode loop for WASM compatibility.
    ///
    /// Uses `into_data_async().await` instead of `into_scalar().elem()` to
    /// avoid the synchronous `block_on()` that panics in the browser.
    async fn decode_with_cache_async(
        &self,
        model: &Q4VoxtralModel,
        audio_embeds: Tensor<Backend, 3>,
        t_embed: Tensor<Backend, 3>,
    ) -> Result<Vec<i32>, String> {
        let seq_len = audio_embeds.dims()[1];
        let d_model = audio_embeds.dims()[2];

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Ok(Vec::new());
        }

        let mut decoder_cache = model.decoder().create_cache();

        // Build prefix: BOS + 37 STREAMING_PAD
        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        // Embed prefix tokens (from CPU IDs — avoids GPU readback on WASM)
        let prefix_text_embeds = model
            .decoder()
            .embed_tokens_from_ids(&prefix, 1, PREFIX_LEN);

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

        // Get first prediction (async-safe)
        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = Tensor::<Backend, 3, Int>::into_data_async(first_pred)
            .await
            .map_err(|e| format!("Failed to read prediction tensor: {e}"))?
            .to_vec::<i32>()
            .map_err(|e| format!("Failed to extract prediction data: {e}"))?[0];

        let mut generated = prefix;
        generated.push(first_token);

        // Autoregressive generation with cache
        for pos in PREFIX_LEN + 1..seq_len {
            let new_token = generated[pos - 1];
            let text_embed = model.decoder().embed_tokens_from_ids(&[new_token], 1, 1);

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
            let next_token: i32 = Tensor::<Backend, 3, Int>::into_data_async(pred)
                .await
                .map_err(|e| format!("Failed to read prediction tensor: {e}"))?
                .to_vec::<i32>()
                .map_err(|e| format!("Failed to extract prediction data: {e}"))?[0];
            generated.push(next_token);
        }

        Ok(generated.into_iter().skip(PREFIX_LEN).collect())
    }
}

impl Default for VoxtralQ4 {
    fn default() -> Self {
        Self::new()
    }
}
