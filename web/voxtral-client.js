/**
 * Voxtral Client - Browser API for speech transcription.
 *
 * Handles WebWorker communication and Web Audio API for microphone input.
 *
 * Usage:
 *   const client = new VoxtralClient();
 *   await client.init();
 *   await client.loadModel(modelBytes, tokenizerJson);
 *
 *   // Transcribe file
 *   const text = await client.transcribeFile(audioFile);
 *
 *   // Or use microphone
 *   await client.startMicrophone();
 *   // ... recording ...
 *   const text = await client.stopAndTranscribe();
 */

export class VoxtralClient {
    static CACHE_NAME = 'voxtral-weights';

    static async clearCache() {
        return caches.delete(VoxtralClient.CACHE_NAME);
    }

    static async getCachedShardCount() {
        try {
            const cache = await caches.open(VoxtralClient.CACHE_NAME);
            const keys = await cache.keys();
            return keys.length;
        } catch {
            return 0;
        }
    }

    constructor() {
        this.worker = null;
        this.ready = false;
        this.modelLoaded = false;
        this.pendingResolve = null;
        this.pendingReject = null;
        this.onProgress = null;

        // Microphone state
        this.audioContext = null;
        this.mediaStream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];

        // Audio processing
        this.targetSampleRate = 16000;
    }

    /**
     * Initialize the WebWorker and WASM module.
     */
    async init() {
        return new Promise((resolve, reject) => {
            // Use module worker for ES module support
            this.worker = new Worker('./worker.js', { type: 'module' });

            this.worker.onmessage = (e) => this._handleMessage(e);
            this.worker.onerror = (e) => {
                reject(new Error(`Worker error: ${e.message}`));
            };

            this.pendingResolve = () => {
                this.ready = true;
                resolve();
            };
            this.pendingReject = reject;

            this.worker.postMessage({ type: 'init' });
        });
    }

    /**
     * Load model weights and tokenizer.
     * @param {ArrayBuffer|Uint8Array} modelBytes - Model weights
     * @param {string} tokenizerJson - Tokenizer JSON string
     */
    async loadModel(modelBytes, tokenizerJson) {
        if (!this.ready) {
            throw new Error('Client not initialized. Call init() first.');
        }

        return new Promise((resolve, reject) => {
            this.pendingResolve = () => {
                this.modelLoaded = true;
                resolve();
            };
            this.pendingReject = reject;

            // Transfer the ArrayBuffer for efficiency
            const bytes = modelBytes instanceof Uint8Array
                ? modelBytes.buffer
                : modelBytes;

            this.worker.postMessage(
                { type: 'loadModel', modelBytes: bytes, tokenizerJson },
                [bytes]
            );
        });
    }

    /**
     * Check if the model is ready for transcription.
     */
    isReady() {
        return this.ready && this.modelLoaded;
    }

    /**
     * Transcribe audio samples.
     * @param {Float32Array} audio - 16kHz mono audio samples
     * @returns {Promise<string>} Transcribed text
     */
    async transcribe(audio) {
        if (!this.isReady()) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;

            // Transfer the buffer for efficiency
            this.worker.postMessage(
                { type: 'transcribe', audio },
                [audio.buffer]
            );
        });
    }

    /**
     * Transcribe an audio file.
     * @param {File|Blob} file - Audio file (WAV, MP3, etc.)
     * @returns {Promise<string>} Transcribed text
     */
    async transcribeFile(file) {
        const audio = await this._decodeAudioFile(file);
        return this.transcribe(audio);
    }

    /**
     * Start microphone recording.
     * @returns {Promise<void>}
     */
    async startMicrophone() {
        // Request microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: this.targetSampleRate,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        // Create audio context for later processing
        this.audioContext = new AudioContext({ sampleRate: this.targetSampleRate });

        // Record using MediaRecorder
        this.recordedChunks = [];
        this.mediaRecorder = new MediaRecorder(this.mediaStream, {
            mimeType: this._getSupportedMimeType()
        });

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                this.recordedChunks.push(e.data);
            }
        };

        this.mediaRecorder.start(100); // Collect data every 100ms
    }

    /**
     * Stop microphone and transcribe the recording.
     * @returns {Promise<string>} Transcribed text
     */
    async stopAndTranscribe() {
        if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
            throw new Error('Microphone not recording.');
        }

        // Stop recording and wait for final data
        const audioBlob = await new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, {
                    type: this.mediaRecorder.mimeType
                });
                resolve(blob);
            };
            this.mediaRecorder.stop();
        });

        // Clean up microphone
        this._stopMicrophone();

        // Decode and transcribe
        const audio = await this._decodeAudioFile(audioBlob);
        return this.transcribe(audio);
    }

    /**
     * Cancel microphone recording without transcribing.
     */
    cancelMicrophone() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        this._stopMicrophone();
    }

    /**
     * Check if currently recording.
     */
    isRecording() {
        return this.mediaRecorder && this.mediaRecorder.state === 'recording';
    }

    /**
     * Set progress callback.
     * @param {function(string, number?)} callback - Called with (stage, percent?)
     */
    setProgressCallback(callback) {
        this.onProgress = callback;
    }

    /**
     * Clean up resources.
     */
    dispose() {
        this._stopMicrophone();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.ready = false;
        this.modelLoaded = false;
    }

    // Private methods

    _handleMessage(e) {
        const { type, ...data } = e.data;

        switch (type) {
            case 'ready':
            case 'modelLoaded':
            case 'tokenizerLoaded':
            case 'encoderLoaded':
            case 'audioEncoded':
            case 'encoderFreed':
            case 'decoderLoaded':
            case 'decoderFreed':
            case 'decoderEmbeddingsLoaded':
            case 'decoderLayerLoaded':
                if (this.pendingResolve) {
                    this.pendingResolve();
                    this.pendingResolve = null;
                    this.pendingReject = null;
                }
                break;

            case 'transcription':
                if (this.pendingResolve) {
                    this.pendingResolve(data.text);
                    this.pendingResolve = null;
                    this.pendingReject = null;
                }
                break;

            case 'error':
                if (this.pendingReject) {
                    this.pendingReject(new Error(data.message));
                    this.pendingResolve = null;
                    this.pendingReject = null;
                }
                break;

            case 'progress':
                if (this.onProgress) {
                    this.onProgress(data.stage, data.percent);
                }
                break;
        }
    }

    async _decodeAudioFile(file) {
        // Read file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();

        // Decode to AudioBuffer
        const audioContext = new AudioContext({ sampleRate: this.targetSampleRate });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Get mono channel (mix down if stereo)
        let samples;
        if (audioBuffer.numberOfChannels === 1) {
            samples = audioBuffer.getChannelData(0);
        } else {
            // Mix stereo to mono
            const left = audioBuffer.getChannelData(0);
            const right = audioBuffer.getChannelData(1);
            samples = new Float32Array(left.length);
            for (let i = 0; i < left.length; i++) {
                samples[i] = (left[i] + right[i]) / 2;
            }
        }

        // Resample if needed
        if (audioBuffer.sampleRate !== this.targetSampleRate) {
            samples = this._resample(samples, audioBuffer.sampleRate, this.targetSampleRate);
        }

        await audioContext.close();
        return samples;
    }

    _resample(samples, fromRate, toRate) {
        const ratio = fromRate / toRate;
        const newLength = Math.floor(samples.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, samples.length - 1);
            const t = srcIndex - srcIndexFloor;

            // Linear interpolation
            result[i] = samples[srcIndexFloor] * (1 - t) + samples[srcIndexCeil] * t;
        }

        return result;
    }

    _stopMicrophone() {
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        this.mediaRecorder = null;
        this.recordedChunks = [];
    }

    /**
     * Fetch a gzipped URL and decompress it, using Cache API when available.
     * Caches the decompressed result so subsequent loads skip fetch + decompress.
     * @param {string} url - URL to a .gz file
     * @returns {Promise<ArrayBuffer>} Decompressed bytes
     */
    async _fetchGzipped(url) {
        // Check cache first
        const cache = await caches.open(VoxtralClient.CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) {
            return await cached.arrayBuffer();
        }

        // Fetch and decompress
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Fetch failed ${url}: ${resp.status}`);
        const ds = new DecompressionStream('gzip');
        const reader = resp.body.pipeThrough(ds).getReader();
        const chunks = [];
        let total = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            total += value.length;
        }
        const result = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }

        // Cache decompressed result (clone since buffer may be transferred later)
        try {
            await cache.put(url, new Response(result.buffer.slice(0)));
        } catch (e) {
            console.warn('Failed to cache shard:', e);
        }

        return result.buffer;
    }

    // --- Phased loading API ---

    /**
     * Transcribe audio using phased loading (fits wasm32 4GiB limit).
     *
     * Fetches encoder and decoder shards sequentially, keeping peak memory
     * at ~3.6 GB instead of ~4.2 GB.
     *
     * If `urls.manifest` is provided, the manifest is fetched and used to
     * determine the decoder format. When `decoder_format` is `"streamed"`,
     * the decoder is loaded as 28 individual per-layer shards for lower
     * peak memory. Otherwise, falls back to the monolithic decoder URL.
     *
     * @param {Float32Array} audio - 16kHz mono audio samples
     * @param {object} urls - Shard URLs
     * @param {string} urls.tokenizer - URL to tekken.json
     * @param {string} urls.encoder - URL to encoder.safetensors.gz
     * @param {string} urls.adapter - URL to adapter.safetensors.gz
     * @param {string} [urls.decoder] - URL to decoder.safetensors.gz (monolithic)
     * @param {string} [urls.manifest] - URL to manifest.json (enables streamed loading)
     * @param {string} [urls.baseUrl] - Base URL for streamed shard files
     * @returns {Promise<string>} Transcribed text
     */
    async transcribePhased(audio, urls) {
        if (!this.ready) {
            throw new Error('Client not initialized. Call init() first.');
        }

        // Load tokenizer
        if (this.onProgress) this.onProgress('Fetching tokenizer...', 0);
        const tokenizerResp = await fetch(urls.tokenizer);
        const tokenizerJson = await tokenizerResp.text();
        await this._sendAndWait('loadTokenizer', { tokenizerJson });

        // Phase 1: Encode
        if (this.onProgress) this.onProgress('Fetching encoder + adapter shards...', 10);
        const [encoderBytes, adapterBytes] = await Promise.all([
            this._fetchGzipped(urls.encoder),
            this._fetchGzipped(urls.adapter),
        ]);

        if (this.onProgress) this.onProgress('Loading encoder...', 30);
        await this._sendAndWait('loadEncoder', { encoderBytes, adapterBytes }, [encoderBytes, adapterBytes]);

        if (this.onProgress) this.onProgress('Encoding audio...', 40);
        await this._sendAndWait('encodeAudio', { audio }, [audio.buffer]);

        if (this.onProgress) this.onProgress('Freeing encoder...', 45);
        await this._sendAndWait('freeEncoder', {});

        // Phase 2: Decode — check manifest for streamed format
        let manifest = null;
        if (urls.manifest) {
            manifest = await fetch(urls.manifest).then(r => r.json());
        }

        if (manifest && manifest.decoder_format === 'streamed') {
            const baseUrl = urls.baseUrl || urls.manifest.replace(/\/[^/]*$/, '');
            await this._loadDecoderStreamed(baseUrl, manifest);
        } else {
            if (this.onProgress) this.onProgress('Fetching decoder shard...', 50);
            const decoderBytes = await fetch(urls.decoder).then(r => r.arrayBuffer());

            if (this.onProgress) this.onProgress('Loading decoder...', 70);
            await this._sendAndWait('loadDecoder', { decoderBytes }, [decoderBytes]);
        }

        if (this.onProgress) this.onProgress('Transcribing...', 85);
        const text = await this._sendAndWait('transcribePhased', {});

        if (this.onProgress) this.onProgress('Cleaning up...', 95);
        await this._sendAndWait('freeDecoder', {});

        if (this.onProgress) this.onProgress('Done', 100);
        return text;
    }

    /**
     * Load decoder as streamed per-layer shards for lower peak memory.
     *
     * Fetches and loads: embeddings → 26 layers sequentially → norm.
     * Each step: fetch bytes → init small f32 scaffold → load record → drop init.
     *
     * @param {string} baseUrl - Base URL where shard files are served
     * @param {object} manifest - Parsed manifest.json with decoder_layers count
     * @private
     */
    async _loadDecoderStreamed(baseUrl, manifest) {
        const nLayers = manifest.decoder_layers;
        // Total steps: 1 (embeddings) + nLayers + 1 (norm) = nLayers + 2
        const totalSteps = nLayers + 2;

        // Embeddings
        if (this.onProgress) this.onProgress('Fetching decoder embeddings...', 50);
        const embBytes = await this._fetchGzipped(`${baseUrl}/decoder_embeddings.safetensors.gz`);
        await this._sendAndWait('loadDecoderEmbeddings', { bytes: embBytes }, [embBytes]);

        // Layers 0..nLayers-1
        for (let i = 0; i < nLayers; i++) {
            const pct = 50 + Math.round(((i + 1) / totalSteps) * 35);
            if (this.onProgress) this.onProgress(`Fetching decoder layer ${i}/${nLayers}...`, pct);

            const layerName = `decoder_layer_${String(i).padStart(2, '0')}.safetensors.gz`;
            const layerBytes = await this._fetchGzipped(`${baseUrl}/${layerName}`);
            await this._sendAndWait('loadDecoderLayer', { index: i, bytes: layerBytes }, [layerBytes]);
        }

        // Norm (also assembles the full decoder)
        if (this.onProgress) this.onProgress('Loading decoder norm...', 85);
        const normBytes = await this._fetchGzipped(`${baseUrl}/decoder_norm.safetensors.gz`);
        await this._sendAndWait('loadDecoderNorm', { bytes: normBytes }, [normBytes]);
    }

    /**
     * Transcribe an audio file using phased loading.
     * @param {File|Blob} file - Audio file
     * @param {object} urls - Shard URLs (see transcribePhased)
     * @returns {Promise<string>} Transcribed text
     */
    async transcribeFilePhased(file, urls) {
        const audio = await this._decodeAudioFile(file);
        return this.transcribePhased(audio, urls);
    }

    /**
     * Stop microphone and transcribe using phased loading.
     * @param {object} urls - Shard URLs (see transcribePhased)
     * @returns {Promise<string>} Transcribed text
     */
    async stopAndTranscribePhased(urls) {
        if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
            throw new Error('Microphone not recording.');
        }

        const audioBlob = await new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, {
                    type: this.mediaRecorder.mimeType
                });
                resolve(blob);
            };
            this.mediaRecorder.stop();
        });

        this._stopMicrophone();

        const audio = await this._decodeAudioFile(audioBlob);
        return this.transcribePhased(audio, urls);
    }

    /**
     * Helper: send a message to the worker and wait for the response.
     * @private
     */
    _sendAndWait(type, data, transfers = []) {
        return new Promise((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.worker.postMessage({ type, ...data }, transfers);
        });
    }

    _getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return ''; // Let browser choose default
    }
}
