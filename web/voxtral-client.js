/**
 * Voxtral Client - Browser API for Q4 GGUF speech transcription.
 *
 * Handles WebWorker communication and Web Audio API for microphone input.
 *
 * Usage:
 *   const client = new VoxtralClient();
 *   await client.init();
 *   await client.loadModel(ggufBytes, tokenizerJson);
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
     * Load Q4 GGUF model weights and tokenizer.
     * @param {ArrayBuffer|Uint8Array} ggufBytes - GGUF model weights (~2GB)
     * @param {string} tokenizerJson - Tokenizer JSON string
     */
    async loadModel(ggufBytes, tokenizerJson) {
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
            const bytes = ggufBytes instanceof Uint8Array
                ? ggufBytes.buffer
                : ggufBytes;

            this.worker.postMessage(
                { type: 'loadModel', ggufBytes: bytes, tokenizerJson },
                [bytes]
            );
        });
    }

    /**
     * Load model shards from the server (no local files needed).
     */
    async loadFromServer() {
        if (!this.ready) {
            throw new Error('Client not initialized. Call init() first.');
        }

        return new Promise((resolve, reject) => {
            this.pendingResolve = () => {
                this.modelLoaded = true;
                resolve();
            };
            this.pendingReject = reject;

            this.worker.postMessage({ type: 'loadFromServer' });
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
