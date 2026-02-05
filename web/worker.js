/**
 * Voxtral WebWorker for off-main-thread inference.
 *
 * Messages:
 *   { type: 'init' } - Initialize WASM module
 *   { type: 'loadModel', modelBytes: ArrayBuffer, tokenizerJson: string } - Load model
 *   { type: 'transcribe', audio: Float32Array } - Transcribe audio
 *
 * Responses:
 *   { type: 'ready' } - WASM initialized
 *   { type: 'modelLoaded' } - Model loaded successfully
 *   { type: 'transcription', text: string } - Transcription result
 *   { type: 'error', message: string } - Error occurred
 *   { type: 'progress', stage: string, percent?: number } - Progress update
 */

let voxtral = null;

// Import WASM module
importScripts('./pkg/voxtral_mini_realtime.js');

const { Voxtral, default: init } = wasm_bindgen;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;
            case 'loadModel':
                await handleLoadModel(data.modelBytes, data.tokenizerJson);
                break;
            case 'transcribe':
                handleTranscribe(data.audio);
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({ type: 'error', message: error.message || String(error) });
    }
};

async function handleInit() {
    self.postMessage({ type: 'progress', stage: 'Initializing WASM...' });

    // Initialize WASM module
    await init('./pkg/voxtral_mini_realtime_bg.wasm');

    // Create Voxtral instance
    voxtral = new Voxtral();

    self.postMessage({ type: 'ready' });
}

async function handleLoadModel(modelBytes, tokenizerJson) {
    if (!voxtral) {
        throw new Error('Worker not initialized. Call init first.');
    }

    self.postMessage({ type: 'progress', stage: 'Loading model weights...' });

    // Convert ArrayBuffer to Uint8Array if needed
    const bytes = modelBytes instanceof Uint8Array
        ? modelBytes
        : new Uint8Array(modelBytes);

    voxtral.loadModel(bytes, tokenizerJson);

    self.postMessage({ type: 'modelLoaded' });
}

function handleTranscribe(audio) {
    if (!voxtral) {
        throw new Error('Worker not initialized. Call init first.');
    }

    if (!voxtral.isReady()) {
        throw new Error('Model not loaded. Call loadModel first.');
    }

    self.postMessage({ type: 'progress', stage: 'Transcribing...' });

    // Ensure we have a Float32Array
    const audioData = audio instanceof Float32Array
        ? audio
        : new Float32Array(audio);

    const text = voxtral.transcribe(audioData);

    self.postMessage({ type: 'transcription', text });
}
