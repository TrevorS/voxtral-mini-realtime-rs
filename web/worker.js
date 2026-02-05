/**
 * Voxtral WebWorker for off-main-thread inference (ES Module Worker).
 *
 * Supports two modes:
 *
 * 1. Full model (requires >4GB memory, non-wasm32):
 *    init → loadModel → transcribe
 *
 * 2. Phased loading (fits wasm32, loads encoder/decoder sequentially):
 *    init → loadTokenizer → loadEncoder → encodeAudio → freeEncoder
 *         → loadDecoder → transcribePhased → freeDecoder
 *
 * 3. Streamed phased loading (per-layer decoder, lower peak memory):
 *    init → loadTokenizer → loadEncoder → encodeAudio → freeEncoder
 *         → loadDecoderEmbeddings → loadDecoderLayer(0..25)
 *         → loadDecoderNorm → transcribePhased → freeDecoder
 *
 * Messages:
 *   { type: 'init' }
 *   { type: 'loadModel', modelBytes, tokenizerJson }
 *   { type: 'transcribe', audio }
 *
 *   // Phased loading (monolithic decoder)
 *   { type: 'loadTokenizer', tokenizerJson }
 *   { type: 'loadEncoder', encoderBytes, adapterBytes }
 *   { type: 'encodeAudio', audio }
 *   { type: 'freeEncoder' }
 *   { type: 'loadDecoder', decoderBytes }
 *   { type: 'transcribePhased' }
 *   { type: 'freeDecoder' }
 *
 *   // Streamed decoder loading (per-layer)
 *   { type: 'loadDecoderEmbeddings', bytes }
 *   { type: 'loadDecoderLayer', index, bytes }
 *   { type: 'loadDecoderNorm', bytes }
 */

import init, { Voxtral, VoxtralPhased, initWgpuDevice } from '../pkg/voxtral_mini_realtime.js';

let voxtral = null;
let voxtralPhased = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;

            // Full model mode
            case 'loadModel':
                await handleLoadModel(data.modelBytes, data.tokenizerJson);
                break;
            case 'transcribe':
                handleTranscribe(data.audio);
                break;

            // Phased loading mode
            case 'loadTokenizer':
                handleLoadTokenizer(data.tokenizerJson);
                break;
            case 'loadEncoder':
                handleLoadEncoder(data.encoderBytes, data.adapterBytes);
                break;
            case 'encodeAudio':
                handleEncodeAudio(data.audio);
                break;
            case 'freeEncoder':
                handleFreeEncoder();
                break;
            case 'loadDecoder':
                handleLoadDecoder(data.decoderBytes);
                break;
            case 'transcribePhased':
                handleTranscribePhased();
                break;
            case 'freeDecoder':
                handleFreeDecoder();
                break;

            // Streamed decoder loading (per-layer)
            case 'loadDecoderEmbeddings':
                handleLoadDecoderEmbeddings(data.bytes);
                break;
            case 'loadDecoderLayer':
                handleLoadDecoderLayer(data.index, data.bytes);
                break;
            case 'loadDecoderNorm':
                handleLoadDecoderNorm(data.bytes);
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
    await init();
    self.postMessage({ type: 'progress', stage: 'Initializing WebGPU device...' });
    await initWgpuDevice();
    voxtral = new Voxtral();
    voxtralPhased = new VoxtralPhased();
    self.postMessage({ type: 'ready' });
}

// --- Full model mode ---

async function handleLoadModel(modelBytes, tokenizerJson) {
    if (!voxtral) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading model weights...' });

    const bytes = modelBytes instanceof Uint8Array
        ? modelBytes
        : new Uint8Array(modelBytes);

    voxtral.loadModel(bytes, tokenizerJson);
    self.postMessage({ type: 'modelLoaded' });
}

function handleTranscribe(audio) {
    if (!voxtral || !voxtral.isReady()) {
        throw new Error('Model not loaded.');
    }
    self.postMessage({ type: 'progress', stage: 'Transcribing...' });

    const audioData = audio instanceof Float32Array
        ? audio
        : new Float32Array(audio);

    const text = voxtral.transcribe(audioData);
    self.postMessage({ type: 'transcription', text });
}

// --- Phased loading mode ---

function handleLoadTokenizer(tokenizerJson) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading tokenizer...' });
    voxtralPhased.loadTokenizer(tokenizerJson);
    self.postMessage({ type: 'tokenizerLoaded' });
}

function handleLoadEncoder(encoderBytes, adapterBytes) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');

    self.postMessage({ type: 'progress', stage: 'Loading encoder shard...' });
    const encBytes = encoderBytes instanceof Uint8Array
        ? encoderBytes
        : new Uint8Array(encoderBytes);
    voxtralPhased.loadEncoderShard(encBytes);

    self.postMessage({ type: 'progress', stage: 'Loading adapter shard...' });
    const adpBytes = adapterBytes instanceof Uint8Array
        ? adapterBytes
        : new Uint8Array(adapterBytes);
    voxtralPhased.loadAdapterShard(adpBytes);

    self.postMessage({ type: 'encoderLoaded' });
}

async function handleEncodeAudio(audio) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Encoding audio...' });

    const audioData = audio instanceof Float32Array
        ? audio
        : new Float32Array(audio);

    await voxtralPhased.encodeAudio(audioData);
    self.postMessage({ type: 'audioEncoded' });
}

function handleFreeEncoder() {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    voxtralPhased.freeEncoder();
    self.postMessage({ type: 'encoderFreed' });
}

function handleLoadDecoder(decoderBytes) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading decoder shard...' });

    const bytes = decoderBytes instanceof Uint8Array
        ? decoderBytes
        : new Uint8Array(decoderBytes);

    voxtralPhased.loadDecoderShard(bytes);
    self.postMessage({ type: 'decoderLoaded' });
}

async function handleTranscribePhased() {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Transcribing...' });

    const text = await voxtralPhased.transcribe();
    self.postMessage({ type: 'transcription', text });
}

function handleFreeDecoder() {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    voxtralPhased.freeDecoder();
    self.postMessage({ type: 'decoderFreed' });
}

// --- Streamed decoder loading (per-layer) ---

function handleLoadDecoderEmbeddings(bytes) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading decoder embeddings...' });

    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    voxtralPhased.loadDecoderEmbeddings(data);
    self.postMessage({ type: 'decoderEmbeddingsLoaded' });
}

function handleLoadDecoderLayer(index, bytes) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: `Loading decoder layer ${index}...` });

    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    voxtralPhased.loadDecoderLayer(index, data);
    self.postMessage({ type: 'decoderLayerLoaded', index });
}

function handleLoadDecoderNorm(bytes) {
    if (!voxtralPhased) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading decoder norm (assembling decoder)...' });

    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    voxtralPhased.loadDecoderNorm(data);
    self.postMessage({ type: 'decoderLoaded' });
}
