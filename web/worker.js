/**
 * Voxtral WebWorker for off-main-thread Q4 GGUF inference (ES Module Worker).
 *
 * Messages:
 *   { type: 'init' }
 *   { type: 'loadModel', ggufBytes, tokenizerJson }
 *   { type: 'loadFromServer' }
 *   { type: 'transcribe', audio }
 */

import init, { VoxtralQ4, initWgpuDevice } from '../pkg/voxtral_mini_realtime.js';

let voxtral = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;
            case 'loadModel':
                handleLoadModel(data.ggufBytes, data.tokenizerJson);
                break;
            case 'loadFromServer':
                await handleLoadFromServer();
                break;
            case 'transcribe':
                await handleTranscribe(data.audio);
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
    voxtral = new VoxtralQ4();
    self.postMessage({ type: 'ready' });
}

function handleLoadModel(ggufBytes, tokenizerJson) {
    if (!voxtral) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading Q4 GGUF model...' });

    const bytes = ggufBytes instanceof Uint8Array
        ? ggufBytes
        : new Uint8Array(ggufBytes);

    voxtral.loadModel(bytes, tokenizerJson);
    self.postMessage({ type: 'modelLoaded' });
}

async function handleLoadFromServer() {
    if (!voxtral) throw new Error('Worker not initialized.');

    // Discover shards from server
    self.postMessage({ type: 'progress', stage: 'Discovering model shards...' });
    const shardsResp = await fetch('/api/shards');
    const { shards } = await shardsResp.json();

    if (!shards || shards.length === 0) {
        throw new Error('No model shards found on server.');
    }

    // Download shards sequentially (each ≤512 MB)
    let totalBytes = 0;
    for (let i = 0; i < shards.length; i++) {
        const name = shards[i];
        self.postMessage({
            type: 'progress',
            stage: `Downloading shard ${i + 1}/${shards.length} (${name})...`,
            percent: Math.round((i / shards.length) * 60),
        });

        const resp = await fetch(`/models/voxtral-q4-shards/${name}`);
        const buf = await resp.arrayBuffer();
        voxtral.appendModelShard(new Uint8Array(buf));
        totalBytes += buf.byteLength;
    }

    // Fetch tokenizer
    self.postMessage({ type: 'progress', stage: 'Loading tokenizer...', percent: 65 });
    const tokResp = await fetch('/models/voxtral/tekken.json');
    const tokenizerJson = await tokResp.text();

    // Parse GGUF and load into WebGPU
    self.postMessage({ type: 'progress', stage: 'Loading model into WebGPU...', percent: 70 });
    voxtral.loadModelFromShards(tokenizerJson);

    self.postMessage({ type: 'modelLoaded' });
}

async function handleTranscribe(audio) {
    if (!voxtral || !voxtral.isReady()) {
        throw new Error('Model not loaded.');
    }
    self.postMessage({ type: 'progress', stage: 'Transcribing...' });

    const audioData = audio instanceof Float32Array
        ? audio
        : new Float32Array(audio);

    const text = await voxtral.transcribe(audioData);
    self.postMessage({ type: 'transcription', text });
}
