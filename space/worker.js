/**
 * Voxtral WebWorker — Q4 GGUF inference off the main thread.
 *
 * Messages:
 *   { type: 'init' }
 *   { type: 'loadFromServer' }
 *   { type: 'transcribe', audio }
 *   { type: 'clearCache' }
 *   { type: 'checkCache' }
 */

import init, { VoxtralQ4, initWgpuDevice } from './pkg/voxtral_mini_realtime.js';

const HF_BASE = "https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf/resolve/main";
const SHARD_NAMES = ["shard-aa", "shard-ab", "shard-ac", "shard-ad", "shard-ae"];
const CACHE_NAME = "voxtral-weights-v1";

let voxtral = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;
            case 'loadFromServer':
                await handleLoadFromServer();
                break;
            case 'transcribe':
                await handleTranscribe(data.audio);
                break;
            case 'clearCache':
                await handleClearCache();
                break;
            case 'checkCache':
                await handleCheckCache();
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

async function cachedFetch(cache, url) {
    const cached = await cache.match(url);
    if (cached) return { response: cached, fromCache: true };
    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`Failed to download ${url}: ${resp.status} ${resp.statusText}`);
    }
    await cache.put(url, resp.clone());
    return { response: resp, fromCache: false };
}

async function handleLoadFromServer() {
    if (!voxtral) throw new Error('Worker not initialized.');

    const cache = await caches.open(CACHE_NAME);

    for (let i = 0; i < SHARD_NAMES.length; i++) {
        const name = SHARD_NAMES[i];
        const url = `${HF_BASE}/${name}`;

        self.postMessage({
            type: 'progress',
            stage: `Loading ${name} (${i + 1}/${SHARD_NAMES.length})...`,
            percent: Math.round((i / SHARD_NAMES.length) * 60),
        });

        const { response, fromCache } = await cachedFetch(cache, url);

        if (fromCache) {
            self.postMessage({
                type: 'progress',
                stage: `Loaded ${name} from cache (${i + 1}/${SHARD_NAMES.length})`,
                percent: Math.round(((i + 1) / SHARD_NAMES.length) * 60),
            });
        }

        const buf = await response.arrayBuffer();
        voxtral.appendModelShard(new Uint8Array(buf));
    }

    // Tokenizer
    self.postMessage({ type: 'progress', stage: 'Loading tokenizer...', percent: 65 });
    const tokUrl = `${HF_BASE}/tekken.json`;
    const { response: tokResp } = await cachedFetch(cache, tokUrl);
    const tokenizerJson = await tokResp.text();

    // Finalize
    self.postMessage({ type: 'progress', stage: 'Loading into WebGPU...', percent: 70 });
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

async function handleClearCache() {
    const deleted = await caches.delete(CACHE_NAME);
    self.postMessage({ type: 'cacheCleared', deleted });
}

async function handleCheckCache() {
    try {
        const cache = await caches.open(CACHE_NAME);
        const keys = await cache.keys();
        const shardsCached = SHARD_NAMES.filter(name =>
            keys.some(k => k.url.endsWith(name))
        );
        self.postMessage({
            type: 'cacheStatus',
            cached: shardsCached.length === SHARD_NAMES.length,
            shardsCached: shardsCached.length,
            shardsTotal: SHARD_NAMES.length,
        });
    } catch {
        self.postMessage({ type: 'cacheStatus', cached: false, shardsCached: 0, shardsTotal: SHARD_NAMES.length });
    }
}
