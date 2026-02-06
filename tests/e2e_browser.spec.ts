/**
 * E2E headless browser test: WASM + WebGPU Q4 GGUF inference.
 *
 * Serves files, loads sharded Q4 model (≤512 MB each to stay under
 * Chrome's 2 GB ArrayBuffer limit), transcribes mary_had_lamb.wav,
 * and checks the output.
 *
 * Run: bunx playwright test tests/e2e_browser.spec.ts
 */

import { test, expect } from "@playwright/test";
import { createServer, type Server } from "node:http";
import { createReadStream, existsSync, statSync, readdirSync } from "node:fs";
import { join, extname } from "node:path";

const ROOT = join(import.meta.dirname, "..");
const SHARD_DIR = join(ROOT, "models/voxtral-q4-shards");
const TOKENIZER_PATH = join(ROOT, "models/voxtral/tekken.json");
const AUDIO_PATH = join(ROOT, "test_data/mary_had_lamb.wav");

const MIME: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".wav": "audio/wav",
};

// Discover shard filenames (sorted alphabetically so order is correct)
const SHARD_NAMES = existsSync(SHARD_DIR)
  ? readdirSync(SHARD_DIR)
      .filter((f) => f.startsWith("shard-"))
      .sort()
  : [];

// The test page fetches shards one-by-one, passes each to WASM's
// appendModelShard(), then calls loadModelFromShards() to assemble.
const TEST_HTML = `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body>
<pre id="log"></pre>
<script type="module">
const log = (msg) => {
  console.log(msg);
  document.getElementById('log').textContent += msg + '\\n';
};

const SHARDS = ${JSON.stringify(SHARD_NAMES)};

try {
  log('[e2e] Importing WASM module...');
  const mod = await import('/pkg/voxtral_mini_realtime.js');
  log('[e2e] WASM module loaded');

  await mod.default();
  log('[e2e] WASM initialized');

  await mod.initWgpuDevice();
  log('[e2e] WebGPU device initialized');

  const voxtral = new mod.VoxtralQ4();

  // Load shards sequentially — each ≤512 MB fits in an ArrayBuffer
  log('[e2e] Loading ' + SHARDS.length + ' GGUF shards...');
  const t0 = performance.now();
  let totalBytes = 0;
  for (const name of SHARDS) {
    const resp = await fetch('/models/voxtral-q4-shards/' + name);
    const buf = await resp.arrayBuffer();
    voxtral.appendModelShard(new Uint8Array(buf));
    totalBytes += buf.byteLength;
    log('[e2e] Shard ' + name + ': ' + (buf.byteLength / 1e6).toFixed(0) + ' MB (total: ' + (totalBytes / 1e9).toFixed(2) + ' GB)');
  }
  log('[e2e] All shards downloaded in ' + Math.round(performance.now() - t0) + ' ms');

  // Fetch tokenizer
  log('[e2e] Fetching tokenizer...');
  const tokResp = await fetch('/models/voxtral/tekken.json');
  const tokJson = await tokResp.text();
  log('[e2e] Tokenizer loaded');

  // Parse GGUF and load into WebGPU
  log('[e2e] Loading Q4 model into WebGPU...');
  const t1 = performance.now();
  voxtral.loadModelFromShards(tokJson);
  const loadMs = Math.round(performance.now() - t1);
  log('[e2e] Model loaded in ' + loadMs + ' ms');

  // Decode audio
  log('[e2e] Fetching audio...');
  const audioResp = await fetch('/test_data/mary_had_lamb.wav');
  const audioArrayBuf = await audioResp.arrayBuffer();

  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const audioBuf = await audioCtx.decodeAudioData(audioArrayBuf);
  const samples = audioBuf.getChannelData(0);
  log('[e2e] Audio: ' + samples.length + ' samples, ' + (samples.length / 16000).toFixed(2) + 's');
  await audioCtx.close();

  // Transcribe
  log('[e2e] Transcribing...');
  const t2 = performance.now();
  const text = await voxtral.transcribe(samples);
  const transcribeMs = Math.round(performance.now() - t2);
  log('[e2e] Transcription complete in ' + transcribeMs + ' ms');
  log('[e2e] Text: ' + text);

  document.body.setAttribute('data-result', JSON.stringify({ text, loadMs, transcribeMs }));
  document.body.setAttribute('data-done', 'true');
} catch (e) {
  console.error('[e2e] Error:', e);
  document.body.setAttribute('data-error', e.message || String(e));
  document.body.setAttribute('data-done', 'true');
}
</script>
</body></html>`;

const canRun =
  SHARD_NAMES.length > 0 &&
  existsSync(TOKENIZER_PATH) &&
  existsSync(AUDIO_PATH);

test.describe("WASM Q4 GGUF E2E", () => {
  test.skip(!canRun, "Model shards or test files not found");

  let server: Server;
  let port: number;

  test.beforeAll(async () => {
    server = createServer((req, res) => {
      const url = req.url ?? "/";

      if (url === "/" || url === "/test.html") {
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(TEST_HTML);
        return;
      }

      let filePath: string;
      if (url.startsWith("/pkg/")) {
        filePath = join(ROOT, url);
      } else if (url.startsWith("/models/") || url.startsWith("/test_data/")) {
        filePath = join(ROOT, url);
      } else {
        filePath = join(ROOT, "web", url);
      }

      if (!existsSync(filePath)) {
        res.writeHead(404);
        res.end("Not found: " + url);
        return;
      }

      const ext = extname(filePath);
      const mime = MIME[ext] ?? "application/octet-stream";
      const stat = statSync(filePath);

      res.writeHead(200, {
        "Content-Type": mime,
        "Content-Length": stat.size,
      });

      createReadStream(filePath).pipe(res);
    });

    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => {
        const addr = server.address();
        port = typeof addr === "object" ? addr!.port : 0;
        console.log(`Test server listening on port ${port}`);
        resolve();
      });
    });
  });

  test.afterAll(async () => {
    server?.close();
  });

  test(
    "transcribe mary_had_lamb.wav via WebGPU",
    { timeout: 600_000 },
    async ({ page }) => {
      const logs: string[] = [];
      page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));
      page.on("pageerror", (err) => logs.push(`[pageerror] ${err.message}`));

      await page.goto(`http://127.0.0.1:${port}/`);

      // Wait for the script to signal completion
      try {
        await page.waitForSelector('body[data-done="true"]', {
          timeout: 590_000,
        });
      } catch (e) {
        // Print logs even on timeout so we can see progress
        console.log("\n--- Browser console logs (on timeout) ---");
        for (const line of logs) console.log(line);
        console.log("-----------------------------------------\n");
        throw e;
      }

      console.log("\n--- Browser console logs ---");
      for (const line of logs) console.log(line);
      console.log("----------------------------\n");

      const error = await page.getAttribute("body", "data-error");
      if (error) {
        throw new Error(`Browser error: ${error}`);
      }

      const resultJson = await page.getAttribute("body", "data-result");
      expect(resultJson).toBeTruthy();
      const result = JSON.parse(resultJson!);

      console.log(`Transcription: "${result.text}"`);
      console.log(`Model load: ${result.loadMs} ms`);
      console.log(`Transcribe: ${result.transcribeMs} ms`);

      const lower = result.text.toLowerCase();
      expect(lower).toContain("mary");
      expect(lower).toContain("lamb");
    }
  );
});
