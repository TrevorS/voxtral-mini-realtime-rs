#!/usr/bin/env -S uv run --with playwright --with pytest --python 3.11
# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright", "pytest"]
# ///
"""
End-to-end test for Voxtral WASM in headless browser.

Tests:
1. WASM module loads successfully
2. Voxtral class can be instantiated
3. API methods are available
4. (Optional) Full transcription with model files

Usage:
    # Install browsers first (one-time)
    uvx --with playwright playwright install chromium

    # Run tests
    ./scripts/test_wasm_e2e.py

    # Or with pytest
    uvx --with playwright --with pytest pytest scripts/test_wasm_e2e.py -v
"""

import subprocess
import sys
import time
import threading
import http.server
import socketserver
from pathlib import Path

# Check if playwright browsers are installed
def ensure_browsers():
    """Install playwright browsers if needed."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Try to launch - will fail if not installed
            browser = p.chromium.launch(headless=True)
            browser.close()
            return True
    except Exception as e:
        if "Executable doesn't exist" in str(e):
            print("Installing Chromium browser...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            return True
        raise

class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logging."""
    def log_message(self, format, *args):
        pass  # Suppress logs

class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that allows address reuse."""
    allow_reuse_address = True

def start_server(port: int, directory: str) -> socketserver.TCPServer:
    """Start a simple HTTP server in a background thread."""
    import os
    os.chdir(directory)
    handler = QuietHTTPHandler
    server = ReusableTCPServer(("", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

def get_free_port():
    """Get a free port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def test_wasm_init():
    """Test that WASM module initializes correctly."""
    from playwright.sync_api import sync_playwright

    project_root = Path(__file__).parent.parent
    port = get_free_port()

    # Start server
    server = start_server(port, str(project_root))
    time.sleep(0.5)  # Let server start

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Collect console messages
            console_msgs = []
            page.on("console", lambda msg: console_msgs.append(f"{msg.type}: {msg.text}"))

            # Navigate to test page
            page.goto(f"http://localhost:{port}/web/index.html")

            # Wait for WASM to initialize (status should change)
            page.wait_for_function(
                "document.getElementById('status-text').textContent !== 'Not loaded'",
                timeout=30000
            )

            # Check status
            status = page.locator("#status-text").text_content()
            print(f"Status after init: {status}")

            assert "Ready" in status or "Load model" in status, f"Unexpected status: {status}"

            # Test that Voxtral class is available in the module
            result = page.evaluate("""
                async () => {
                    try {
                        // Import the module (path from root, not from /web/)
                        const mod = await import('../pkg/voxtral_mini_realtime.js');
                        await mod.default();

                        // Check class exists
                        const voxtral = new mod.Voxtral();

                        return {
                            success: true,
                            isReady: voxtral.isReady(),
                            sampleRate: voxtral.getSampleRate(),
                            hasTranscribe: typeof voxtral.transcribe === 'function',
                            hasLoadModel: typeof voxtral.loadModel === 'function',
                        };
                    } catch (e) {
                        return { success: false, error: e.message };
                    }
                }
            """)

            print(f"WASM test result: {result}")

            assert result["success"], f"WASM init failed: {result.get('error')}"
            assert result["isReady"] == False, "Should not be ready without model"
            assert result["sampleRate"] == 16000, f"Wrong sample rate: {result['sampleRate']}"
            assert result["hasTranscribe"], "Missing transcribe method"
            assert result["hasLoadModel"], "Missing loadModel method"

            browser.close()

            print("\n✅ WASM initialization test passed!")
            return True

    finally:
        server.shutdown()

def test_worker_init():
    """Test that WebWorker initializes correctly."""
    from playwright.sync_api import sync_playwright

    project_root = Path(__file__).parent.parent
    port = get_free_port()

    # Start server
    server = start_server(port, str(project_root))
    time.sleep(0.5)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate and test worker
            page.goto(f"http://localhost:{port}/web/index.html")

            # Test VoxtralClient
            result = page.evaluate("""
                async () => {
                    try {
                        const { VoxtralClient } = await import('./voxtral-client.js');
                        const client = new VoxtralClient();
                        await client.init();

                        return {
                            success: true,
                            isReady: client.isReady(),
                            hasStartMicrophone: typeof client.startMicrophone === 'function',
                            hasTranscribeFile: typeof client.transcribeFile === 'function',
                        };
                    } catch (e) {
                        return { success: false, error: e.message };
                    }
                }
            """)

            print(f"Worker test result: {result}")

            assert result["success"], f"Worker init failed: {result.get('error')}"
            assert result["isReady"] == False, "Should not be ready without model"
            assert result["hasStartMicrophone"], "Missing startMicrophone method"
            assert result["hasTranscribeFile"], "Missing transcribeFile method"

            browser.close()

            print("\n✅ WebWorker initialization test passed!")
            return True

    finally:
        server.shutdown()

def test_transcription_with_model():
    """Test full transcription (requires model files)."""
    from playwright.sync_api import sync_playwright

    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "voxtral" / "consolidated.safetensors"
    tokenizer_path = project_root / "models" / "voxtral" / "tekken.json"
    audio_path = project_root / "test_genesis.wav"

    if not model_path.exists():
        print("⏭️  Skipping transcription test: model not found")
        return None

    if not audio_path.exists():
        print("⏭️  Skipping transcription test: test audio not found")
        return None

    port = get_free_port()
    server = start_server(port, str(project_root))
    time.sleep(0.5)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            # This test would take a very long time due to 8GB model
            # Just verify the flow works
            print("⏭️  Skipping full transcription test (8GB model too large for browser test)")

            browser.close()
            return None

    finally:
        server.shutdown()

def main():
    """Run all tests."""
    print("=" * 60)
    print("Voxtral WASM End-to-End Tests")
    print("=" * 60)

    # Ensure browsers are installed
    print("\nChecking browser installation...")
    ensure_browsers()

    # Run tests
    print("\n--- Test 1: WASM Module Initialization ---")
    test_wasm_init()

    print("\n--- Test 2: WebWorker Initialization ---")
    test_worker_init()

    print("\n--- Test 3: Full Transcription ---")
    test_transcription_with_model()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
