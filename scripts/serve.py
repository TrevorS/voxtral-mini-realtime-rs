#!/usr/bin/env -S uv run --script --python 3.11
# /// script
# requires-python = ">=3.11"
# ///
"""Dev server with COOP/COEP headers for WASM+WebGPU."""

import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"  {args[0]}")


with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as server:
    print(f"Serving on http://0.0.0.0:{PORT}")
    print(f"Open http://localhost:{PORT}/web/index.html")
    server.serve_forever()
