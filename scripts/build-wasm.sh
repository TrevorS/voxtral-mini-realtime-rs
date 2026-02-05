#!/usr/bin/env bash
set -euo pipefail

# Build Voxtral for WASM/browser
#
# Usage:
#   ./scripts/build-wasm.sh [ndarray|wgpu]
#
# Defaults to ndarray (CPU) backend which works in all browsers.
# Use wgpu for WebGPU acceleration (requires browser support).

BACKEND="${1:-ndarray}"

echo "Building Voxtral for WASM with $BACKEND backend..."

# Add wasm32 target
rustup target add wasm32-unknown-unknown

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Set feature based on backend
if [ "$BACKEND" = "wgpu" ]; then
    FEATURE="wasm-wgpu"
else
    FEATURE="wasm"
fi

# Optimization flags for WASM
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=s --cfg web_sys_unstable_apis --cfg getrandom_backend=\"wasm_js\""

# Build with wasm-pack
mkdir -p pkg
wasm-pack build \
    --out-dir pkg \
    --release \
    --target web \
    --no-default-features \
    --features "$FEATURE"

echo ""
echo "Build complete! Output in pkg/"
echo ""
echo "To test locally:"
echo "  python3 -m http.server 8080"
echo "  # Open http://localhost:8080/web/index.html"
