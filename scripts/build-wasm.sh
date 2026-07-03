#!/bin/bash
# wasm-pack build script for Crucible WASM module.
# Compiles the Rust WASM package for the web target and saves it in the web assets path.

set -e

echo "Building Crucible WASM module..."
wasm-pack build wasm --target web --out-dir ../web/public/wasm
echo "WASM build completed successfully."
