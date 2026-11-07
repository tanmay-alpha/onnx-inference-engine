#!/usr/bin/env bash
# Build the Crucible WASM runtime (pure-Rust interpreter in wasm/, NOT the
# native C++ engine) with wasm-pack for the web target. Output goes to
# web/public/wasm (served by the web app) and is mirrored to wasm/pkg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/web/public/wasm"
PKG="$ROOT/wasm/pkg"

echo "Building Crucible WASM module..."
wasm-pack build "$ROOT/wasm" --release --target web --out-dir "$OUT"

mkdir -p "$PKG"
for f in crucible_wasm.js crucible_wasm.d.ts crucible_wasm_bg.wasm crucible_wasm_bg.wasm.d.ts package.json; do
    cp "$OUT/$f" "$PKG/$f"
done

echo "WASM build complete: $(wc -c < "$OUT/crucible_wasm_bg.wasm") bytes ($OUT/crucible_wasm_bg.wasm)"
