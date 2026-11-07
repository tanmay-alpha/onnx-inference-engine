// Crucible WASM runtime benchmark worker (Node.js).
//
// Loads the wasm-bindgen "web" target that the site ships
// (web/public/wasm/crucible_wasm.js + crucible_wasm_bg.wasm) with
// initSync() on the raw .wasm bytes, then times the public
// runInference(modelBytes, input, shape) export -- the same call the
// browser playground makes. That API takes the model bytes on every
// call, so each timed call includes parsing the ONNX protobuf.
//
// load_ms = WebAssembly compile + instantiate (initSync) time.
// Timing uses process.hrtime.bigint() around each call.
//
// Usage:
//   node bench_wasm.mjs --wasm-dir DIR --model M.onnx --input-bin x.f32
//        --shape 1,7 --out-bin y.f32 --result res.json --warmup W --iterations N
//
// Writes {"status":"ok","load_ms","timings_ms":[...],"threads":1,"note"}
// or {"status":"unsupported","note"} if runInference throws.

import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith("--") || argv[i + 1] === undefined) {
      throw new Error(`bad argument list near ${argv[i]}`);
    }
    out[argv[i].slice(2)] = argv[i + 1];
  }
  for (const k of ["wasm-dir", "model", "input-bin", "shape", "out-bin", "result",
                   "warmup", "iterations"]) {
    if (out[k] === undefined) throw new Error(`missing --${k}`);
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const wasmDir = resolve(args["wasm-dir"]);
const bindings = await import(pathToFileURL(join(wasmDir, "crucible_wasm.js")).href);

const wasmBytes = readFileSync(join(wasmDir, "crucible_wasm_bg.wasm"));
const t0 = process.hrtime.bigint();
bindings.initSync({ module: wasmBytes });
const loadMs = Number(process.hrtime.bigint() - t0) / 1e6;

const modelBytes = new Uint8Array(readFileSync(args.model));
const raw = readFileSync(args["input-bin"]);
const input = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const shape = Int32Array.from(args.shape.split(",").map(Number));
const warmup = Number(args.warmup);
const iterations = Number(args.iterations);

const call = () => bindings.runInference(modelBytes, input, shape);

let out;
try {
  out = call();
} catch (err) {
  writeFileSync(args.result, JSON.stringify({
    status: "unsupported",
    note: `runInference threw: ${err instanceof Error ? err.message : String(err)}`,
    threads: 1,
  }));
  process.exit(0);
}

for (let i = 0; i < warmup; i++) out = call();
const timings = [];
for (let i = 0; i < iterations; i++) {
  const s = process.hrtime.bigint();
  out = call();
  timings.push(Number(process.hrtime.bigint() - s) / 1e6);
}

writeFileSync(args["out-bin"], Buffer.from(out.buffer, out.byteOffset, out.byteLength));
writeFileSync(args.result, JSON.stringify({
  status: "ok",
  load_ms: loadMs,
  timings_ms: timings,
  threads: 1,
  note: `Node ${process.version}; runInference re-parses the model bytes on every call`,
}));
