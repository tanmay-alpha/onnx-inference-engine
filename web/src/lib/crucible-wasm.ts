/**
 * Thin wrapper around the pure-Rust WASM runtime (wasm/src/lib.rs, built
 * with wasm-pack into public/wasm). This is a separate, small interpreter —
 * not the native C++ engine — supporting MatMul, Add, Relu, Sigmoid and
 * Softmax with a single float32 graph input. Every function here either
 * returns a real result from the runtime or throws its error message.
 */
import init, {
  modelInfo as wasmModelInfo,
  outputShape as wasmOutputShape,
  runFraudModel as wasmRunFraudModel,
  runInference as wasmRunInference,
  supportedOps as wasmSupportedOps,
} from "../../public/wasm/crucible_wasm";

let initPromise: Promise<void> | null = null;
let fraudModelBytes: Uint8Array | null = null;

export const FRAUD_MODEL_URL = "/models/fraud_detector.onnx";

/** wasm-bindgen throws plain strings; normalise to Error. */
function asError(e: unknown): Error {
  return e instanceof Error ? e : new Error(String(e));
}

/** Loads and instantiates the WASM module once. */
export function initWasm(): Promise<void> {
  if (!initPromise) {
    initPromise = init().then(
      () => undefined,
      (e: unknown) => {
        initPromise = null;
        throw asError(e);
      },
    );
  }
  return initPromise;
}

export interface WasmTensorInfo {
  name: string;
  elem_type: number;
  /** -1 marks a symbolic dimension; null when the model declares no shape. */
  shape: number[] | null;
}

/** Output of the runtime's `modelInfo` export. */
export interface WasmModelInfo {
  opset: number | null;
  num_nodes: number;
  inputs: WasmTensorInfo[];
  outputs: WasmTensorInfo[];
  ops_used: string[];
  /** Whether this WASM runtime can execute the model. */
  supported: boolean;
  unsupported_reasons: string[];
}

/** Parses a model with the WASM runtime and reports whether it can run it. */
export async function getWasmModelInfo(modelBytes: Uint8Array): Promise<WasmModelInfo> {
  await initWasm();
  try {
    return JSON.parse(wasmModelInfo(modelBytes)) as WasmModelInfo;
  } catch (e) {
    throw asError(e);
  }
}

/** Operators the WASM runtime implements (reported by the runtime itself). */
export async function getWasmSupportedOps(): Promise<string[]> {
  await initWasm();
  return wasmSupportedOps();
}

export interface WasmInferenceResult {
  output: number[];
  outputShape: number[];
  /** Wall time of the runInference call: model parse + graph execution in WASM. */
  latencyMs: number;
}

/**
 * Runs the model's first output on one float32 input in the browser.
 * Throws the runtime's error (unsupported op, shape mismatch, ...) as-is.
 */
export async function runWasmInference(
  modelBytes: Uint8Array,
  inputData: Float32Array,
  inputShape: number[],
): Promise<WasmInferenceResult> {
  await initWasm();
  const shape = new Int32Array(inputShape);
  try {
    const t0 = performance.now();
    const output = wasmRunInference(modelBytes, inputData, shape);
    const latencyMs = performance.now() - t0;
    // Separate (untimed) call: the runtime reports the output dims.
    const outputShape = Array.from(wasmOutputShape(modelBytes, inputData, shape));
    return { output: Array.from(output), outputShape, latencyMs };
  } catch (e) {
    throw asError(e);
  }
}

/** Fetches the bundled fraud model once and caches its bytes. */
export async function getFraudModelBytes(): Promise<Uint8Array> {
  if (!fraudModelBytes) {
    const response = await fetch(FRAUD_MODEL_URL);
    if (!response.ok) throw new Error(`Failed to fetch fraud model: HTTP ${response.status}`);
    fraudModelBytes = new Uint8Array(await response.arrayBuffer());
  }
  return fraudModelBytes;
}

export interface FraudDetectionParams {
  amount: number;
  oldBalanceOrig: number;
  newBalanceOrig: number;
  oldBalanceDest: number;
  newBalanceDest: number;
  type: "CASH_OUT" | "TRANSFER" | "OTHER";
}

export interface FraudDetectionResult {
  probability: number;
  label: "FRAUD" | "LEGITIMATE";
  /** Wall time of the runFraudModel call (parse + execute) in this tab. */
  latencyMs: number;
  modelBytes: number;
}

/**
 * Scores one transaction with the bundled fraud model, entirely in the
 * browser. The model file is fetched once; the transaction is never sent.
 */
export async function runFraudDetection(
  params: FraudDetectionParams,
): Promise<FraudDetectionResult> {
  await initWasm();
  const bytes = await getFraudModelBytes();

  const isCashOut = params.type === "CASH_OUT" ? 1.0 : 0.0;
  const isTransfer = params.type === "TRANSFER" ? 1.0 : 0.0;

  let probability: number;
  const t0 = performance.now();
  try {
    probability = wasmRunFraudModel(
      bytes,
      params.amount,
      params.oldBalanceOrig,
      params.newBalanceOrig,
      params.oldBalanceDest,
      params.newBalanceDest,
      isCashOut,
      isTransfer,
    );
  } catch (e) {
    throw asError(e);
  }
  const latencyMs = performance.now() - t0;

  return {
    probability,
    label: probability >= 0.5 ? "FRAUD" : "LEGITIMATE",
    latencyMs,
    modelBytes: bytes.length,
  };
}
