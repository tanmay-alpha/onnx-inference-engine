import init, {
  runInference as wasmRunInference,
  runFraudModel as wasmRunFraudModel,
} from "../../public/wasm/crucible_wasm";

let initialized = false;
let fraudModelBytes: Uint8Array | null = null;

/**
 * Ensures the Crucible WASM module is loaded and initialized.
 */
export async function initWasm(): Promise<void> {
  if (!initialized) {
    await init();
    initialized = true;
  }
}

/**
 * Runs client-side WebAssembly inference on the given model bytes and input data.
 *
 * @param modelBytes The raw binary data of the ONNX model.
 * @param inputData Flat Float32Array containing input tensor values.
 * @param inputShape Dimension shape of the input tensor (e.g. [1, 3, 224, 224]).
 * @returns Float32Array containing the output prediction values.
 */
export async function runWasmInference(
  modelBytes: Uint8Array,
  inputData: Float32Array,
  inputShape: number[],
): Promise<Float32Array> {
  await initWasm();

  // Note: wasmRunInference is mapped from Rust's run_inference(model_bytes, input_data, input_shape)
  // using wasm-bindgen. It accepts Uint8Array, Float32Array, and Int32Array (or standard number Array).
  const shapeArray = new Int32Array(inputShape);
  const output = wasmRunInference(modelBytes, inputData, shapeArray);
  return output;
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
  latencyMs: number;
}

/**
 * Runs privacy-preserving fraud detection entirely in the browser via WASM.
 * Fetches and caches the ONNX model on first call.
 * No transaction data leaves the device.
 */
export async function runFraudDetection(
  params: FraudDetectionParams,
): Promise<FraudDetectionResult> {
  await initWasm();

  // Lazy-load and cache the fraud model bytes
  if (!fraudModelBytes) {
    const response = await fetch("/models/fraud_detector.onnx");
    if (!response.ok) throw new Error(`Failed to fetch fraud model: ${response.status}`);
    const buffer = await response.arrayBuffer();
    fraudModelBytes = new Uint8Array(buffer);
  }

  const isCashOut = params.type === "CASH_OUT" ? 1.0 : 0.0;
  const isTransfer = params.type === "TRANSFER" ? 1.0 : 0.0;

  const t0 = performance.now();
  const probability = wasmRunFraudModel(
    fraudModelBytes,
    params.amount,
    params.oldBalanceOrig,
    params.newBalanceOrig,
    params.oldBalanceDest,
    params.newBalanceDest,
    isCashOut,
    isTransfer,
  );
  const latencyMs = performance.now() - t0;

  return {
    probability,
    label: probability >= 0.5 ? "FRAUD" : "LEGITIMATE",
    latencyMs,
  };
}
