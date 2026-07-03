import init, { runInference as wasmRunInference } from '../../public/wasm/crucible_wasm';

let initialized = false;

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
  inputShape: number[]
): Promise<Float32Array> {
  await initWasm();
  
  // Note: wasmRunInference is mapped from Rust's run_inference(model_bytes, input_data, input_shape)
  // using wasm-bindgen. It accepts Uint8Array, Float32Array, and Int32Array (or standard number Array).
  const shapeArray = new Int32Array(inputShape);
  const output = wasmRunInference(modelBytes, inputData, shapeArray);
  return output;
}
