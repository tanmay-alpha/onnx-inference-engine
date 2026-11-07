/* tslint:disable */
/* eslint-disable */

/**
 * Model metadata as a JSON string (parse with `JSON.parse`).
 */
export function modelInfo(model_bytes: Uint8Array): string;

/**
 * Runs the model like `runInference` and returns only the output shape
 * (row-major dims of the first graph output).
 */
export function outputShape(model_bytes: Uint8Array, input_data: Float32Array, input_shape: Int32Array): Int32Array;

/**
 * Fraud probability for one raw transaction: z-score normalises the seven
 * features with the training constants, then runs the model on `[1, 7]`.
 */
export function runFraudModel(model_bytes: Uint8Array, amount: number, old_balance_orig: number, new_balance_orig: number, old_balance_dest: number, new_balance_dest: number, is_cash_out: number, is_transfer: number): number;

/**
 * Runs the model on one float32 input and returns the first output's data
 * (row-major). Throws a string on unsupported ops or bad shapes.
 */
export function runInference(model_bytes: Uint8Array, input_data: Float32Array, input_shape: Int32Array): Float32Array;

/**
 * Operators this WASM runtime implements.
 */
export function supportedOps(): string[];

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly modelInfo: (a: number, b: number) => [number, number, number, number];
    readonly outputShape: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly runFraudModel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly runInference: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly supportedOps: () => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
