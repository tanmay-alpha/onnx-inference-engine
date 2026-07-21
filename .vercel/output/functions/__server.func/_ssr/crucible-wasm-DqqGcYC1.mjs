//#region node_modules/.nitro/vite/services/ssr/assets/crucible-wasm-DqqGcYC1.js
/**
* @param {Uint8Array} model_bytes
* @param {number} amount
* @param {number} old_balance_orig
* @param {number} new_balance_orig
* @param {number} old_balance_dest
* @param {number} new_balance_dest
* @param {number} is_cash_out
* @param {number} is_transfer
* @returns {number}
*/
function runFraudModel(model_bytes, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, is_cash_out, is_transfer) {
	const ptr0 = passArray8ToWasm0(model_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ret = wasm.runFraudModel(ptr0, len0, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest, is_cash_out, is_transfer);
	if (ret[2]) throw takeFromExternrefTable0(ret[1]);
	return ret[0];
}
/**
* @param {Uint8Array} model_bytes
* @param {Float32Array} input_data
* @param {Int32Array} input_shape
* @returns {Float32Array}
*/
function runInference(model_bytes, input_data, input_shape) {
	const ptr0 = passArray8ToWasm0(model_bytes, wasm.__wbindgen_malloc);
	const len0 = WASM_VECTOR_LEN;
	const ptr1 = passArrayF32ToWasm0(input_data, wasm.__wbindgen_malloc);
	const len1 = WASM_VECTOR_LEN;
	const ptr2 = passArray32ToWasm0(input_shape, wasm.__wbindgen_malloc);
	const len2 = WASM_VECTOR_LEN;
	const ret = wasm.runInference(ptr0, len0, ptr1, len1, ptr2, len2);
	if (ret[3]) throw takeFromExternrefTable0(ret[2]);
	var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
	wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
	return v4;
}
function __wbg_get_imports() {
	return {
		__proto__: null,
		"./crucible_wasm_bg.js": {
			__proto__: null,
			__wbindgen_cast_0000000000000001: function(arg0, arg1) {
				return getStringFromWasm0(arg0, arg1);
			},
			__wbindgen_init_externref_table: function() {
				const table = wasm.__wbindgen_externrefs;
				const offset = table.grow(4);
				table.set(0, void 0);
				table.set(offset + 0, void 0);
				table.set(offset + 1, null);
				table.set(offset + 2, true);
				table.set(offset + 3, false);
			}
		}
	};
}
function getArrayF32FromWasm0(ptr, len) {
	ptr = ptr >>> 0;
	return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
var cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
	if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
	return cachedFloat32ArrayMemory0;
}
function getStringFromWasm0(ptr, len) {
	return decodeText(ptr >>> 0, len);
}
var cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
	if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
	return cachedUint32ArrayMemory0;
}
var cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
	if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
	return cachedUint8ArrayMemory0;
}
function passArray32ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 4, 4) >>> 0;
	getUint32ArrayMemory0().set(arg, ptr / 4);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
function passArray8ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 1, 1) >>> 0;
	getUint8ArrayMemory0().set(arg, ptr / 1);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
function passArrayF32ToWasm0(arg, malloc) {
	const ptr = malloc(arg.length * 4, 4) >>> 0;
	getFloat32ArrayMemory0().set(arg, ptr / 4);
	WASM_VECTOR_LEN = arg.length;
	return ptr;
}
function takeFromExternrefTable0(idx) {
	const value = wasm.__wbindgen_externrefs.get(idx);
	wasm.__externref_table_dealloc(idx);
	return value;
}
var cachedTextDecoder = new TextDecoder("utf-8", {
	ignoreBOM: true,
	fatal: true
});
cachedTextDecoder.decode();
var MAX_SAFARI_DECODE_BYTES = 2146435072;
var numBytesDecoded = 0;
function decodeText(ptr, len) {
	numBytesDecoded += len;
	if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
		cachedTextDecoder = new TextDecoder("utf-8", {
			ignoreBOM: true,
			fatal: true
		});
		cachedTextDecoder.decode();
		numBytesDecoded = len;
	}
	return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}
var WASM_VECTOR_LEN = 0;
var wasm;
function __wbg_finalize_init(instance, module) {
	wasm = instance.exports;
	cachedFloat32ArrayMemory0 = null;
	cachedUint32ArrayMemory0 = null;
	cachedUint8ArrayMemory0 = null;
	wasm.__wbindgen_start();
	return wasm;
}
async function __wbg_load(module, imports) {
	if (typeof Response === "function" && module instanceof Response) {
		if (typeof WebAssembly.instantiateStreaming === "function") try {
			return await WebAssembly.instantiateStreaming(module, imports);
		} catch (e) {
			if (module.ok && expectedResponseType(module.type) && module.headers.get("Content-Type") !== "application/wasm") console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);
			else throw e;
		}
		const bytes = await module.arrayBuffer();
		return await WebAssembly.instantiate(bytes, imports);
	} else {
		const instance = await WebAssembly.instantiate(module, imports);
		if (instance instanceof WebAssembly.Instance) return {
			instance,
			module
		};
		else return instance;
	}
	function expectedResponseType(type) {
		switch (type) {
			case "basic":
			case "cors":
			case "default": return true;
		}
		return false;
	}
}
async function __wbg_init(module_or_path) {
	if (wasm !== void 0) return wasm;
	if (module_or_path !== void 0) if (Object.getPrototypeOf(module_or_path) === Object.prototype) ({module_or_path} = module_or_path);
	else console.warn("using deprecated parameters for the initialization function; pass a single object instead");
	if (module_or_path === void 0) module_or_path = new URL("crucible_wasm_bg.wasm", import.meta.url);
	const imports = __wbg_get_imports();
	if (typeof module_or_path === "string" || typeof Request === "function" && module_or_path instanceof Request || typeof URL === "function" && module_or_path instanceof URL) module_or_path = fetch(module_or_path);
	const { instance, module } = await __wbg_load(await module_or_path, imports);
	return __wbg_finalize_init(instance, module);
}
var initialized = false;
var fraudModelBytes = null;
/**
* Ensures the Crucible WASM module is loaded and initialized.
*/
async function initWasm() {
	if (!initialized) {
		await __wbg_init();
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
async function runWasmInference(modelBytes, inputData, inputShape) {
	await initWasm();
	return runInference(modelBytes, inputData, new Int32Array(inputShape));
}
/**
* Runs privacy-preserving fraud detection entirely in the browser via WASM.
* Fetches and caches the ONNX model on first call.
* No transaction data leaves the device.
*/
async function runFraudDetection(params) {
	await initWasm();
	if (!fraudModelBytes) {
		const response = await fetch("/models/fraud_detector.onnx");
		if (!response.ok) throw new Error(`Failed to fetch fraud model: ${response.status}`);
		const buffer = await response.arrayBuffer();
		fraudModelBytes = new Uint8Array(buffer);
	}
	const isCashOut = params.type === "CASH_OUT" ? 1 : 0;
	const isTransfer = params.type === "TRANSFER" ? 1 : 0;
	const t0 = performance.now();
	const probability = runFraudModel(fraudModelBytes, params.amount, params.oldBalanceOrig, params.newBalanceOrig, params.oldBalanceDest, params.newBalanceDest, isCashOut, isTransfer);
	const latencyMs = performance.now() - t0;
	return {
		probability,
		label: probability >= .5 ? "FRAUD" : "LEGITIMATE",
		latencyMs
	};
}
//#endregion
export { runFraudDetection as n, runWasmInference as r, initWasm as t };
