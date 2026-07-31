import { i as __toESM } from "../_runtime.mjs";
import { i as performance_default, r as require_react } from "../_libs/@hookform/resolvers+[...].mjs";
import { n as require_jsx_runtime } from "../_libs/radix-ui__react-context+react.mjs";
import { A as Cpu, C as History, F as ChevronDown, P as ChevronRight, i as Upload, l as Settings, p as Play, u as Server } from "../_libs/lucide-react.mjs";
import { f as runInference, i as fetchModels, t as CrucibleLayout } from "./Layout-CgtWwv96.mjs";
import { r as runWasmInference, t as initWasm } from "./crucible-wasm-DqqGcYC1.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/playground-Cx4NgPHu.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var DEFAULT_MODEL = {
	name: "fraud_mlp_v3.onnx",
	irVersion: 8,
	opset: 17,
	producer: "pytorch 2.3.0",
	bytes: 220,
	nodes: [
		{
			op: "MatMul",
			inputs: "input, W0[7,16]",
			output: "h0_pre",
			attrs: { transB: "0" }
		},
		{
			op: "Add",
			inputs: "h0_pre, b0[16]",
			output: "h0_bias"
		},
		{
			op: "Relu",
			inputs: "h0_bias",
			output: "h0"
		},
		{
			op: "MatMul",
			inputs: "h0, W1[16,8]",
			output: "h1_pre"
		},
		{
			op: "Add",
			inputs: "h1_pre, b1[8]",
			output: "h1_bias"
		},
		{
			op: "Relu",
			inputs: "h1_bias",
			output: "h1"
		},
		{
			op: "MatMul",
			inputs: "h1, W2[8,1]",
			output: "logit"
		},
		{
			op: "Sigmoid",
			inputs: "logit",
			output: "output"
		}
	]
};
function PlaygroundPage() {
	const [modelName, setModelName] = (0, import_react.useState)(DEFAULT_MODEL.name);
	const [modelBytes, setModelBytes] = (0, import_react.useState)(null);
	const [shape, setShape] = (0, import_react.useState)([1, 7]);
	const [values, setValues] = (0, import_react.useState)("0.31, 0.55, 0.02, 1.0, 0.88, 0.12, 0.44");
	const [running, setRunning] = (0, import_react.useState)(false);
	const [error, setError] = (0, import_react.useState)(null);
	const [result, setResult] = (0, import_react.useState)(null);
	const [expanded, setExpanded] = (0, import_react.useState)(null);
	const [dragActive, setDragActive] = (0, import_react.useState)(false);
	const [mode, setMode] = (0, import_react.useState)("wasm");
	const [sampleMode, setSampleMode] = (0, import_react.useState)("custom");
	const [serverModels, setServerModels] = (0, import_react.useState)([]);
	const [selectedModelId, setSelectedModelId] = (0, import_react.useState)("");
	const [history, setHistory] = (0, import_react.useState)([]);
	const [showHistory, setShowHistory] = (0, import_react.useState)(false);
	const dropRef = (0, import_react.useRef)(null);
	const fileInputRef = (0, import_react.useRef)(null);
	const nodes = DEFAULT_MODEL.nodes;
	(0, import_react.useEffect)(() => {
		fetchModels().then((data) => {
			setServerModels(data);
			if (data.length > 0 && !selectedModelId) setSelectedModelId(data[0].id);
		});
	}, []);
	const parsedValues = (0, import_react.useMemo)(() => values.split(/[\s,]+/).filter(Boolean).map(Number), [values]);
	const expectedSize = shape.reduce((a, b) => a * b, 1);
	const valid = parsedValues.length === expectedSize && parsedValues.every((v) => !Number.isNaN(v));
	const applySample = (0, import_react.useCallback)((mode) => {
		setSampleMode(mode);
		const size = shape.reduce((a, b) => a * b, 1);
		if (mode === "zeros") setValues(Array.from({ length: size }, () => "0").join(", "));
		else if (mode === "random") {
			const vals = Array.from({ length: size }, () => Math.random().toFixed(4));
			setValues(vals.join(", "));
		}
	}, [shape]);
	const handleModelFile = (0, import_react.useCallback)((f) => {
		setModelName(f.name);
		setError(null);
		const reader = new FileReader();
		reader.onload = (ev) => {
			const buf = ev.target?.result;
			if (buf instanceof ArrayBuffer) setModelBytes(new Uint8Array(buf));
		};
		reader.onerror = () => setError(`Failed to read file: ${f.name}`);
		reader.readAsArrayBuffer(f);
	}, []);
	const onDrop = (e) => {
		e.preventDefault();
		setDragActive(false);
		const f = e.dataTransfer.files[0];
		if (f) handleModelFile(f);
	};
	const runWasm = (0, import_react.useCallback)(async () => {
		if (!modelBytes) throw new Error("No model loaded. Drop an .onnx file or click the drop zone to choose one.");
		const inputData = new Float32Array(parsedValues);
		const t0 = performance_default.now();
		const output = await runWasmInference(modelBytes, inputData, shape);
		const latency = performance_default.now() - t0;
		return {
			output: Array.from(output),
			shape: shape.slice(1).concat([output.length]),
			latency: Number(latency.toFixed(3))
		};
	}, [
		modelBytes,
		parsedValues,
		shape
	]);
	const runServer = (0, import_react.useCallback)(async () => {
		if (!selectedModelId) throw new Error("No server model selected. Upload a model or select one from the list.");
		const t0 = performance_default.now();
		const res = await runInference(selectedModelId, parsedValues, shape);
		const latency = performance_default.now() - t0;
		return {
			output: res.output,
			shape: res.shape,
			latency: Number(latency.toFixed(3))
		};
	}, [
		selectedModelId,
		parsedValues,
		shape
	]);
	const runBoth = (0, import_react.useCallback)(async () => {
		try {
			return {
				...await runServer(),
				engine: "server (C++ backend)"
			};
		} catch {
			return {
				...await runWasm(),
				engine: "wasm-simd128 (fallback)"
			};
		}
	}, [runServer, runWasm]);
	const run = async () => {
		if (!valid) return;
		setRunning(true);
		setResult(null);
		setError(null);
		if (mode === "wasm" || mode === "both") try {
			await initWasm();
		} catch (e) {
			console.warn("WASM init failed:", e);
			if (mode === "wasm") {
				setError("WebAssembly runtime unavailable. Try server mode.");
				setRunning(false);
				return;
			}
		}
		try {
			let inferenceResult;
			if (mode === "wasm") inferenceResult = {
				...await runWasm(),
				engine: "wasm-simd128"
			};
			else if (mode === "server") inferenceResult = {
				...await runServer(),
				engine: "server (C++ backend)"
			};
			else inferenceResult = await runBoth();
			setResult({
				latencyMs: inferenceResult.latency,
				output: inferenceResult.output,
				shape: inferenceResult.shape,
				engine: inferenceResult.engine
			});
			const entry = {
				id: crypto.randomUUID(),
				timestamp: Date.now(),
				model: modelName,
				mode,
				values: [...parsedValues],
				output: inferenceResult.output,
				latency: inferenceResult.latency,
				shape: inferenceResult.shape
			};
			setHistory((prev) => [entry, ...prev].slice(0, 5));
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		} finally {
			setRunning(false);
		}
	};
	const updateShape = (i, v) => {
		setShape((s) => s.map((x, idx) => {
			if (idx !== i) return x;
			const parsed = Number.parseInt(String(v), 10);
			if (Number.isNaN(parsed) || parsed < 1) return 1;
			return parsed;
		}));
	};
	const restoreFromHistory = (entry) => {
		setValues(entry.values.map((v) => String(v)).join(", "));
		setShape(entry.shape);
		setModelName(entry.model);
		setResult({
			latencyMs: entry.latency,
			output: entry.output,
			shape: entry.shape,
			engine: entry.mode === "wasm" ? "wasm-simd128 (from history)" : "server (from history)"
		});
		setShowHistory(false);
	};
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					marginBottom: 32,
					maxWidth: 720,
					display: "flex",
					justifyContent: "space-between",
					alignItems: "flex-start",
					flexWrap: "wrap",
					gap: 12
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "c-badge c-badge-info",
						children: "Developer Console"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: {
							fontSize: 44,
							marginTop: 14
						},
						children: "WASM Inference Playground"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "c-muted",
						children: [
							"Drop an ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "mono",
								children: ".onnx"
							}),
							" model, define input shape and tensor values, then execute a forward pass inside the browser sandbox or via the server."
						]
					})
				] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
					onClick: () => setShowHistory(!showHistory),
					className: "c-btn c-btn-ghost",
					style: {
						display: "inline-flex",
						alignItems: "center",
						gap: 6
					},
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)(History, { size: 15 }),
						"History (",
						history.length,
						")"
					]
				})]
			}),
			showHistory && history.length > 0 && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-card",
				style: { marginBottom: 24 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h3", {
					className: "c-h3",
					style: { marginBottom: 12 },
					children: [
						"Recent Inferences (",
						history.length,
						")"
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					style: {
						display: "flex",
						flexDirection: "column",
						gap: 8
					},
					children: history.map((entry) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-history-item",
						onClick: () => restoreFromHistory(entry),
						style: {
							display: "flex",
							alignItems: "center",
							gap: 12,
							padding: "10px 12px",
							border: "1px solid var(--rule)",
							borderRadius: 2,
							cursor: "pointer",
							transition: "background 0.12s"
						},
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								style: {
									fontSize: 10,
									padding: "2px 6px",
									borderRadius: 2,
									background: entry.mode === "wasm" ? "#ddeedf" : "#fce8d5",
									color: entry.mode === "wasm" ? "#166534" : "#9a3412",
									fontFamily: "var(--f-mono)",
									flexShrink: 0
								},
								children: entry.mode.toUpperCase()
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								style: {
									fontFamily: "var(--f-mono)",
									fontSize: 12,
									color: "var(--ink)"
								},
								children: entry.model
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
								className: "c-muted",
								style: {
									fontSize: 11,
									flex: 1
								},
								children: [
									"[",
									entry.shape.join(", "),
									"]"
								]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
								className: "mono",
								style: {
									fontSize: 12,
									color: "var(--forge)",
									flexShrink: 0
								},
								children: [entry.latency, "ms"]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
								className: "mono",
								style: {
									fontSize: 12,
									color: entry.output[0] >= .5 ? "var(--risk)" : "var(--ok)",
									flexShrink: 0
								},
								children: [(entry.output[0] * 100).toFixed(1), "%"]
							})
						]
					}, entry.id))
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-two-col",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Inference Console"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 12 },
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
								className: "c-label",
								children: "Inference Mode"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								style: {
									display: "flex",
									gap: 6,
									flexWrap: "wrap"
								},
								children: [
									{
										value: "wasm",
										label: "WASM (Browser)",
										icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cpu, { size: 13 })
									},
									{
										value: "server",
										label: "Server (C++)",
										icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Server, { size: 13 })
									},
									{
										value: "both",
										label: "Auto-fallback",
										icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Settings, { size: 13 })
									}
								].map((m) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
									className: `c-preset ${mode === m.value ? "active" : ""}`,
									style: {
										width: "auto",
										flex: "1 1 30%",
										display: "inline-flex",
										alignItems: "center",
										gap: 4,
										justifyContent: "center",
										background: mode === m.value ? "var(--forge-tint)" : void 0,
										borderColor: mode === m.value ? "var(--trace)" : void 0
									},
									onClick: () => setMode(m.value),
									children: [m.icon, m.label]
								}, m.value))
							})]
						}),
						(mode === "server" || mode === "both") && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 12 },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
									className: "c-label",
									children: "Server Model"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("select", {
									className: "c-select",
									value: selectedModelId,
									onChange: (e) => setSelectedModelId(e.target.value),
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("option", {
										value: "",
										children: "-- Select a server model --"
									}), serverModels.map((m) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("option", {
										value: m.id,
										children: [
											m.name,
											" (",
											m.operators.length,
											" ops)"
										]
									}, m.id))]
								}),
								serverModels.length === 0 && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
									className: "c-muted",
									style: {
										fontSize: 11,
										marginTop: 4
									},
									children: "No server models found. Upload one via the Models page."
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							ref: dropRef,
							className: `c-drop${dragActive ? " hover" : ""}`,
							role: "button",
							tabIndex: 0,
							"aria-label": "Drop an ONNX model file here, or press Enter to browse",
							onClick: () => fileInputRef.current?.click(),
							onKeyDown: (e) => {
								if (e.key === "Enter" || e.key === " ") {
									e.preventDefault();
									fileInputRef.current?.click();
								}
							},
							onDragOver: (e) => {
								e.preventDefault();
								setDragActive(true);
							},
							onDragLeave: () => setDragActive(false),
							onDrop,
							style: {
								marginTop: 12,
								cursor: "pointer"
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Upload, {
									size: 22,
									style: { marginBottom: 8 }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									style: {
										fontSize: 14,
										fontWeight: 600,
										color: "var(--ink)"
									},
									children: [
										"Drop ",
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "mono",
											children: ".onnx"
										}),
										" model here"
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-muted",
									style: {
										fontSize: 12,
										marginTop: 4
									},
									children: [
										"Loaded:",
										" ",
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "mono",
											style: { color: "var(--trace)" },
											children: modelName
										})
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
									ref: fileInputRef,
									type: "file",
									accept: ".onnx",
									style: { display: "none" },
									onChange: (e) => {
										const f = e.target.files?.[0];
										if (f) handleModelFile(f);
										e.target.value = "";
									}
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 18 },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
									className: "c-label",
									children: "Input Shape"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									style: {
										display: "flex",
										gap: 8,
										alignItems: "center",
										flexWrap: "wrap"
									},
									children: [
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "mono",
											style: { color: "var(--ink-muted)" },
											children: "["
										}),
										shape.map((d, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
											className: "c-input",
											type: "number",
											value: d,
											onChange: (e) => updateShape(i, Number(e.target.value)),
											style: {
												width: 72,
												textAlign: "center"
											}
										}, i)),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "mono",
											style: { color: "var(--ink-muted)" },
											children: "]"
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
											className: "c-btn c-btn-ghost",
											onClick: () => setShape((s) => [...s, 1]),
											style: { padding: "6px 10px" },
											children: "+ dim"
										}),
										shape.length > 1 && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
											className: "c-btn c-btn-ghost",
											onClick: () => setShape((s) => s.slice(0, -1)),
											style: { padding: "6px 10px" },
											children: "− dim"
										})
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
									className: "c-muted",
									style: {
										fontSize: 12,
										marginTop: 6
									},
									children: [
										"Expecting",
										" ",
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "mono",
											style: { color: "var(--trace)" },
											children: expectedSize
										}),
										" ",
										"values."
									]
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 14 },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
									className: "c-label",
									children: "Sample Input"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
									style: {
										display: "flex",
										gap: 6,
										marginBottom: 8
									},
									children: [
										{
											value: "zeros",
											label: "Zeros"
										},
										{
											value: "random",
											label: "Random"
										},
										{
											value: "custom",
											label: "Custom"
										}
									].map((s) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
										className: `c-preset ${sampleMode === s.value ? "active" : ""}`,
										style: {
											width: "auto",
											padding: "5px 12px",
											background: sampleMode === s.value ? "var(--forge-tint)" : void 0,
											borderColor: sampleMode === s.value ? "var(--trace)" : void 0
										},
										onClick: () => applySample(s.value),
										children: s.label
									}, s.value))
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("textarea", {
									className: "c-textarea",
									value: values,
									onChange: (e) => {
										setValues(e.target.value);
										setSampleMode("custom");
									},
									placeholder: "Comma or whitespace-separated numbers",
									rows: 3
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
									className: "c-muted",
									style: {
										fontSize: 12,
										marginTop: 6,
										color: valid ? "var(--ok)" : "var(--risk)"
									},
									children: valid ? `✓ ${parsedValues.length} values parsed` : `✗ Got ${parsedValues.length} — need ${expectedSize}`
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
							className: "c-btn c-btn-primary c-btn-full",
							onClick: run,
							disabled: !valid || running,
							style: { marginTop: 18 },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Play, { size: 15 }),
								" ",
								running ? "Running..." : "Run Inference"
							]
						}),
						running && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-loading-bar",
							style: { marginTop: 12 }
						}),
						error && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							role: "alert",
							className: "c-fade-in",
							style: {
								marginTop: 12,
								color: "var(--risk)",
								fontSize: 13
							},
							children: error
						}),
						result && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-fade-in",
							style: { marginTop: 22 },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
									className: "c-label",
									children: "Output Tensor"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-grid-3",
									style: {
										gap: 10,
										marginTop: 8
									},
									children: [
										/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
											className: "c-metric",
											style: { padding: 12 },
											children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
												className: "c-metric-label",
												children: "Latency"
											}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
												className: "c-metric-value",
												style: { fontSize: 18 },
												children: [result.latencyMs, " ms"]
											})]
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
											className: "c-metric",
											style: { padding: 12 },
											children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
												className: "c-metric-label",
												children: "Shape"
											}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
												className: "c-metric-value",
												style: { fontSize: 18 },
												children: [
													"[",
													result.shape.join(", "),
													"]"
												]
											})]
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
											className: "c-metric",
											style: { padding: 12 },
											children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
												className: "c-metric-label",
												children: "Engine"
											}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
												className: "c-metric-value",
												style: {
													fontSize: 14,
													color: "var(--trace)"
												},
												children: result.engine
											})]
										})
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("pre", {
									className: "c-code",
									style: { marginTop: 12 },
									children: [
										`{
  `,
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "s",
											children: "\"output\""
										}),
										`: [`,
										result.output.map((v) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "n",
											children: v
										}, v)),
										`],
  `,
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "s",
											children: "\"shape\""
										}),
										`: [`,
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "n",
											children: result.shape.join(", ")
										}),
										`],
  `,
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "s",
											children: "\"backend\""
										}),
										`: `,
										/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
											className: "s",
											children: [
												"\"",
												result.engine,
												"\""
											]
										}),
										`
}`
									]
								})
							]
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Model Graph Inspector"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-grid-2",
							style: {
								gap: 10,
								marginTop: 12
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-metric",
									style: { padding: 12 },
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-label",
										children: "Model"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "mono",
										style: {
											fontSize: 13,
											marginTop: 6,
											color: "var(--trace)"
										},
										children: modelName
									})]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-metric",
									style: { padding: 12 },
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-label",
										children: "Producer"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "mono",
										style: {
											fontSize: 13,
											marginTop: 6,
											color: "var(--trace)"
										},
										children: DEFAULT_MODEL.producer
									})]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-metric",
									style: { padding: 12 },
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-label",
										children: "IR Version"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-value",
										style: { fontSize: 20 },
										children: DEFAULT_MODEL.irVersion
									})]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-metric",
									style: { padding: 12 },
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-label",
										children: "Opset"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-metric-value",
										style: { fontSize: 20 },
										children: DEFAULT_MODEL.opset
									})]
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-label",
							style: { marginTop: 20 },
							children: [
								"Graph Nodes (",
								nodes.length,
								")"
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								display: "grid",
								gap: 6,
								marginTop: 8
							},
							children: nodes.map((n, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
								className: "c-node",
								style: { width: "100%" },
								onClick: () => setExpanded(expanded === i ? null : i),
								"aria-expanded": expanded === i,
								"aria-controls": `node-detail-${i}`,
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									style: {
										display: "flex",
										alignItems: "center",
										gap: 10
									},
									children: [
										expanded === i ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ChevronDown, { size: 14 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ChevronRight, { size: 14 }),
										/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
											style: {
												color: "var(--ink-muted)",
												width: 22
											},
											className: "mono",
											children: ["#", i]
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "c-node-op",
											children: n.op
										})
									]
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
									className: "c-node-shape",
									children: ["→ ", n.output]
								})]
							}), expanded === i && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								id: `node-detail-${i}`,
								className: "c-node-details c-fade-in",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										style: { color: "var(--ink-muted)" },
										children: "inputs: "
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										className: "mono",
										style: { color: "var(--trace)" },
										children: n.inputs
									})] }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										style: { color: "var(--ink-muted)" },
										children: "output: "
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										className: "mono",
										style: { color: "var(--ok)" },
										children: n.output
									})] }),
									n.attrs && Object.entries(n.attrs).map(([k, v]) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
										style: { color: "var(--ink-muted)" },
										children: [k, ": "]
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										className: "mono",
										style: { color: "var(--warn)" },
										children: v
									})] }, k))
								]
							})] }, i))
						})
					]
				})]
			})
		]
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
        .c-history-item:hover {
          background: var(--paper-2);
        }
      ` })] });
}
//#endregion
export { PlaygroundPage as component };
