import { i as __toESM } from "../_runtime.mjs";
import { n as require_jsx_runtime, r as require_react } from "../_libs/react+tanstack__react-query.mjs";
import { m as Check, t as X } from "../_libs/lucide-react.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
import { n as getChartData, t as getBenchmarkResults } from "./api--5xYD8y3.mjs";
import { a as Bar, c as Tooltip, i as CartesianGrid, l as Legend, n as YAxis, o as Cell, r as XAxis, s as ResponsiveContainer, t as BarChart } from "../_libs/recharts+[...].mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/benchmark-CHYYxq28.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var FOOTPRINT = [
	{
		runtime: "Crucible WASM",
		binaryMB: 3.1,
		coldMs: 48,
		browser: true
	},
	{
		runtime: "Crucible Native",
		binaryMB: 1.4,
		coldMs: 12,
		browser: false
	},
	{
		runtime: "TFLite",
		binaryMB: 2.1,
		coldMs: 35,
		browser: true
	},
	{
		runtime: "ONNX Runtime",
		binaryMB: 51.2,
		coldMs: 820,
		browser: false
	},
	{
		runtime: "PyTorch",
		binaryMB: 756,
		coldMs: 2100,
		browser: false
	}
];
var COLORS = {
	"Crucible Native": "#152A66",
	"ONNX Runtime": "#7A7A73",
	PyTorch: "#B45309"
};
var tooltipStyle = {
	background: "#FFFFFF",
	border: "1px solid #DCDCD3",
	borderRadius: 2,
	color: "#0E0E10",
	fontSize: 12,
	fontFamily: "JetBrains Mono, monospace"
};
function BenchmarkPage() {
	const _bdata = (0, import_react.useMemo)(() => getBenchmarkResults(), []);
	const _crucible = _bdata?.results?.find((r) => r.engine === "crucible");
	const _ort = _bdata?.results?.find((r) => r.engine === "onnxruntime");
	const _torch = _bdata?.results?.find((r) => r.engine === "pytorch");
	if (!_crucible || !_ort || !_torch) return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
			className: "c-h2",
			children: "Benchmark Console"
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
			className: "c-muted",
			children: "No benchmark data available. Run the benchmark suite first."
		})]
	}) });
	const LATENCY = getChartData().map((d) => ({
		size: d.size,
		"Crucible Native": d.crucible,
		"ONNX Runtime": d.onnxruntime,
		PyTorch: d.pytorch
	}));
	const STATS = [
		{
			runtime: "Crucible Native (C++/Eigen)",
			min: _crucible.stats.min_ms,
			max: _crucible.stats.max_ms,
			median: _crucible.stats.median_ms,
			p95: _crucible.stats.p95_ms,
			p99: _crucible.stats.p99_ms,
			mean: _crucible.stats.mean_ms,
			throughput: _crucible.stats.throughput_inf_per_sec
		},
		{
			runtime: "ONNX Runtime (CPU / MLAS)",
			min: _ort.stats.min_ms,
			max: _ort.stats.max_ms,
			median: _ort.stats.median_ms,
			p95: _ort.stats.p95_ms,
			p99: _ort.stats.p99_ms,
			mean: _ort.stats.mean_ms,
			throughput: _ort.stats.throughput_inf_per_sec
		},
		{
			runtime: "PyTorch (CPU / ATen)",
			min: _torch.stats.min_ms,
			max: _torch.stats.max_ms,
			median: _torch.stats.median_ms,
			p95: _torch.stats.p95_ms,
			p99: _torch.stats.p99_ms,
			mean: _torch.stats.mean_ms,
			throughput: _torch.stats.throughput_inf_per_sec
		}
	];
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					marginBottom: 32,
					maxWidth: 720
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "c-badge c-badge-info",
						children: "Performance"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: {
							fontSize: 42,
							marginTop: 14
						},
						children: "Benchmark Console"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "c-muted",
						children: [
							"Head-to-head measurements across MLP model sizes. Mean of 1,000 warm iterations on a Ryzen 7 7840U, single-threaded. Native numbers are the C++/Eigen build; WASM numbers are the pure-Rust build compiled to ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "mono",
								children: "wasm-simd128"
							}),
							" — the two are separate implementations, reported separately."
						]
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-grid-4",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-metric hl",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-label",
								children: "Crucible Native (C++/Eigen)"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-metric-value",
								children: [_crucible.stats.mean_ms.toFixed(1), " ms"]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-metric-sub",
								children: [
									"mean · MobileNetV2 · ",
									_crucible.stats.throughput_inf_per_sec.toFixed(0),
									" inf/s"
								]
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-metric",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-label",
								children: "ONNX Runtime (CPU / MLAS)"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-metric-value",
								children: [_ort.stats.mean_ms.toFixed(1), " ms"]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-sub",
								children: "mean · MobileNetV2"
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-metric",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-label",
								children: "PyTorch (CPU / ATen)"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-metric-value",
								children: [_torch.stats.mean_ms.toFixed(1), " ms"]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-sub",
								children: "mean · MobileNetV2"
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-metric hl",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-label",
								children: "Crucible WASM — Binary Size"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-value",
								children: "3.1 MB"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-metric-sub",
								children: "16× smaller than ONNX Runtime"
							})
						]
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-card",
				style: { marginTop: 28 },
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "c-h3",
						children: "Latency by Model Size"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: { marginBottom: 18 },
						children: "Lower is better. Milliseconds per forward pass."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: "100%",
							height: 340
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(BarChart, {
							data: LATENCY,
							margin: {
								top: 8,
								right: 12,
								left: 0,
								bottom: 8
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
									strokeDasharray: "3 3",
									stroke: "#DCDCD3"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
									dataKey: "size",
									stroke: "#5A5A55",
									style: { fontSize: 12 }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
									stroke: "#5A5A55",
									style: { fontSize: 12 },
									label: {
										value: "ms",
										angle: -90,
										position: "insideLeft",
										fill: "#5A5A55",
										fontSize: 11
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
									contentStyle: tooltipStyle,
									cursor: { fill: "rgba(31,58,138,.08)" }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Legend, { wrapperStyle: { fontSize: 12 } }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
									dataKey: "Crucible Native",
									fill: COLORS["Crucible Native"],
									radius: [
										2,
										2,
										0,
										0
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
									dataKey: "ONNX Runtime",
									fill: COLORS["ONNX Runtime"],
									radius: [
										2,
										2,
										0,
										0
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
									dataKey: "PyTorch",
									fill: COLORS.PyTorch,
									radius: [
										2,
										2,
										0,
										0
									]
								})
							]
						}) })
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-two-col",
				style: { marginTop: 28 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Binary Size"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							style: { marginBottom: 12 },
							children: "MB shipped to the client."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								width: "100%",
								height: 260
							},
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(BarChart, {
								data: FOOTPRINT,
								layout: "vertical",
								margin: {
									top: 8,
									right: 12,
									left: 20,
									bottom: 8
								},
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
										strokeDasharray: "3 3",
										stroke: "#DCDCD3",
										horizontal: false
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
										type: "number",
										stroke: "#5A5A55",
										style: { fontSize: 12 }
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
										type: "category",
										dataKey: "runtime",
										stroke: "#5A5A55",
										style: { fontSize: 12 },
										width: 120
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
										contentStyle: tooltipStyle,
										cursor: { fill: "rgba(31,58,138,.08)" },
										formatter: (v) => `${v} MB`
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
										dataKey: "binaryMB",
										radius: [
											0,
											2,
											2,
											0
										],
										children: FOOTPRINT.map((f) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cell, { fill: f.runtime.startsWith("Crucible") ? "#1F3A8A" : "#B8B8AE" }, f.runtime))
									})
								]
							}) })
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Cold-Start Time"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							style: { marginBottom: 12 },
							children: "Milliseconds to first inference."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								width: "100%",
								height: 260
							},
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(BarChart, {
								data: FOOTPRINT,
								layout: "vertical",
								margin: {
									top: 8,
									right: 12,
									left: 20,
									bottom: 8
								},
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
										strokeDasharray: "3 3",
										stroke: "#DCDCD3",
										horizontal: false
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
										type: "number",
										stroke: "#5A5A55",
										style: { fontSize: 12 }
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
										type: "category",
										dataKey: "runtime",
										stroke: "#5A5A55",
										style: { fontSize: 12 },
										width: 120
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
										contentStyle: tooltipStyle,
										cursor: { fill: "rgba(31,58,138,.08)" },
										formatter: (v) => `${v} ms`
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
										dataKey: "coldMs",
										radius: [
											0,
											2,
											2,
											0
										],
										children: FOOTPRINT.map((f) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cell, { fill: f.runtime.startsWith("Crucible") ? "#152A66" : "#B8B8AE" }, f.runtime))
									})
								]
							}) })
						})
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-card",
				style: { marginTop: 28 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
					className: "c-h3",
					children: "Runtime Footprint"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("table", {
					className: "c-table",
					style: { marginTop: 12 },
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("thead", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", { children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Runtime" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Binary Size" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Cold-Start" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Browser Capable" })
					] }) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", { children: FOOTPRINT.map((f) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
						className: f.runtime.startsWith("Crucible") ? "hl" : "",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
								style: { color: f.runtime.startsWith("Crucible") ? "var(--trace)" : void 0 },
								children: f.runtime
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("td", { children: [f.binaryMB.toFixed(1), " MB"] }),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("td", { children: [f.coldMs, " ms"] }),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: f.browser ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Check, {
								className: "c-check",
								size: 18
							}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(X, {
								className: "c-cross",
								size: 18
							}) })
						]
					}, f.runtime)) })]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-card",
				style: { marginTop: 28 },
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "c-h3",
						children: "Latency Distribution — Medium MLP"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: { marginBottom: 12 },
						children: "1,000 warm iterations. Milliseconds unless noted."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: { overflowX: "auto" },
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("table", {
							className: "c-table",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("thead", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", { children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Runtime" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Min" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Median" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Mean" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "P95" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "P99" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Max" }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", { children: "Throughput (inf/s)" })
							] }) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", { children: STATS.map((s) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
								className: s.runtime.startsWith("Crucible") ? "hl" : "",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										style: { color: s.runtime.startsWith("Crucible") ? "var(--trace)" : void 0 },
										children: s.runtime
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.min }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.median }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.mean }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.p95 }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.p99 }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.max }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", { children: s.throughput.toFixed(1) })
								]
							}, s.runtime)) })]
						})
					})
				]
			})
		]
	}) });
}
//#endregion
export { BenchmarkPage as component };
