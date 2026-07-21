import { n as require_jsx_runtime } from "../_libs/react+tanstack__react-query.mjs";
import { g as Link } from "../_libs/@tanstack/react-router+[...].mjs";
import { h as ArrowRight } from "../_libs/lucide-react.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/architecture-DaXdelex.js
var import_jsx_runtime = require_jsx_runtime();
function Pipe({ from, to }) {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		style: {
			display: "flex",
			alignItems: "center",
			gap: 8,
			color: "var(--ink-muted)"
		},
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
				className: "mono",
				style: { color: "var(--ink)" },
				children: from
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(ArrowRight, { size: 12 }),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
				className: "mono",
				style: { color: "var(--ink)" },
				children: to
			})
		]
	});
}
function ArchitecturePage() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					maxWidth: 720,
					marginBottom: 40
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "c-badge c-badge-info",
						children: "Architecture"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: {
							fontSize: 42,
							marginTop: 14
						},
						children: "How Crucible runs a model."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						children: "Two implementations of the same engine — native C++17 for benchmarks and embedded targets, and a Rust reimplementation compiled to WebAssembly for the browser. Same graph, same operator semantics, same numeric output on the fixed test set."
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-rule left",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "num",
					children: "01"
				}), " · Request pipeline"]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				className: "c-plate",
				style: { marginBottom: 32 },
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("svg", {
					viewBox: "0 0 900 220",
					style: {
						width: "100%",
						height: "auto"
					},
					role: "img",
					"aria-label": "Request pipeline",
					children: [
						[
							{
								x: 20,
								label: "React page",
								sub: "src/app/fraud"
							},
							{
								x: 200,
								label: "crucible-wasm.ts",
								sub: "TS wrapper"
							},
							{
								x: 400,
								label: "WASM bindings",
								sub: "wasm-bindgen"
							},
							{
								x: 620,
								label: "Rust engine",
								sub: "kernels + scheduler"
							},
							{
								x: 810,
								label: "Score",
								sub: "f32 out"
							}
						].map((s, i, arr) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("g", { children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
								x: s.x,
								y: 70,
								width: 140,
								height: 80,
								fill: "var(--paper)",
								stroke: "var(--ink)"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("text", {
								x: s.x + 70,
								y: 104,
								textAnchor: "middle",
								fontFamily: "Inter Tight, sans-serif",
								fontSize: "13",
								fontWeight: "600",
								fill: "var(--ink)",
								children: s.label
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("text", {
								x: s.x + 70,
								y: 122,
								textAnchor: "middle",
								fontFamily: "JetBrains Mono, monospace",
								fontSize: "10",
								fill: "var(--ink-muted)",
								children: s.sub
							}),
							i < arr.length - 1 && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("line", {
								x1: s.x + 140,
								y1: 110,
								x2: arr[i + 1].x,
								y2: 110,
								stroke: "var(--trace)",
								strokeWidth: "1.5"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("polygon", {
								points: `${arr[i + 1].x - 6},106 ${arr[i + 1].x},110 ${arr[i + 1].x - 6},114`,
								fill: "var(--trace)"
							})] })
						] }, s.label)),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("text", {
							x: 20,
							y: 40,
							fontFamily: "JetBrains Mono, monospace",
							fontSize: "10",
							letterSpacing: "2",
							fill: "var(--ink-muted)",
							children: "SAMPLE PATH · fraud inference"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("line", {
							x1: 20,
							y1: 190,
							x2: 880,
							y2: 190,
							stroke: "var(--rule)",
							strokeDasharray: "3 3"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("text", {
							x: 450,
							y: 210,
							textAnchor: "middle",
							fontFamily: "JetBrains Mono, monospace",
							fontSize: "10",
							fill: "var(--ink-muted)",
							children: "no network hop between any of these boxes"
						})
					]
				})
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-grid-2",
				style: { marginBottom: 32 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Tensors are flat, row-major NCHW."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
							className: "c-muted",
							style: { marginTop: 8 },
							children: [
								"A shape like ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "[N, C, H, W]"
								}),
								" is one contiguous",
								" ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "float32"
								}),
								"buffer. Strides for each axis are precomputed once —",
								" ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "stride[i] = shape[i+1] * stride[i+1]"
								}),
								" — so an index like",
								" ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "(n, c, h, w)"
								}),
								" lowers to a single multiply-add. It matters because every MatMul, every Conv, every bias broadcast is ultimately linear reads through this buffer, and the tighter the memory story the more the SIMD lanes stay full."
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
							className: "c-code",
							style: { marginTop: 12 },
							children: `shape   = [1, 3, 224, 224]
stride  = [3*224*224, 224*224, 224, 1]
offset  = n*stride[0] + c*stride[1] + h*stride[2] + w*stride[3]`
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Execution order is Kahn's algorithm."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
							className: "c-muted",
							style: { marginTop: 8 },
							children: [
								"An ONNX graph is a DAG of operators. Crucible builds an",
								" ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "in-degree[node]"
								}),
								" map, seeds a ready-set with every zero-in-degree node, and pops nodes off in topological order — decrementing successors, appending them when their in-degree hits zero. Deterministic, cycle-detecting, and small."
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
							className: "c-code",
							style: { marginTop: 12 },
							children: `ready = { n : in_degree[n] == 0 }
while ready not empty:
  n = ready.pop()
  execute(n)
  for m in successors(n):
    if --in_degree[m] == 0: ready.add(m)`
						})
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-rule left",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "num",
					children: "02"
				}), " · Data path"]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-two-col",
				style: { marginBottom: 32 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-plate",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-eyebrow",
							style: { color: "var(--risk)" },
							children: "with a hosted model API"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								marginTop: 14,
								display: "flex",
								flexDirection: "column",
								gap: 10
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "tab",
									to: "TLS to your server"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "your server",
									to: "TLS to model host"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "model host",
									to: "raw features in memory"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "model host",
									to: "score back to server"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "your server",
									to: "score back to tab"
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { className: "c-divider" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								fontSize: 13,
								color: "var(--risk)"
							},
							children: "Raw transaction features cross three trust boundaries. The model host becomes an in-scope processor for PCI-DSS / GDPR purposes."
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-plate",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-eyebrow",
							style: { color: "var(--ok)" },
							children: "with crucible"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								marginTop: 14,
								display: "flex",
								flexDirection: "column",
								gap: 10
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "tab (once)",
									to: "fetch .onnx (static CDN)"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "tab",
									to: "wasm execute"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pipe, {
									from: "tab",
									to: "score (in same tab)"
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { className: "c-divider" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								fontSize: 13,
								color: "var(--ok)"
							},
							children: "Raw features never leave the tab. Only the final score is available to the app. The model host is a static file server — not a processor of feature data."
						})
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-rule left",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "num",
					children: "03"
				}), " · Two builds, one engine"]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-spec",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-spec-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "k",
							children: "native core"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "v",
							children: "C++17 · Eigen for GEMM · im2col conv · pthreads optional"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-spec-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "k",
							children: "wasm build"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "v",
							children: "Rust reimplementation · wasm-simd128 · no Eigen · wasm-bindgen glue"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-spec-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "k",
							children: "shared"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "v",
							children: "Same ONNX decoder shape · same operator set · same fixed-seed golden tests"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-spec-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "k",
							children: "parity"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "v",
							children: "Kernels compared bit-for-bit on the fraud model's calibration inputs"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-spec-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "k",
							children: "binary size"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "v",
							children: "Native ≈ 1.4 MB · WASM ≈ 3.1 MB after wasm-opt --O3"
						})]
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					marginTop: 32,
					display: "flex",
					gap: 10,
					flexWrap: "wrap"
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
					to: "/playground",
					className: "c-btn c-btn-primary",
					children: ["Inspect the graph ", /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ArrowRight, { size: 14 })]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
					to: "/benchmark",
					className: "c-btn c-btn-secondary",
					children: "See both builds benchmarked"
				})]
			})
		]
	}) });
}
//#endregion
export { ArchitecturePage as component };
