import { n as require_jsx_runtime } from "../_libs/react+tanstack__react-query.mjs";
import { g as Link } from "../_libs/@tanstack/react-router+[...].mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/story-CYdigG73.js
var import_jsx_runtime = require_jsx_runtime();
function StoryPage() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		style: {
			maxWidth: 780,
			marginLeft: "auto",
			marginRight: "auto"
		},
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
				className: "c-badge c-badge-info",
				children: "Story"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
				className: "c-h2",
				style: {
					fontSize: 44,
					marginTop: 14
				},
				children: "I wanted to run a fraud model inside a browser tab."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				className: "c-muted",
				style: {
					fontSize: 16,
					marginTop: 12
				},
				children: "The existing runtimes couldn't. So I wrote a new one — twice."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				className: "c-divider",
				style: { margin: "32px 0" }
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				style: {
					fontSize: 17,
					lineHeight: 1.7,
					color: "var(--ink)"
				},
				children: "The prompt was small and specific. Score a transaction as fraud or not-fraud, on the client, without shipping the features to a server. Client-side, because the demo brief said \"no server calls,\" and because sending raw account balances across the wire to a hosted inference endpoint has real consequences under PCI-DSS."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				style: {
					fontSize: 17,
					lineHeight: 1.7,
					color: "var(--ink)",
					marginTop: 18
				},
				children: "I tried the obvious thing first: use ONNX Runtime Web. The distributable is over 50 MB before I've loaded a single model. PyTorch's mobile export is 750 MB+ on disk before you strip anything out. Neither is a serious answer for a fraud check that has to fit on a banking landing page. So I stopped looking for a runtime to wrap and started writing one."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					margin: "36px 0",
					padding: "22px 26px",
					background: "var(--paper)",
					border: "1px solid var(--rule)",
					borderLeft: "3px solid var(--trace)"
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "c-eyebrow",
					style: { marginBottom: 8 },
					children: "Constraint"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					style: {
						fontSize: 18,
						color: "var(--ink)",
						lineHeight: 1.5
					},
					children: "A runtime the browser will actually download. Under 5 MB compressed. No native dependencies at load time. Deterministic output."
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
				className: "c-h2",
				style: {
					fontSize: 26,
					marginTop: 40,
					marginBottom: 12
				},
				children: "Two hard problems that don't show on a benchmark chart."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
				className: "c-h3",
				style: {
					marginTop: 24,
					fontSize: 17
				},
				children: "Hand-writing the protobuf decoder."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
				style: {
					fontSize: 16,
					lineHeight: 1.7,
					color: "var(--ink-default)",
					marginTop: 8
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "mono",
						children: ".onnx"
					}),
					" files are protobuf-encoded. The obvious path is to run",
					" ",
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "mono",
						children: "protoc"
					}),
					" against the ONNX ",
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "mono",
						children: ".proto"
					}),
					" ",
					"schema and get generated C++ or Rust readers for free. I didn't want that dependency in the build. So I wrote a decoder from the wire-format spec: read a tag byte, split it into a field number and a wire type, dispatch on the wire type — varint, 64-bit, length-delimited, 32-bit — and repeat. The whole reader is a few hundred lines. It understands only the fields Crucible actually uses; unknown fields are skipped correctly, which is what the spec asks for."
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
				className: "c-h3",
				style: {
					marginTop: 24,
					fontSize: 17
				},
				children: "Keeping two implementations identical."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
				style: {
					fontSize: 16,
					lineHeight: 1.7,
					color: "var(--ink-default)",
					marginTop: 8
				},
				children: [
					"The C++ core uses Eigen for its matmul path. The Rust build for the browser can't rely on Eigen — it's a pure-Rust kernel set that lowers to",
					" ",
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "mono",
						children: "wasm-simd128"
					}),
					". Two implementations means two chances for the numbers to drift. I keep them honest with fixed-seed golden tests: the fraud model gets a fixed input tensor, both implementations execute the graph, and every intermediate tensor is compared bit-for-bit at the representation level for integer tensors and inside a tight epsilon for floats. When the test fails I know before I ship."
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					margin: "40px 0",
					padding: "24px 26px",
					background: "var(--paper-2)",
					border: "1px solid var(--rule)"
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "c-eyebrow",
					style: { marginBottom: 12 },
					children: "The numbers I care about"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-grid-3",
					style: { gap: 12 },
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								fontSize: 22,
								color: "var(--ink)",
								letterSpacing: "-.02em"
							},
							children: "3.1 MB"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-metric-sub",
							children: "runtime binary (wasm)"
						})] }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								fontSize: 22,
								color: "var(--ink)",
								letterSpacing: "-.02em"
							},
							children: "15"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-metric-sub",
							children: "operators implemented"
						})] }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								fontSize: 22,
								color: "var(--ink)",
								letterSpacing: "-.02em"
							},
							children: "1.18 ms"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-metric-sub",
							children: "fraud inference, warm"
						})] })
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
				className: "c-h2",
				style: {
					fontSize: 26,
					marginTop: 32,
					marginBottom: 12
				},
				children: "What I'd do next."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				style: {
					fontSize: 16,
					lineHeight: 1.7,
					color: "var(--ink-default)"
				},
				children: "Int8 quantization on the MatMul path, so the fraud model shrinks another 3–4×. A parity harness in CI so the C++/Rust check runs on every commit, not just when I remember. Biometric matching next — the same \"features never leave the tab\" argument holds even harder for face embeddings."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					marginTop: 36,
					display: "flex",
					gap: 12,
					flexWrap: "wrap"
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
					to: "/architecture",
					className: "c-btn c-btn-primary",
					children: "See the architecture"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
					to: "/roadmap",
					className: "c-btn c-btn-secondary",
					children: "See the roadmap"
				})]
			})
		]
	}) });
}
//#endregion
export { StoryPage as component };
