import { n as require_jsx_runtime } from "../_libs/react+tanstack__react-query.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/roadmap-C9BR7jIW.js
var import_jsx_runtime = require_jsx_runtime();
var ROADMAP = [
	{
		status: "shipped",
		title: "C++17 core with custom tensor class",
		detail: "Row-major flat memory, shape/stride bookkeeping, Eigen-backed matmul and conv2d."
	},
	{
		status: "shipped",
		title: "Hand-written ONNX protobuf decoder",
		detail: "Reads .onnx binary wire format without protoc. Varint, length-delimited fields, embedded tensors."
	},
	{
		status: "shipped",
		title: "Kahn's algorithm graph executor",
		detail: "Topological sort over the DAG, ready-set scheduling, deterministic execution order."
	},
	{
		status: "shipped",
		title: "Pure-Rust WASM reimplementation",
		detail: "Second implementation of the kernels in Rust, compiled to wasm-simd128, no Eigen dependency."
	},
	{
		status: "in-progress",
		title: "Kernel parity harness",
		detail: "Fixed-seed golden tests comparing every C++ kernel output against the Rust kernel bit-for-bit."
	},
	{
		status: "in-progress",
		title: "Quantized int8 MatMul path",
		detail: "Symmetric per-tensor quantization to shrink the fraud model from 220 B to ~64 B."
	},
	{
		status: "planned",
		title: "Biometric matching (face embeddings)",
		detail: "On-device MobileFaceNet inference; cosine similarity in the tab, no biometric data uploaded."
	},
	{
		status: "planned",
		title: "Document OCR (CRNN)",
		detail: "Client-side ID/receipt OCR — pixels never leave the device, only extracted fields cross the wire."
	},
	{
		status: "planned",
		title: "WebGPU backend",
		detail: "Optional GPU kernels for the larger conv-heavy models; keep WASM as the always-available fallback."
	},
	{
		status: "planned",
		title: "Streaming model loader",
		detail: "Range-request the .onnx from CDN and start executing before the full file has landed."
	},
	{
		status: "planned",
		title: "Native binary distribution",
		detail: "Ship the C++ engine as a static library for edge / embedded targets outside the browser."
	}
];
var STATUS_ORDER = [
	"shipped",
	"in-progress",
	"planned"
];
var STATUS_LABEL = {
	shipped: "Shipped",
	"in-progress": "In progress",
	planned: "Planned"
};
var STATUS_COLOR = {
	shipped: "var(--ok)",
	"in-progress": "var(--warn)",
	planned: "var(--ink-muted)"
};
function RoadmapPage() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		style: { maxWidth: 900 },
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
				className: "c-badge c-badge-info",
				children: "Roadmap"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
				className: "c-h2",
				style: {
					fontSize: 42,
					marginTop: 14
				},
				children: "What's done and what's next."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				className: "c-muted",
				style: { maxWidth: "60ch" },
				children: "Crucible is active development, not a closed one-off. Below: everything already in the engine, what I'm working on now, and the on-device use cases the runtime is being aimed at."
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				style: {
					marginTop: 40,
					display: "flex",
					flexDirection: "column",
					gap: 40
				},
				children: STATUS_ORDER.map((status) => {
					const items = ROADMAP.filter((r) => r.status === status);
					return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							alignItems: "baseline",
							gap: 12,
							marginBottom: 16
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
							className: "c-h2",
							style: {
								fontSize: 22,
								margin: 0
							},
							children: STATUS_LABEL[status]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
							className: "mono",
							style: {
								color: STATUS_COLOR[status],
								fontSize: 11,
								letterSpacing: ".18em",
								textTransform: "uppercase"
							},
							children: [
								items.length,
								" item",
								items.length === 1 ? "" : "s"
							]
						})]
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							border: "1px solid var(--rule)",
							background: "var(--paper)"
						},
						children: items.map((r, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								display: "grid",
								gridTemplateColumns: "110px 1fr",
								gap: 20,
								padding: "16px 20px",
								borderTop: i === 0 ? "none" : "1px solid var(--rule)",
								alignItems: "start"
							},
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "mono",
								style: {
									fontSize: 10,
									letterSpacing: ".16em",
									textTransform: "uppercase",
									color: STATUS_COLOR[status],
									paddingTop: 3
								},
								children: status === "in-progress" ? "in prog." : status
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								style: {
									fontSize: 15,
									fontWeight: 600,
									color: "var(--ink)",
									marginBottom: 4
								},
								children: r.title
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-muted",
								children: r.detail
							})] })]
						}, r.title))
					})] }, status);
				})
			})
		]
	}) });
}
//#endregion
export { RoadmapPage as component };
