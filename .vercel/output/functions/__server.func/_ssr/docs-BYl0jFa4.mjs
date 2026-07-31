import { i as __toESM } from "../_runtime.mjs";
import { n as require_jsx_runtime, r as require_react } from "../_libs/react+tanstack__react-query.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/docs-BYl0jFa4.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var CATEGORIES = [
	{
		key: "math",
		label: "Math Ops",
		ops: [
			{
				name: "MatMul",
				signature: "MatMul(A: T[..., M, K], B: T[..., K, N]) → T[..., M, N]",
				inputs: "A, B tensors with broadcastable batch dims",
				output: "product tensor",
				usage: `// crucible_core/ops/matmul.hpp
Tensor y = crucible::matmul(a, b);
// Rust FFI
let y = crucible::matmul(&a, &b)?;`
			},
			{
				name: "Add",
				signature: "Add(A: T, B: T) → T",
				inputs: "broadcastable tensors A, B",
				output: "elementwise sum",
				usage: `Tensor y = crucible::add(a, b);`
			},
			{
				name: "Mul",
				signature: "Mul(A: T, B: T) → T",
				inputs: "broadcastable tensors A, B",
				output: "elementwise product",
				usage: `Tensor y = crucible::mul(a, b);`
			}
		]
	},
	{
		key: "act",
		label: "Activations",
		ops: [
			{
				name: "Relu",
				signature: "Relu(X: T) → T",
				inputs: "X ∈ T[...]",
				output: "max(0, X)",
				usage: `Tensor y = crucible::relu(x); // SIMD-vectorized via Eigen`
			},
			{
				name: "Sigmoid",
				signature: "Sigmoid(X: T) → T",
				inputs: "X ∈ T[...]",
				output: "1 / (1 + exp(-X))",
				usage: `Tensor y = crucible::sigmoid(x);`
			},
			{
				name: "Softmax",
				signature: "Softmax(X: T, axis: int) → T",
				inputs: "X ∈ T[...], axis of reduction",
				output: "normalized distribution",
				attrs: [{
					name: "axis",
					type: "int",
					note: "reduction axis, default -1"
				}],
				usage: `Tensor y = crucible::softmax(x, /*axis=*/-1);`
			}
		]
	},
	{
		key: "conv",
		label: "Convolution",
		ops: [{
			name: "Conv2D",
			signature: "Conv2D(X: T[N,C,H,W], W: T[K,C,kH,kW], B?: T[K]) → T[N,K,Hout,Wout]",
			inputs: "activation X, kernel W, optional bias B",
			output: "convolved feature map",
			attrs: [
				{
					name: "stride",
					type: "int[2]",
					note: "default [1,1]"
				},
				{
					name: "padding",
					type: "int[4]",
					note: "top, left, bottom, right"
				},
				{
					name: "dilation",
					type: "int[2]",
					note: "default [1,1]"
				},
				{
					name: "groups",
					type: "int",
					note: "default 1"
				}
			],
			usage: `ConvOpts o{.stride={1,1}, .padding={1,1,1,1}};
Tensor y = crucible::conv2d(x, w, b, o);`
		}]
	},
	{
		key: "norm",
		label: "Normalization",
		ops: [{
			name: "BatchNorm",
			signature: "BatchNorm(X, scale, B, mean, var, eps) → T",
			inputs: "activation X and 4 param tensors",
			output: "normalized activation",
			attrs: [{
				name: "epsilon",
				type: "float",
				note: "default 1e-5"
			}],
			usage: `Tensor y = crucible::batchnorm(x, scale, bias, mean, var, 1e-5f);`
		}, {
			name: "LayerNorm",
			signature: "LayerNorm(X, scale, B, axis, eps) → T",
			inputs: "activation X, learnable scale + bias",
			output: "normalized activation",
			attrs: [{
				name: "axis",
				type: "int",
				note: "default -1"
			}, {
				name: "epsilon",
				type: "float",
				note: "default 1e-5"
			}],
			usage: `Tensor y = crucible::layernorm(x, scale, bias, -1, 1e-5f);`
		}]
	},
	{
		key: "pool",
		label: "Pooling",
		ops: [
			{
				name: "MaxPool2D",
				signature: "MaxPool2D(X: T[N,C,H,W]) → T[N,C,Hout,Wout]",
				inputs: "activation X",
				output: "max-pooled tensor",
				attrs: [
					{
						name: "kernel",
						type: "int[2]",
						note: "e.g. [2,2]"
					},
					{
						name: "stride",
						type: "int[2]",
						note: "default = kernel"
					},
					{
						name: "padding",
						type: "int[4]",
						note: "default [0,0,0,0]"
					}
				],
				usage: `Tensor y = crucible::maxpool2d(x, {2,2}, {2,2});`
			},
			{
				name: "AvgPool2D",
				signature: "AvgPool2D(X: T[N,C,H,W]) → T[N,C,Hout,Wout]",
				inputs: "activation X",
				output: "average-pooled tensor",
				usage: `Tensor y = crucible::avgpool2d(x, {2,2}, {2,2});`
			},
			{
				name: "GlobalAvgPool",
				signature: "GlobalAvgPool(X: T[N,C,H,W]) → T[N,C,1,1]",
				inputs: "activation X",
				output: "spatially-collapsed tensor",
				usage: `Tensor y = crucible::global_avg_pool(x);`
			}
		]
	},
	{
		key: "shape",
		label: "Shapes",
		ops: [
			{
				name: "Reshape",
				signature: "Reshape(X: T, shape: int[]) → T[shape]",
				inputs: "X and target shape (at most one -1)",
				output: "view over same data",
				usage: `Tensor y = crucible::reshape(x, {1, -1});`
			},
			{
				name: "Transpose",
				signature: "Transpose(X: T, perm: int[]) → T",
				inputs: "X and permutation",
				output: "axis-permuted tensor",
				attrs: [{
					name: "perm",
					type: "int[]",
					note: "axis permutation"
				}],
				usage: `Tensor y = crucible::transpose(x, {0, 2, 1});`
			},
			{
				name: "Concat",
				signature: "Concat(inputs: T[]*, axis: int) → T",
				inputs: "N tensors sharing all dims except axis",
				output: "concatenated tensor",
				attrs: [{
					name: "axis",
					type: "int",
					note: "concatenation axis"
				}],
				usage: `Tensor y = crucible::concat({a, b, c}, /*axis=*/1);`
			}
		]
	}
];
function DocsPage() {
	const [active, setActive] = (0, import_react.useState)(CATEGORIES[0].key);
	const scrollTo = (key) => {
		setActive(key);
		const el = document.getElementById(`op-cat-${key}`);
		if (el) el.scrollIntoView({
			behavior: "smooth",
			block: "start"
		});
	};
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			style: {
				marginBottom: 32,
				maxWidth: 720
			},
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "c-badge c-badge-info",
					children: "Reference"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "c-h2",
					style: {
						fontSize: 44,
						marginTop: 14
					},
					children: "API & Operator Docs"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					children: "Every operator implemented in the Crucible core, with C++ signatures and FFI examples."
				})
			]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "c-docs",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("aside", {
				className: "c-sidebar",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "c-label",
					style: { marginBottom: 10 },
					children: "Categories"
				}), CATEGORIES.map((c) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
					className: `c-sidebar-item${active === c.key ? " active" : ""}`,
					onClick: () => scrollTo(c.key),
					children: [
						c.label,
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
							style: {
								color: "var(--ink-muted)",
								marginLeft: 6
							},
							children: [
								"(",
								c.ops.length,
								")"
							]
						})
					]
				}, c.key))]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { children: CATEGORIES.map((c) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				id: `op-cat-${c.key}`,
				style: {
					scrollMarginTop: 100,
					marginBottom: 40
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "c-h2",
					style: {
						fontSize: 24,
						marginBottom: 16
					},
					children: c.label
				}), c.ops.map((op) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-op-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								display: "flex",
								justifyContent: "space-between",
								alignItems: "flex-start",
								flexWrap: "wrap",
								gap: 8
							},
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-op-name",
								children: op.name
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-op-sig",
								children: op.signature
							})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "c-badge",
								children: c.label
							})]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-divider",
							style: { margin: "16px 0" }
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-grid-2",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-label",
								children: "Inputs"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-muted mono",
								style: { marginTop: 4 },
								children: op.inputs
							})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-label",
								children: "Output"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-muted mono",
								style: { marginTop: 4 },
								children: op.output
							})] })]
						}),
						op.attrs && op.attrs.length > 0 && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 14 },
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-label",
								children: "Attributes"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("table", {
								className: "c-table",
								style: { marginTop: 4 },
								children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", { children: op.attrs.map((a) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", { children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										style: {
											color: "var(--trace)",
											width: "20%"
										},
										children: a.name
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										style: {
											color: "var(--warn)",
											width: "20%"
										},
										children: a.type
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										style: {
											color: "var(--ink-muted)",
											fontFamily: "inherit"
										},
										children: a.note
									})
								] }, a.name)) })
							})]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 14 },
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-label",
								children: "Usage"
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
								className: "c-code",
								style: { marginTop: 6 },
								children: op.usage
							})]
						})
					]
				}, op.name))]
			}, c.key)) })]
		})]
	}) });
}
//#endregion
export { DocsPage as component };
