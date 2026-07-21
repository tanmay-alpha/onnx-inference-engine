import { n as require_jsx_runtime } from "../_libs/react+tanstack__react-query.mjs";
import { g as Link } from "../_libs/@tanstack/react-router+[...].mjs";
import { d as Cpu, h as ArrowRight, o as Lock, u as GitBranch } from "../_libs/lucide-react.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/routes-CfmlYt7E.js
var import_jsx_runtime = require_jsx_runtime();
var inr = new Intl.NumberFormat("en-IN", {
	style: "currency",
	currency: "INR",
	maximumFractionDigits: 0
});
var fmt = (n) => inr.format(n).replace("₹", "₹");
var RECENT = [
	{
		id: "001",
		kind: "transfer",
		amount: 8151,
		verdict: "low risk",
		ms: .66,
		risk: "ok"
	},
	{
		id: "002",
		kind: "transfer",
		amount: 201350,
		verdict: "elevated risk",
		ms: .61,
		risk: "warn"
	},
	{
		id: "003",
		kind: "cash-out",
		amount: 154290,
		verdict: "elevated risk",
		ms: .83,
		risk: "warn"
	},
	{
		id: "004",
		kind: "cash-out",
		amount: 7060,
		verdict: "low risk",
		ms: 1.32,
		risk: "ok"
	},
	{
		id: "005",
		kind: "payment",
		amount: 5492,
		verdict: "low risk",
		ms: 1.05,
		risk: "ok"
	}
];
function InferenceTerminal() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "c-console",
		style: { height: "100%" },
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "c-console-bar",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "flex",
					alignItems: "center",
					gap: 10
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "dots",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { className: "dot" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { className: "dot" }),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { className: "dot on" })
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "c-console-title",
					children: "local inference session"
				})]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
				className: "c-console-title",
				style: { color: "var(--console-accent)" },
				children: "● live"
			})]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "c-console-body",
			style: {
				display: "flex",
				flexDirection: "column",
				gap: 16
			},
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						fontFamily: "var(--f-mono)",
						fontSize: 12,
						color: "var(--console-mute)",
						lineHeight: 1.7
					},
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-accent)" },
							children: "$"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-ink)" },
							children: "crucible run"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-ok)" },
							children: "fraud_mlp_v3.onnx"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-mute)" },
							children: "--amount"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-warn)" },
							children: "₹51,610"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-mute)" },
							children: "--channel"
						}),
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							style: { color: "var(--console-warn)" },
							children: "transfer"
						})
					]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "model loaded"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val s",
							children: "fraud_mlp_v3.onnx"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "runtime"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val",
							children: "wasm-simd128"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "latency"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val n",
							children: "1.18 ms"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "network sent"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val s",
							children: "0 bytes"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "prediction"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val warn",
							children: "elevated risk"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-console-row",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-key",
							children: "probability"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-console-val warn",
							children: "82.4%"
						})]
					})
				] }),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "c-console-title",
					style: { marginBottom: 8 },
					children: "recent activity"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-console-log",
					style: { maxHeight: "none" },
					children: [RECENT.map((r) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "grid",
							gridTemplateColumns: "34px 68px 1fr 100px 62px",
							gap: 8,
							alignItems: "baseline"
						},
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								style: { color: "var(--console-mute)" },
								children: r.id
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								style: { color: "var(--console-ink)" },
								children: r.kind
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								style: { color: "var(--console-warn)" },
								children: fmt(r.amount)
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: r.risk === "ok" ? "ok" : "warn",
								children: r.verdict
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
								style: {
									color: "var(--console-mute)",
									textAlign: "right"
								},
								children: [r.ms.toFixed(2), " ms"]
							})
						]
					}, r.id)), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							marginTop: 8,
							color: "var(--console-accent)"
						},
						children: ["$ ", /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { className: "c-console-caret" })]
					})]
				})] })
			]
		})]
	});
}
function Home() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [
		/* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
			className: "c-container",
			style: {
				paddingTop: 72,
				paddingBottom: 48
			},
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-hero-split",
				style: {
					gap: 56,
					alignItems: "center"
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "c-eyebrow",
						children: "PRIVATE BROWSER INFERENCE"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h1", {
						className: "c-hero-title",
						children: [
							"Fraud checks that run ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("em", { children: "in the browser" }),
							"."
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-hero-lede",
						children: "Crucible runs a compact ONNX fraud model locally with WebAssembly. Transaction data stays in the tab. No inference server, no raw features leaving the device."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							gap: 10,
							flexWrap: "wrap"
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
							to: "/fraud",
							className: "c-btn c-btn-primary c-btn-lg",
							children: ["Run fraud check ", /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ArrowRight, { size: 14 })]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
							to: "/playground",
							className: "c-btn c-btn-secondary c-btn-lg",
							children: "Open model playground"
						})]
					})
				] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					style: { display: "flex" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(InferenceTerminal, {})
				})]
			})
		}),
		/* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
			className: "c-container",
			style: {
				paddingTop: 8,
				paddingBottom: 56
			},
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "repeat(4, 1fr)",
					gap: 0,
					border: "1px solid var(--rule)",
					background: "var(--paper)"
				},
				className: "c-proof-strip",
				children: [
					{
						v: "3.1 MB",
						k: "runtime"
					},
					{
						v: "1.18 ms",
						k: "sample inference"
					},
					{
						v: "0 bytes",
						k: "sent during inference"
					},
					{
						v: "8 nodes",
						k: "fraud graph"
					}
				].map((m, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						padding: "22px 24px",
						borderLeft: i === 0 ? "none" : "1px solid var(--rule)"
					},
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "mono",
						style: {
							fontSize: 24,
							fontWeight: 500,
							color: "var(--ink)",
							letterSpacing: "-.02em"
						},
						children: m.v
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "mono",
						style: {
							fontSize: 10,
							letterSpacing: ".18em",
							textTransform: "uppercase",
							color: "var(--ink-muted)",
							marginTop: 6
						},
						children: m.k
					})]
				}, m.k))
			})
		}),
		/* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
			className: "c-container",
			style: {
				paddingTop: 24,
				paddingBottom: 56
			},
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "repeat(3, 1fr)",
					gap: 20
				},
				className: "c-hero-split",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
						to: "/fraud",
						className: "c-card c-card-hover",
						style: {
							display: "flex",
							flexDirection: "column",
							justifyContent: "space-between",
							minHeight: 210
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							style: {
								fontSize: 20,
								marginBottom: 10
							},
							children: "Fraud Demo"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							children: "Try Indian transaction examples and run a fraud check fully inside the browser."
						})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								marginTop: 24,
								color: "var(--forge)",
								fontSize: 11,
								letterSpacing: ".14em",
								textTransform: "uppercase"
							},
							children: "Run fraud check →"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
						to: "/playground",
						className: "c-card c-card-hover",
						style: {
							display: "flex",
							flexDirection: "column",
							justifyContent: "space-between",
							minHeight: 210
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							style: {
								fontSize: 20,
								marginBottom: 10
							},
							children: "Playground"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							children: "Inspect model inputs, graph steps, and browser-side execution behaviour."
						})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								marginTop: 24,
								color: "var(--forge)",
								fontSize: 11,
								letterSpacing: ".14em",
								textTransform: "uppercase"
							},
							children: "Inspect model →"
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
						to: "/benchmark",
						className: "c-card c-card-hover",
						style: {
							display: "flex",
							flexDirection: "column",
							justifyContent: "space-between",
							minHeight: 210
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							style: {
								fontSize: 20,
								marginBottom: 10
							},
							children: "Benchmark"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							children: "Compare runtime size, latency, and cold-start against heavier runtimes."
						})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "mono",
							style: {
								marginTop: 24,
								color: "var(--forge)",
								fontSize: 11,
								letterSpacing: ".14em",
								textTransform: "uppercase"
							},
							children: "View results →"
						})]
					})
				]
			})
		}),
		/* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
			className: "c-container",
			style: {
				paddingTop: 24,
				paddingBottom: 56
			},
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "1.2fr 1fr",
					gap: 56,
					alignItems: "start"
				},
				className: "c-hero-split",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h2", {
					className: "c-h2",
					style: {
						fontFamily: "var(--f-serif)",
						fontSize: "clamp(28px, 3.4vw, 42px)",
						lineHeight: 1.1
					},
					children: [
						"Built so the data stays ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("em", { children: "where it starts" }),
						"."
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					style: {
						fontSize: 16,
						marginTop: 18,
						maxWidth: "52ch"
					},
					children: "Crucible ships the model as static bytes and evaluates it locally in the browser sandbox. Only the score is returned to the app. Raw transaction features do not leave the device."
				})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					style: {
						display: "flex",
						flexDirection: "column",
						gap: 14
					},
					children: [
						{
							icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cpu, { size: 16 }),
							t: "Local WebAssembly execution",
							d: "Kernels run inside the browser's WASM sandbox."
						},
						{
							icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Lock, { size: 16 }),
							t: "No inference network hop",
							d: "Zero bytes are transmitted for scoring."
						},
						{
							icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(GitBranch, { size: 16 }),
							t: "Inspectable ONNX graph",
							d: "Every node and shape is visible in the Playground."
						}
					].map((p) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "grid",
							gridTemplateColumns: "36px 1fr",
							gap: 14,
							alignItems: "start",
							paddingBottom: 14,
							borderBottom: "1px solid var(--rule)"
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-feature-icon-square",
							style: {
								width: 30,
								height: 30,
								color: "var(--forge)",
								borderColor: "var(--forge)"
							},
							children: p.icon
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								fontWeight: 600,
								color: "var(--ink)",
								fontSize: 14
							},
							children: p.t
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "c-muted",
							style: {
								fontSize: 13,
								marginTop: 2
							},
							children: p.d
						})] })]
					}, p.t))
				})]
			})
		})
	] });
}
//#endregion
export { Home as component };
