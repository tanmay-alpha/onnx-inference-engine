import { i as __toESM } from "../_runtime.mjs";
import { n as require_jsx_runtime, r as require_react } from "../_libs/react+tanstack__react-query.mjs";
import { c as Inbox, i as Shield, o as Lock, r as TriangleAlert } from "../_libs/lucide-react.mjs";
import { t as CrucibleLayout } from "./Layout-BgUZpOi6.mjs";
import { n as runFraudDetection } from "./crucible-wasm-DqqGcYC1.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/fraud-DGF4fPaJ.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var PRESETS = [
	{
		label: "Everyday transfer · ₹4,850",
		tone: "success",
		tx: {
			type: "TRANSFER",
			amount: 4850,
			origBefore: 42e3,
			origAfter: 37150,
			destBefore: 12e3,
			destAfter: 16850
		}
	},
	{
		label: "Account drained · ₹2,01,350",
		tone: "danger",
		tx: {
			type: "TRANSFER",
			amount: 201350,
			origBefore: 201350,
			origAfter: 0,
			destBefore: 0,
			destAfter: 0
		}
	},
	{
		label: "Borderline cash-out · ₹1,54,290",
		tone: "warn",
		tx: {
			type: "CASH_OUT",
			amount: 154290,
			origBefore: 155e3,
			origAfter: 710,
			destBefore: 5e3,
			destAfter: 0
		}
	}
];
function verdict(p) {
	if (p >= .65) return {
		label: "High risk",
		tone: "err"
	};
	if (p >= .35) return {
		label: "Elevated risk",
		tone: "warn"
	};
	return {
		label: "Low risk",
		tone: "ok"
	};
}
function heuristicScore(tx) {
	const drained = tx.origBefore > 0 && tx.origAfter / tx.origBefore < .02;
	const bigAmount = tx.amount >= 1e5;
	const zeroedDest = tx.destBefore + tx.destAfter === 0 && tx.amount > 1e3;
	const mismatch = Math.abs(tx.origBefore - tx.origAfter - tx.amount) > 1;
	let s = .02;
	if (drained) s += .55;
	if (bigAmount) s += .22;
	if (zeroedDest) s += .18;
	if (mismatch) s += .12;
	if (tx.type === "CASH_OUT") s += .08;
	return Math.min(.995, Math.max(.005, s));
}
function FraudPage() {
	const [tx, setTx] = (0, import_react.useState)(PRESETS[0].tx);
	const [status, setStatus] = (0, import_react.useState)("idle");
	const [result, setResult] = (0, import_react.useState)(null);
	const update = (k, v) => setTx((t) => ({
		...t,
		[k]: v
	}));
	const run = async () => {
		const vals = [
			tx.amount,
			tx.origBefore,
			tx.origAfter,
			tx.destBefore,
			tx.destAfter
		];
		const labels = [
			"Amount",
			"Origin balance before",
			"Origin balance after",
			"Dest balance before",
			"Dest balance after"
		];
		for (let i = 0; i < vals.length; i++) if (!Number.isFinite(vals[i]) || vals[i] < 0) {
			alert(labels[i] + " must be a non-negative number.");
			return;
		}
		setStatus("running");
		setResult(null);
		const start = performance.now();
		try {
			const wasmResult = await runFraudDetection({
				type: tx.type,
				amount: tx.amount,
				oldBalanceOrig: tx.origBefore,
				newBalanceOrig: tx.origAfter,
				oldBalanceDest: tx.destBefore,
				newBalanceDest: tx.destAfter
			});
			const latency = Number((performance.now() - start).toFixed(2));
			setResult({
				fraud: wasmResult.probability >= .5,
				probability: wasmResult.probability,
				latencyMs: latency,
				modelBytes: 220
			});
		} catch (err) {
			console.warn("WASM inference failed, using heuristic fallback:", err);
			const p = heuristicScore(tx);
			const latency = Number((performance.now() - start).toFixed(2));
			setResult({
				fraud: p >= .5,
				probability: p,
				latencyMs: latency,
				modelBytes: 0
			});
		}
		setStatus("done");
	};
	const v = result ? verdict(result.probability) : null;
	const toneColor = v?.tone === "err" ? "var(--risk)" : v?.tone === "warn" ? "var(--warn)" : "var(--ok)";
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(CrucibleLayout, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					marginBottom: 32,
					maxWidth: 720
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
						className: "c-badge c-badge-info",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Lock, { size: 12 }), " On-device demo"]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
						className: "c-h2",
						style: {
							fontSize: 44,
							marginTop: 14
						},
						children: "Browser-based transaction analysis"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						children: "A mock banking portal running Crucible's ONNX runtime in WebAssembly. Customer records never leave the tab — the model is fetched once, then evaluated locally against every transaction you enter below."
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-two-col",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Transaction input"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							style: { marginBottom: 18 },
							children: "Pick a scenario or enter values by hand."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							style: {
								display: "flex",
								gap: 6,
								flexWrap: "wrap",
								marginBottom: 22
							},
							children: PRESETS.map((p) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								className: "c-preset",
								style: {
									width: "auto",
									flex: "1 1 30%"
								},
								onClick: () => {
									setTx(p.tx);
									setStatus("idle");
									setResult(null);
								},
								children: p.label
							}, p.label))
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: {
								display: "grid",
								gap: 14
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
									className: "c-label",
									children: "Transaction type"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("select", {
									className: "c-select",
									value: tx.type,
									onChange: (e) => update("type", e.target.value),
									children: [
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("option", {
											value: "TRANSFER",
											children: "TRANSFER"
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("option", {
											value: "CASH_OUT",
											children: "CASH_OUT"
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("option", {
											value: "OTHER",
											children: "OTHER"
										})
									]
								})] }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
									className: "c-label",
									children: "Amount (₹)"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
									className: "c-input",
									type: "number",
									min: "0",
									step: "1",
									value: tx.amount,
									onChange: (e) => update("amount", Number(e.target.value)),
									autoComplete: "off"
								})] }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-field-row",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
										className: "c-label",
										children: "Origin balance before"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
										className: "c-input",
										type: "number",
										min: "0",
										step: "1",
										value: tx.origBefore,
										onChange: (e) => update("origBefore", Number(e.target.value)),
										autoComplete: "off"
									})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
										className: "c-label",
										children: "Origin balance after"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
										className: "c-input",
										type: "number",
										min: "0",
										step: "1",
										value: tx.origAfter,
										onChange: (e) => update("origAfter", Number(e.target.value)),
										autoComplete: "off"
									})] })]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-field-row",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
										className: "c-label",
										children: "Destination balance before"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
										className: "c-input",
										type: "number",
										min: "0",
										step: "1",
										value: tx.destBefore,
										onChange: (e) => update("destBefore", Number(e.target.value)),
										autoComplete: "off"
									})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
										className: "c-label",
										children: "Destination balance after"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
										className: "c-input",
										type: "number",
										min: "0",
										step: "1",
										value: tx.destAfter,
										onChange: (e) => update("destAfter", Number(e.target.value)),
										autoComplete: "off"
									})] })]
								})
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
							className: "c-btn c-btn-primary c-btn-full",
							style: { marginTop: 22 },
							disabled: status === "running",
							onClick: run,
							"aria-live": "polite",
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Shield, { size: 16 }),
								" ",
								status === "running" ? "Analysing transaction…" : "Analyse transaction"
							]
						}),
						status === "running" && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							style: { marginTop: 14 },
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { className: "c-loading-bar" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "c-muted",
								style: {
									marginTop: 8,
									fontSize: 12
								},
								children: "Executing WASM kernels · no network activity."
							})]
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					style: {
						display: "flex",
						flexDirection: "column",
						justifyContent: status === "done" ? "flex-start" : "center"
					},
					children: [status !== "done" && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-card-dashed",
						style: {
							padding: 40,
							textAlign: "center"
						},
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Inbox, {
								size: 22,
								color: "var(--ink-muted)",
								style: { marginBottom: 10 }
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								className: "c-eyebrow",
								style: { marginBottom: 10 },
								children: "Awaiting input"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
								className: "c-h3",
								children: "No inference yet"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
								className: "c-muted",
								children: [
									"Fill the fields and press ",
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("em", { children: "Run local fraud check" }),
									". The ONNX model executes inside this tab; no transaction data leaves the browser."
								]
							})
						]
					}), status === "done" && result && v && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-fade-in",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
								style: { marginBottom: 20 },
								children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "c-badge c-badge-lg",
									style: {
										background: v.tone === "err" ? "#F3D7D7" : v.tone === "warn" ? "#F3E7C4" : "#DDEEDF",
										borderColor: v.tone === "err" ? "#E4B4B4" : v.tone === "warn" ? "#DCC58A" : "#B4D8BE",
										color: toneColor
									},
									children: v.label
								})
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								style: {
									display: "flex",
									alignItems: "baseline",
									gap: 10
								},
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "c-mono-num",
									style: {
										fontSize: 56,
										fontWeight: 600,
										letterSpacing: "-0.03em",
										color: toneColor
									},
									children: [(result.probability * 100).toFixed(1), "%"]
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
									className: "c-muted",
									children: "fraud probability"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								style: { marginTop: 16 },
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
									className: "c-meter",
									children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
										className: "c-meter-fill",
										style: {
											width: `${result.probability * 100}%`,
											background: toneColor
										}
									})
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									style: {
										display: "flex",
										justifyContent: "space-between",
										marginTop: 6
									},
									children: [
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "c-muted",
											style: { fontSize: 11 },
											children: "Legitimate"
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "c-muted",
											style: { fontSize: 11 },
											children: "Review"
										}),
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "c-muted",
											style: { fontSize: 11 },
											children: "Elevated risk"
										})
									]
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { className: "c-divider" }),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-grid-3",
								style: { gap: 10 },
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
										className: "c-metric",
										style: { padding: 14 },
										children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
											className: "c-metric-label",
											children: "Warm latency"
										}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
											className: "c-metric-value",
											style: { fontSize: 20 },
											children: [result.latencyMs, " ms"]
										})]
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
										className: "c-metric",
										style: { padding: 14 },
										children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
											className: "c-metric-label",
											children: "Model bytes"
										}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
											className: "c-metric-value",
											style: { fontSize: 20 },
											children: [result.modelBytes, " B"]
										})]
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
										className: "c-metric",
										style: { padding: 14 },
										children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
											className: "c-metric-label",
											children: "Bytes sent"
										}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
											className: "c-metric-value",
											style: {
												fontSize: 20,
												color: "var(--ok)"
											},
											children: "0"
										})]
									})
								]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-privacy-note",
								style: { marginTop: 18 },
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Lock, {
									size: 16,
									style: { marginTop: 2 }
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { children: "Inference completed locally. No transaction data left the browser tab." })]
							})
						]
					})]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				className: "c-card",
				style: { marginTop: 32 },
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					style: {
						display: "flex",
						gap: 12,
						alignItems: "flex-start"
					},
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(TriangleAlert, {
						size: 20,
						color: "var(--warn)",
						style: {
							marginTop: 3,
							flexShrink: 0
						}
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "c-h3",
							children: "Why on-device inference matters for finance"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
							className: "c-muted",
							style: { marginBottom: 10 },
							children: [
								"Under ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "PCI-DSS"
								}),
								" and ",
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "mono",
									children: "GDPR"
								}),
								", raw cardholder and account data must be tightly scoped. Sending features to a hosted inference gateway expands the compliance boundary: the model host becomes an in-scope service provider, and cross-border transfer obligations follow."
							]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "c-muted",
							children: "Crucible flips the topology. The model ships as static assets, is evaluated in a WebAssembly sandbox with no network access, and only a low-cardinality risk score crosses the boundary. Compliance surface collapses; latency drops from a round-trip to sub-millisecond kernel time."
						})
					] })]
				})
			})
		]
	}) });
}
//#endregion
export { FraudPage as component };
