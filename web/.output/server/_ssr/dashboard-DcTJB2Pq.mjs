import { i as __toESM } from "../_runtime.mjs";
import { r as require_react } from "../_libs/@hookform/resolvers+[...].mjs";
import { g as Link } from "../_libs/@tanstack/react-router+[...].mjs";
import { n as require_jsx_runtime } from "../_libs/radix-ui__react-context+react.mjs";
import { A as Cpu, B as Activity, L as ChartColumn, c as Shield, d as RefreshCw, o as TrendingUp, t as Zap, z as ArrowLeft } from "../_libs/lucide-react.mjs";
import { i as fetchModels, t as CrucibleLayout } from "./Layout-CgtWwv96.mjs";
import { a as YAxis, c as Line, d as Pie, f as Cell, h as Legend, i as LineChart, l as CartesianGrid, m as Tooltip, n as PieChart, o as XAxis, p as ResponsiveContainer, r as BarChart, s as Area, t as AreaChart, u as Bar } from "../_libs/recharts+[...].mjs";
import { t as Button } from "./button-DRsC1qZi.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/dashboard-DcTJB2Pq.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var COLORS = [
	"#c2410c",
	"#152A66",
	"#166534",
	"#b45309",
	"#7c3aed",
	"#0891b2"
];
var fmtNum = (n) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(n);
var fmtPct = (n) => `${(n * 100).toFixed(1)}%`;
function StatCard({ label, value, sub, icon }) {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "c-card",
		style: { padding: 20 },
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "flex",
					alignItems: "center",
					gap: 8,
					marginBottom: 8
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					style: { color: "var(--forge)" },
					children: icon
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "c-muted",
					style: {
						fontSize: 11,
						textTransform: "uppercase",
						letterSpacing: ".12em"
					},
					children: label
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				className: "mono",
				style: {
					fontSize: 28,
					fontWeight: 600,
					color: "var(--ink)",
					letterSpacing: "-.02em"
				},
				children: value
			}),
			sub && /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
				className: "c-muted",
				style: {
					fontSize: 12,
					marginTop: 4
				},
				children: sub
			})
		]
	});
}
function DashboardPage() {
	const [analytics, setAnalytics] = (0, import_react.useState)(null);
	const [fraudHistory, setFraudHistory] = (0, import_react.useState)([]);
	const [loading, setLoading] = (0, import_react.useState)(true);
	const [period, setPeriod] = (0, import_react.useState)(7);
	const [refreshing, setRefreshing] = (0, import_react.useState)(false);
	const [models, setModels] = (0, import_react.useState)([]);
	const loadData = (0, import_react.useCallback)(async (days) => {
		setLoading(true);
		try {
			const [inf, fraud, mods] = await Promise.all([
				fetch(`/analytics/inference?days=${days}`).then((r) => r.json()).catch(() => null),
				fetch(`/analytics/fraud?days=${days}`).then((r) => r.json()).catch(() => null),
				fetchModels()
			]);
			if (inf) setAnalytics(inf);
			else setAnalytics({
				inference: {
					period_days: days,
					data: []
				},
				fraud: {
					period_days: days,
					data: []
				},
				models: []
			});
			setFraudHistory(fraud?.history || []);
			setModels(mods);
		} catch {
			setAnalytics({
				inference: {
					period_days: days,
					data: []
				},
				fraud: {
					period_days: days,
					data: []
				},
				models: []
			});
			setFraudHistory([]);
			setModels([]);
		} finally {
			setLoading(false);
		}
	}, []);
	(0, import_react.useEffect)(() => {
		loadData(period);
	}, [period, loadData]);
	const handleRefresh = async () => {
		setRefreshing(true);
		await loadData(period);
		setRefreshing(false);
	};
	const totalInferences = analytics?.inference.data.reduce((s, d) => s + d.count, 0) || 0;
	const totalFraud = fraudHistory.length;
	const fraudCount = fraudHistory.filter((f) => f.probability >= .5).length;
	const avgLatency = analytics?.inference.data.length ? analytics.inference.data.reduce((s, d) => s + d.avg_latency_ms, 0) / analytics.inference.data.length : 0;
	const latencyBins = (0, import_react.useMemo)(() => {
		const bins = [
			{
				range: "0-1ms",
				count: 0
			},
			{
				range: "1-5ms",
				count: 0
			},
			{
				range: "5-10ms",
				count: 0
			},
			{
				range: "10-50ms",
				count: 0
			},
			{
				range: "50ms+",
				count: 0
			}
		];
		const latencies = fraudHistory.map((f) => f.latency_ms);
		analytics?.inference.data.forEach((d) => {
			latencies.push(d.avg_latency_ms);
		});
		latencies.forEach((ms) => {
			if (ms < 1) bins[0].count++;
			else if (ms < 5) bins[1].count++;
			else if (ms < 10) bins[2].count++;
			else if (ms < 50) bins[3].count++;
			else bins[4].count++;
		});
		return bins.filter((b) => b.count > 0);
	}, [fraudHistory, analytics]);
	const modelPieData = (0, import_react.useMemo)(() => {
		if (!models.length) return [];
		return models.map((m) => ({
			name: m.name.replace(".onnx", ""),
			value: m.usage_count
		})).filter((m) => m.value > 0);
	}, [models]);
	const inferenceChartData = (0, import_react.useMemo)(() => {
		return (analytics?.inference.data || []).map((d) => ({
			...d,
			dateLabel: new Date(d.date).toLocaleDateString("en", {
				month: "short",
				day: "numeric"
			})
		}));
	}, [analytics]);
	const fraudChartData = (0, import_react.useMemo)(() => {
		return (analytics?.fraud.data || []).map((d) => ({
			...d,
			dateLabel: new Date(d.date).toLocaleDateString("en", {
				month: "short",
				day: "numeric"
			})
		}));
	}, [analytics]);
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(CrucibleLayout, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		className: "c-container",
		style: {
			paddingTop: 48,
			paddingBottom: 56
		},
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			style: {
				marginBottom: 32,
				display: "flex",
				justifyContent: "space-between",
				alignItems: "center",
				flexWrap: "wrap",
				gap: 16
			},
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "c-eyebrow",
					children: "PLATFORM OVERVIEW"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "c-h2",
					style: { fontSize: 36 },
					children: "Analytics Dashboard"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					style: { marginTop: 8 },
					children: "Inference volume, fraud detection metrics, and model performance."
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "flex",
					gap: 8,
					alignItems: "center",
					flexWrap: "wrap"
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						style: {
							display: "flex",
							gap: 4,
							alignItems: "center"
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "c-muted",
							style: { fontSize: 12 },
							children: "Period:"
						}), [
							7,
							14,
							30
						].map((d) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
							className: `c-preset ${period === d ? "active" : ""}`,
							style: {
								padding: "6px 14px",
								width: "auto",
								background: period === d ? "var(--forge-tint)" : void 0,
								borderColor: period === d ? "var(--trace)" : void 0
							},
							onClick: () => setPeriod(d),
							children: [d, "d"]
						}, d))]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Button, {
						variant: "secondary",
						onClick: handleRefresh,
						disabled: refreshing,
						style: { padding: "6px 14px" },
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RefreshCw, {
							size: 14,
							className: refreshing ? "c-spin" : ""
						}), "Refresh"]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
						to: "/",
						className: "c-btn c-btn-secondary",
						style: {
							padding: "6px 14px",
							display: "inline-flex",
							alignItems: "center",
							gap: 6
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(ArrowLeft, { size: 14 }), " Home"]
					})
				]
			})]
		}), loading ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			style: {
				textAlign: "center",
				padding: 40
			},
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { className: "c-loading-bar" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				className: "c-muted",
				style: { marginTop: 12 },
				children: "Loading analytics..."
			})]
		}) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
					gap: 16,
					marginBottom: 32
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(StatCard, {
						label: "Total Inferences",
						value: fmtNum(totalInferences),
						sub: `Last ${period} days`,
						icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Activity, { size: 16 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(StatCard, {
						label: "Fraud Detected",
						value: fmtPct(fraudCount / Math.max(totalFraud, 1)),
						sub: `${fraudCount} of ${totalFraud} checks`,
						icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(TrendingUp, { size: 16 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(StatCard, {
						label: "Avg Latency",
						value: `${avgLatency.toFixed(1)}ms`,
						sub: "Per inference",
						icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Zap, { size: 16 })
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(StatCard, {
						label: "Active Models",
						value: fmtNum(models.length),
						sub: "Registered",
						icon: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cpu, { size: 16 })
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
					gap: 20,
					marginBottom: 24
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h3", {
						className: "c-h3",
						style: {
							marginBottom: 16,
							display: "flex",
							alignItems: "center",
							gap: 8
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(ChartColumn, {
							size: 16,
							style: { color: "var(--forge)" }
						}), "Inference Volume"]
					}), inferenceChartData.length > 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: "100%",
							height: 260
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(AreaChart, {
							data: inferenceChartData,
							margin: {
								top: 5,
								right: 5,
								left: -10,
								bottom: 5
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("defs", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("linearGradient", {
									id: "inferenceGrad",
									x1: "0",
									y1: "0",
									x2: "0",
									y2: "1",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("stop", {
										offset: "5%",
										stopColor: "#c2410c",
										stopOpacity: .15
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("stop", {
										offset: "95%",
										stopColor: "#c2410c",
										stopOpacity: 0
									})]
								}) }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
									strokeDasharray: "3 3",
									stroke: "#d6cfbe"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
									dataKey: "dateLabel",
									stroke: "#6b655b",
									style: { fontSize: 11 },
									tick: { fontSize: 11 }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
									stroke: "#6b655b",
									style: { fontSize: 11 },
									allowDecimals: false
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, { contentStyle: {
									background: "#fbf7ee",
									border: "1px solid #d6cfbe",
									borderRadius: 2,
									fontSize: 12,
									fontFamily: "var(--f-mono)"
								} }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Area, {
									type: "monotone",
									dataKey: "count",
									stroke: "#c2410c",
									strokeWidth: 2,
									fill: "url(#inferenceGrad)",
									name: "Inferences"
								})
							]
						}) })
					}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							textAlign: "center",
							padding: 40
						},
						children: "No inference data for this period. Run some inferences to see stats."
					})]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h3", {
						className: "c-h3",
						style: {
							marginBottom: 16,
							display: "flex",
							alignItems: "center",
							gap: 8
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Shield, {
							size: 16,
							style: { color: "var(--warn)" }
						}), "Fraud Detection Trend"]
					}), fraudChartData.length > 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: "100%",
							height: 260
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(LineChart, {
							data: fraudChartData,
							margin: {
								top: 5,
								right: 5,
								left: -10,
								bottom: 5
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
									strokeDasharray: "3 3",
									stroke: "#d6cfbe"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
									dataKey: "dateLabel",
									stroke: "#6b655b",
									style: { fontSize: 11 }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
									stroke: "#6b655b",
									style: { fontSize: 11 },
									domain: [0, 1]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
									formatter: (v, name) => {
										if (name === "avg_probability") return [(v * 100).toFixed(1) + "%", "Avg Probability"];
										return [v, name];
									},
									contentStyle: {
										background: "#fbf7ee",
										border: "1px solid #d6cfbe",
										borderRadius: 2,
										fontSize: 12,
										fontFamily: "var(--f-mono)"
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Legend, { wrapperStyle: { fontSize: 11 } }),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Line, {
									type: "monotone",
									dataKey: "total",
									stroke: "#6b655b",
									strokeWidth: 1.5,
									dot: false,
									name: "Total checks"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Line, {
									type: "monotone",
									dataKey: "fraud_count",
									stroke: "#b91c1c",
									strokeWidth: 2,
									dot: false,
									name: "Flagged"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Line, {
									type: "monotone",
									dataKey: "avg_probability",
									stroke: "#b45309",
									strokeWidth: 1.5,
									strokeDasharray: "4 3",
									dot: false,
									name: "Avg probability"
								})
							]
						}) })
					}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							textAlign: "center",
							padding: 40
						},
						children: "No fraud data for this period. Run fraud checks to see trends."
					})]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				style: {
					display: "grid",
					gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
					gap: 20,
					marginBottom: 24
				},
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h3", {
						className: "c-h3",
						style: {
							marginBottom: 16,
							display: "flex",
							alignItems: "center",
							gap: 8
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Zap, {
							size: 16,
							style: { color: "var(--forge)" }
						}), "Latency Distribution"]
					}), latencyBins.length > 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: "100%",
							height: 260
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(BarChart, {
							data: latencyBins,
							margin: {
								top: 5,
								right: 5,
								left: -10,
								bottom: 5
							},
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CartesianGrid, {
									strokeDasharray: "3 3",
									stroke: "#d6cfbe"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(XAxis, {
									dataKey: "range",
									stroke: "#6b655b",
									style: { fontSize: 11 }
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(YAxis, {
									stroke: "#6b655b",
									style: { fontSize: 11 },
									allowDecimals: false
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
									formatter: (v) => [`${v} inferences`, "Count"],
									contentStyle: {
										background: "#fbf7ee",
										border: "1px solid #d6cfbe",
										borderRadius: 2,
										fontSize: 12,
										fontFamily: "var(--f-mono)"
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Bar, {
									dataKey: "count",
									fill: "#c2410c",
									radius: [
										2,
										2,
										0,
										0
									],
									name: "Inferences",
									children: latencyBins.map((_, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cell, { fill: COLORS[i % COLORS.length] }, i))
								})
							]
						}) })
					}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							textAlign: "center",
							padding: 40
						},
						children: "No latency data available. Run inferences to see distribution."
					})]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-card",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h3", {
						className: "c-h3",
						style: {
							marginBottom: 16,
							display: "flex",
							alignItems: "center",
							gap: 8
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cpu, {
							size: 16,
							style: { color: "var(--trace)" }
						}), "Model Usage Breakdown"]
					}), modelPieData.length > 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						style: {
							width: "100%",
							height: 260
						},
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ResponsiveContainer, { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(PieChart, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Pie, {
							data: modelPieData,
							dataKey: "value",
							nameKey: "name",
							cx: "50%",
							cy: "50%",
							outerRadius: 90,
							label: ({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`,
							labelLine: {
								stroke: "#6b655b",
								strokeWidth: .5
							},
							style: { fontSize: 11 },
							children: modelPieData.map((_, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Cell, {
								fill: COLORS[i % COLORS.length],
								stroke: "#fbf7ee",
								strokeWidth: 1
							}, i))
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Tooltip, {
							formatter: (v) => [`${v} inferences`, "Usage"],
							contentStyle: {
								background: "#fbf7ee",
								border: "1px solid #d6cfbe",
								borderRadius: 2,
								fontSize: 12,
								fontFamily: "var(--f-mono)"
							}
						})] }) })
					}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "c-muted",
						style: {
							textAlign: "center",
							padding: 40
						},
						children: "No model usage data yet. Upload and run models to see analytics."
					})]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "c-card",
				style: { marginBottom: 24 },
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
					className: "c-h3",
					style: { marginBottom: 16 },
					children: "Recent Fraud Checks"
				}), fraudHistory.length ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					style: { overflowX: "auto" },
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("table", {
						style: {
							width: "100%",
							borderCollapse: "collapse",
							fontSize: 13
						},
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("thead", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
							style: { borderBottom: "2px solid var(--rule)" },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
									style: {
										textAlign: "left",
										padding: "8px 12px",
										fontSize: 11,
										textTransform: "uppercase",
										letterSpacing: ".1em",
										color: "var(--ink-muted)"
									},
									children: "Time"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
									style: {
										textAlign: "left",
										padding: "8px 12px",
										fontSize: 11,
										textTransform: "uppercase",
										letterSpacing: ".1em",
										color: "var(--ink-muted)"
									},
									children: "Type"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
									style: {
										textAlign: "right",
										padding: "8px 12px",
										fontSize: 11,
										textTransform: "uppercase",
										letterSpacing: ".1em",
										color: "var(--ink-muted)"
									},
									children: "Amount"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
									style: {
										textAlign: "right",
										padding: "8px 12px",
										fontSize: 11,
										textTransform: "uppercase",
										letterSpacing: ".1em",
										color: "var(--ink-muted)"
									},
									children: "Probability"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
									style: {
										textAlign: "center",
										padding: "8px 12px",
										fontSize: 11,
										textTransform: "uppercase",
										letterSpacing: ".1em",
										color: "var(--ink-muted)"
									},
									children: "Verdict"
								})
							]
						}) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", { children: fraudHistory.slice(0, 20).map((f) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
							style: { borderBottom: "1px solid var(--rule)" },
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
									style: {
										padding: "10px 12px",
										color: "var(--ink-muted)",
										fontSize: 12
									},
									children: new Date(f.created_at).toLocaleString("en-US", {
										month: "short",
										day: "numeric",
										hour: "2-digit",
										minute: "2-digit"
									})
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
									style: { padding: "10px 12px" },
									children: f.tx_type
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("td", {
									style: {
										textAlign: "right",
										padding: "10px 12px",
										fontVariantNumeric: "tabular-nums"
									},
									children: ["₹", fmtNum(f.amount)]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("td", {
									style: {
										textAlign: "right",
										padding: "10px 12px",
										fontVariantNumeric: "tabular-nums",
										color: f.probability >= .5 ? "var(--risk)" : f.probability >= .35 ? "var(--warn)" : "var(--ok)"
									},
									children: [(f.probability * 100).toFixed(1), "%"]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
									style: {
										textAlign: "center",
										padding: "10px 12px"
									},
									children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										style: {
											fontSize: 11,
											padding: "2px 8px",
											borderRadius: 2,
											background: f.probability >= .5 ? "#fadcdc" : f.probability >= .35 ? "#f3e7c4" : "#ddeedf",
											border: "1px solid " + (f.probability >= .5 ? "#e4b4b4" : f.probability >= .35 ? "#dcc58a" : "#b4d8be"),
											color: f.probability >= .5 ? "var(--risk)" : f.probability >= .35 ? "var(--warn)" : "var(--ok)",
											fontFamily: "var(--f-mono)"
										},
										children: f.verdict
									})
								})
							]
						}, f.id)) })]
					})
				}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "c-muted",
					style: {
						textAlign: "center",
						padding: 30
					},
					children: "No fraud detection history. Run fraud checks to see data here."
				})]
			})
		] })]
	}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("style", { children: `
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      ` })] });
}
//#endregion
export { DashboardPage as component };
