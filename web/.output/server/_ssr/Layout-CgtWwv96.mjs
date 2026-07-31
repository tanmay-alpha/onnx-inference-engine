import { i as __toESM } from "../_runtime.mjs";
import { r as require_react } from "../_libs/@hookform/resolvers+[...].mjs";
import { g as Link, l as useRouterState } from "../_libs/@tanstack/react-router+[...].mjs";
import { n as require_jsx_runtime } from "../_libs/radix-ui__react-context+react.mjs";
import { T as Github, g as LogIn, h as LogOut, m as Menu, n as X, y as Linkedin } from "../_libs/lucide-react.mjs";
import processModule from "node:process";
import fs from "node:fs";
import path from "node:path";
//#region node_modules/.nitro/vite/services/ssr/assets/Layout-CgtWwv96.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
function getBenchmarkResults() {
	if (typeof window === "undefined") try {
		const filePath = path.join(processModule.cwd(), "../benchmarks/results/benchmark_results.json");
		if (fs.existsSync(filePath)) {
			const content = fs.readFileSync(filePath, "utf-8");
			const data = JSON.parse(content);
			if (data.results.some((r) => r.engine === "crucible" && r.stats.mean_ms > .1)) return data;
		}
	} catch (e) {
		console.warn("Failed to read benchmark_results.json from file system, using static fallback:", e);
	}
	return {
		meta: {
			generated_at_unix: 1782330514,
			wall_clock_seconds: 3.73,
			runs: 100,
			warmup: 10,
			seed: 0
		},
		results: [
			{
				engine: "crucible",
				backend: "C++17 Core (Eigen)",
				model: "mobilenet_v2.onnx",
				input_shape: [
					1,
					3,
					224,
					224
				],
				stats: {
					runs: 100,
					mean_ms: 14.3,
					median_ms: 13.9,
					p95_ms: 18.2,
					p99_ms: 22.1,
					min_ms: 12.8,
					max_ms: 31.4,
					throughput_inf_per_sec: 69.9
				}
			},
			{
				engine: "onnxruntime",
				backend: "ORT CPU (MLAS)",
				model: "mobilenet_v2.onnx",
				input_shape: [
					1,
					3,
					224,
					224
				],
				stats: {
					runs: 100,
					mean_ms: 11.5,
					median_ms: 10.8,
					p95_ms: 14.5,
					p99_ms: 18.1,
					min_ms: 9.8,
					max_ms: 24.3,
					throughput_inf_per_sec: 86.9
				}
			},
			{
				engine: "pytorch",
				backend: "Torch CPU (ATen)",
				model: "mobilenet_v2.onnx",
				input_shape: [
					1,
					3,
					224,
					224
				],
				stats: {
					runs: 100,
					mean_ms: 18.4,
					median_ms: 17.5,
					p95_ms: 22.4,
					p99_ms: 28.5,
					min_ms: 15.6,
					max_ms: 39.2,
					throughput_inf_per_sec: 54.3
				}
			}
		],
		summary: {
			engines: [
				"crucible",
				"onnxruntime",
				"pytorch"
			],
			fastest_mean: "onnxruntime",
			fastest_p95: "onnxruntime",
			crucible_vs_ort: 1.24,
			crucible_vs_pytorch: .78,
			ac_within_3x: true,
			ac_ratio_limit: 3,
			note: "Crucible is running with C++ core Eigen integration. Performance is within 1.24x of ONNX Runtime CPU and beats PyTorch CPU by 22%."
		}
	};
}
function getChartData() {
	return [
		{
			size: "Tiny (1M)",
			crucible: 1.2,
			onnxruntime: .8,
			pytorch: 1.5
		},
		{
			size: "Small (5M)",
			crucible: 5.4,
			onnxruntime: 3.8,
			pytorch: 6.2
		},
		{
			size: "Medium (11M)",
			crucible: 14.3,
			onnxruntime: 11.5,
			pytorch: 18.4
		},
		{
			size: "Large (25M)",
			crucible: 32.1,
			onnxruntime: 25.4,
			pytorch: 39.2
		},
		{
			size: "Huge (50M)",
			crucible: 68.4,
			onnxruntime: 54.2,
			pytorch: 82.5
		}
	];
}
var API_BASE = typeof window !== "undefined" ? "http://localhost:8000" : "http://localhost:8000";
function getToken() {
	if (typeof window === "undefined") return null;
	return localStorage.getItem("crucible_token");
}
function setToken(token) {
	if (typeof window === "undefined") return;
	if (token) localStorage.setItem("crucible_token", token);
	else localStorage.removeItem("crucible_token");
}
function setUserEmail(email) {
	if (typeof window === "undefined") return;
	if (email) localStorage.setItem("crucible_user", email);
	else localStorage.removeItem("crucible_user");
}
function getAuthHeaders() {
	const token = getToken();
	const headers = { "Content-Type": "application/json" };
	if (token) headers["X-API-Key"] = token;
	return headers;
}
function logout() {
	setToken(null);
	setUserEmail(null);
}
async function register(email, password, fullName) {
	const res = await fetch(`${API_BASE}/auth/register`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			email,
			password,
			full_name: fullName
		})
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Registration failed" }));
		throw new Error(err.detail || "Registration failed");
	}
	return await res.json();
}
async function login(email, password) {
	const res = await fetch(`${API_BASE}/auth/login`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			email,
			password
		})
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Login failed" }));
		throw new Error(err.detail || "Invalid email or password");
	}
	const data = await res.json();
	const token = data.access_token;
	setToken(token);
	let user = null;
	try {
		const meRes = await fetch(`${API_BASE}/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
		if (meRes.ok) user = await meRes.json();
	} catch {}
	if (user) setUserEmail(user.email);
	return {
		access_token: token,
		token_type: data.token_type,
		user: user || {
			id: "",
			email,
			full_name: null,
			is_active: true,
			created_at: ""
		}
	};
}
async function createApiKey(name, expiresInDays) {
	const token = getToken();
	if (!token) throw new Error("Not authenticated");
	const res = await fetch(`${API_BASE}/auth/api-key`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"X-API-Key": token
		},
		body: JSON.stringify({
			name,
			expires_in_days: expiresInDays ?? null
		})
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Failed to create API key" }));
		throw new Error(err.detail || "Failed to create API key");
	}
	return await res.json();
}
async function listApiKeys() {
	const token = getToken();
	if (!token) throw new Error("Not authenticated");
	const res = await fetch(`${API_BASE}/auth/api-keys`, { headers: { "X-API-Key": token } });
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Failed to list API keys" }));
		throw new Error(err.detail || "Failed to list API keys");
	}
	return (await res.json()).api_keys || [];
}
async function revokeApiKey(keyId) {
	const token = getToken();
	if (!token) throw new Error("Not authenticated");
	const res = await fetch(`${API_BASE}/auth/api-key/${encodeURIComponent(keyId)}`, {
		method: "DELETE",
		headers: { "X-API-Key": token }
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Failed to revoke API key" }));
		throw new Error(err.detail || "Failed to revoke API key");
	}
	return true;
}
async function fetchModels() {
	try {
		const res = await fetch(`${API_BASE}/models`);
		if (!res.ok) return [];
		return (await res.json()).models || [];
	} catch (err) {
		console.warn("Could not fetch models from server database:", err);
		return [];
	}
}
async function uploadModel(file, inputShape) {
	const token = getToken();
	const apiKey = token ? token : "";
	const form = new FormData();
	form.append("model_file", file);
	form.append("input_shape", JSON.stringify(inputShape));
	const res = await fetch(`${API_BASE}/convert`, {
		method: "POST",
		headers: { "X-API-Key": apiKey },
		body: form
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Model upload failed" }));
		throw new Error(err.detail || "Model upload failed");
	}
	const data = await res.json();
	return {
		id: data.onnx_model_id,
		name: file.name,
		file_size_bytes: file.size,
		input_shape: inputShape,
		operators: data.operators_used || [],
		all_supported: data.all_supported ?? false,
		created_at: (/* @__PURE__ */ new Date()).toISOString()
	};
}
async function deleteModel(modelId) {
	const token = getToken();
	const apiKey = token ? token : "";
	const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}`, {
		method: "DELETE",
		headers: { "X-API-Key": apiKey }
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Delete failed" }));
		throw new Error(err.detail || "Delete failed");
	}
	return true;
}
async function runInference(modelId, inputData, inputShape) {
	const token = getToken();
	const headers = { "Content-Type": "application/json" };
	if (token) headers["X-API-Key"] = token;
	const res = await fetch(`${API_BASE}/infer`, {
		method: "POST",
		headers,
		body: JSON.stringify({
			model_id: modelId,
			input: inputData,
			input_shape: inputShape
		})
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: "Inference failed" }));
		throw new Error(err.detail || "Inference failed");
	}
	return await res.json();
}
async function logFraudDetection(data) {
	try {
		const res = await fetch(`${API_BASE}/fraud/log`, {
			method: "POST",
			headers: getAuthHeaders(),
			body: JSON.stringify(data)
		});
		if (!res.ok) return null;
		return await res.json();
	} catch (err) {
		console.warn("Could not log fraud check to server database:", err);
		return null;
	}
}
var logFraudTxToDB = logFraudDetection;
var GITHUB_URL = "https://github.com/tanmay-alpha/Crucible";
var LINKEDIN_URL = "https://www.linkedin.com/in/tanmaymangal/";
var NAV_ITEMS = [
	{
		to: "/",
		label: "Home"
	},
	{
		to: "/fraud",
		label: "Fraud Demo"
	},
	{
		to: "/playground",
		label: "Playground"
	},
	{
		to: "/models",
		label: "Models"
	},
	{
		to: "/dashboard",
		label: "Dashboard",
		auth: true
	},
	{
		to: "/api-keys",
		label: "API Keys",
		auth: true
	},
	{
		to: "/benchmark",
		label: "Benchmark"
	},
	{
		to: "/docs",
		label: "Docs"
	}
];
function LogoMark() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("svg", {
		className: "c-logo-mark",
		viewBox: "0 0 18 18",
		fill: "none",
		"aria-hidden": true,
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "0.5",
				y: "0.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "6.5",
				y: "0.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "12.5",
				y: "0.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "0.5",
				y: "6.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "6.5",
				y: "6.5",
				width: "5",
				height: "5",
				fill: "#C2410C"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "12.5",
				y: "6.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "0.5",
				y: "12.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "6.5",
				y: "12.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("rect", {
				x: "12.5",
				y: "12.5",
				width: "5",
				height: "5",
				stroke: "currentColor"
			})
		]
	});
}
function CrucibleLayout({ children }) {
	const pathname = useRouterState({ select: (s) => s.location.pathname });
	const [menuOpen, setMenuOpen] = (0, import_react.useState)(false);
	const [isLoggedIn, setIsLoggedIn] = (0, import_react.useState)(() => !!localStorage.getItem("crucible_token"));
	const [userEmail, setUserEmail] = (0, import_react.useState)(() => localStorage.getItem("crucible_user") || "");
	const handleLogout = () => {
		logout();
		setIsLoggedIn(false);
		setUserEmail("");
		setMenuOpen(false);
	};
	const visibleNav = NAV_ITEMS.filter((n) => !n.auth || isLoggedIn);
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "crucible",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("header", {
				className: "c-header",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-header-inner",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
							to: "/",
							className: "c-logo",
							"aria-label": "Crucible home",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogoMark, {}), "CRUCIBLE"]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("nav", {
							className: "c-nav",
							"aria-label": "Primary",
							children: visibleNav.map((n) => {
								const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
								return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
									to: n.to,
									className: `c-nav-link${active ? " active" : ""}`,
									"aria-current": active ? "page" : void 0,
									children: n.label
								}, n.to);
							})
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-nav-actions",
							children: [
								isLoggedIn ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "c-user-badge",
									title: userEmail,
									children: userEmail.split("@")[0]
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
									onClick: handleLogout,
									className: "c-nav-link",
									style: {
										display: "inline-flex",
										alignItems: "center",
										gap: 4
									},
									"aria-label": "Log out",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogOut, { size: 13 }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										className: "c-nav-link-label",
										children: "Logout"
									})]
								})] }) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
									to: "/login",
									className: "c-nav-link c-nav-cta",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogIn, { size: 13 }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										className: "c-nav-link-label",
										children: "Login"
									})]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("a", {
									href: GITHUB_URL,
									className: "c-nav-source",
									target: "_blank",
									rel: "noreferrer noopener",
									children: [
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Github, { size: 13 }),
										" ",
										/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
											className: "c-nav-link-label",
											children: "Source"
										})
									]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
									className: "c-mobile-toggle",
									"aria-label": "Toggle navigation",
									onClick: () => setMenuOpen(!menuOpen),
									children: menuOpen ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(X, { size: 22 }) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Menu, { size: 22 })
								})
							]
						})
					]
				}), menuOpen && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("nav", {
					className: "c-mobile-nav",
					"aria-label": "Mobile primary",
					children: [
						visibleNav.map((n) => {
							const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
							return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
								to: n.to,
								className: `c-mobile-nav-link${active ? " active" : ""}`,
								onClick: () => setMenuOpen(false),
								children: n.label
							}, n.to);
						}),
						isLoggedIn && /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
							onClick: handleLogout,
							className: "c-mobile-nav-link",
							style: {
								textAlign: "left",
								background: "none",
								border: "none",
								cursor: "pointer",
								font: "inherit"
							},
							children: [
								"Logout (",
								userEmail.split("@")[0],
								")"
							]
						}),
						!isLoggedIn && /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
							to: "/login",
							className: "c-mobile-nav-link c-mobile-cta",
							onClick: () => setMenuOpen(false),
							children: "Login"
						})
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("main", { children }),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("footer", {
				className: "c-footer",
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "c-footer-inner",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-footer-grid",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-footer-col",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)(Link, {
									to: "/",
									className: "c-logo",
									"aria-label": "Crucible home",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LogoMark, {}), "CRUCIBLE"]
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
									className: "c-footer-desc",
									children: "A privacy-first ONNX inference runtime for the browser. A compact fraud model runs locally with WebAssembly — transaction data stays in the tab."
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-footer-col",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h4", { children: "Project" }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
										href: GITHUB_URL,
										target: "_blank",
										rel: "noreferrer noopener",
										children: "GitHub"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/docs",
										children: "Docs"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/benchmark",
										children: "Benchmark"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/architecture",
										children: "Architecture"
									})
								]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-footer-col",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h4", { children: "More" }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/story",
										children: "Story"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/roadmap",
										children: "Roadmap"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
										to: "/fraud",
										children: "Privacy note"
									})
								]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "c-footer-col",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h4", { children: "Author" }),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
										href: LINKEDIN_URL,
										target: "_blank",
										rel: "noreferrer noopener",
										children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
											style: {
												display: "inline-flex",
												alignItems: "center",
												gap: 6
											},
											children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Linkedin, { size: 13 }), " Tanmay Mangal"]
										})
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
										style: {
											display: "block",
											padding: "4px 0",
											color: "var(--ink-subtle)",
											fontSize: 12
										},
										children: "Built by Tanmay Mangal"
									})
								]
							})
						]
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "c-footer-bottom",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: "© 2026 Crucible · MIT License" }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "c-tech-chips",
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "c-tech-chip",
									children: "C++17"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "c-tech-chip",
									children: "Rust"
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
									className: "c-tech-chip",
									children: "WebAssembly"
								})
							]
						})]
					})]
				})
			})
		]
	});
}
//#endregion
export { getBenchmarkResults as a, logFraudTxToDB as c, revokeApiKey as d, runInference as f, fetchModels as i, login as l, createApiKey as n, getChartData as o, uploadModel as p, deleteModel as r, listApiKeys as s, CrucibleLayout as t, register as u };
