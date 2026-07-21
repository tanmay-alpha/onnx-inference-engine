import { n as require_jsx_runtime } from "../_libs/react+tanstack__react-query.mjs";
import { g as Link, l as useRouterState } from "../_libs/@tanstack/react-router+[...].mjs";
import { l as Github, s as Linkedin } from "../_libs/lucide-react.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/Layout-BgUZpOi6.js
var import_jsx_runtime = require_jsx_runtime();
var GITHUB_URL = "https://github.com/tanmay-alpha/Crucible";
var LINKEDIN_URL = "https://www.linkedin.com/in/tanmaymangal/";
var NAV = [
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
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "crucible",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)("header", {
				className: "c-header",
				children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
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
							children: NAV.map((n) => {
								const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
								return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
									to: n.to,
									className: `c-nav-link${active ? " active" : ""}`,
									"aria-current": active ? "page" : void 0,
									children: n.label
								}, n.to);
							})
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("a", {
							href: GITHUB_URL,
							className: "c-nav-source",
							target: "_blank",
							rel: "noreferrer noopener",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Github, { size: 13 }), " SOURCE"]
						})
					]
				})
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
export { CrucibleLayout as t };
