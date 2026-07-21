import { i as __toESM } from "../_runtime.mjs";
import { n as require_jsx_runtime, r as require_react, t as QueryClientProvider } from "../_libs/react+tanstack__react-query.mjs";
import { _ as useRouter, c as HeadContent, d as createRouter, f as Outlet, g as Link, h as createRootRouteWithContext, m as createFileRoute, p as lazyRouteComponent, s as Scripts } from "../_libs/@tanstack/react-router+[...].mjs";
import { t as QueryClient } from "../_libs/tanstack__query-core.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/router-Bsevoe8v.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var styles_default = "/assets/styles-QvDWBgX7.css";
var crucible_default = "/assets/crucible-DRnn0G9j.css";
function reportLovableError(error, context = {}) {
	if (typeof window === "undefined") return;
	window.__lovableEvents?.captureException?.(error, {
		source: "react_error_boundary",
		route: window.location.pathname,
		...context
	}, {
		mechanism: "react_error_boundary",
		handled: false,
		severity: "error"
	});
}
function NotFoundComponent() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "flex min-h-screen items-center justify-center bg-background px-4",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "max-w-md text-center",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "text-7xl font-bold text-foreground",
					children: "404"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "mt-4 text-xl font-semibold text-foreground",
					children: "Page not found"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-2 text-sm text-muted-foreground",
					children: "The page you're looking for doesn't exist or has been moved."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "mt-6",
					children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Link, {
						to: "/",
						className: "inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90",
						children: "Go home"
					})
				})
			]
		})
	});
}
function ErrorComponent({ error, reset }) {
	console.error(error);
	const router = useRouter();
	(0, import_react.useEffect)(() => {
		reportLovableError(error, { boundary: "tanstack_root_error_component" });
	}, [error]);
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
		className: "flex min-h-screen items-center justify-center bg-background px-4",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "max-w-md text-center",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h1", {
					className: "text-xl font-semibold tracking-tight text-foreground",
					children: "This page didn't load"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-2 text-sm text-muted-foreground",
					children: "Something went wrong on our end. You can try refreshing or head back home."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "mt-6 flex flex-wrap justify-center gap-2",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
						onClick: () => {
							router.invalidate();
							reset();
						},
						className: "inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90",
						children: "Try again"
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: "/",
						className: "inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent",
						children: "Go home"
					})]
				})
			]
		})
	});
}
var Route$8 = createRootRouteWithContext()({
	head: () => ({
		meta: [
			{ charSet: "utf-8" },
			{
				name: "viewport",
				content: "width=device-width, initial-scale=1"
			},
			{ title: "Crucible — ONNX inference engine for the browser" },
			{
				name: "description",
				content: "A from-scratch ONNX runtime. C++17 core with a hand-written protobuf decoder and Kahn's-algorithm graph executor. Pure-Rust reimplementation compiled to WebAssembly. Fraud inference in the browser tab."
			},
			{
				name: "author",
				content: "Crucible"
			},
			{
				property: "og:title",
				content: "Crucible — ONNX inference engine for the browser"
			},
			{
				property: "og:description",
				content: "3.1 MB WASM runtime. Zero server calls. C++ core, Rust WASM build, honest benchmarks."
			},
			{
				property: "og:type",
				content: "website"
			},
			{
				property: "og:site_name",
				content: "Crucible"
			},
			{
				name: "twitter:card",
				content: "summary_large_image"
			},
			{
				name: "theme-color",
				content: "#FAFAF7"
			}
		],
		links: [
			{
				rel: "preconnect",
				href: "https://fonts.googleapis.com"
			},
			{
				rel: "preconnect",
				href: "https://fonts.gstatic.com",
				crossOrigin: "anonymous"
			},
			{
				rel: "stylesheet",
				href: "https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap"
			},
			{
				rel: "stylesheet",
				href: styles_default
			},
			{
				rel: "stylesheet",
				href: crucible_default
			},
			{
				rel: "icon",
				href: "/favicon.svg",
				type: "image/svg+xml"
			}
		]
	}),
	shellComponent: RootShell,
	component: RootComponent,
	notFoundComponent: NotFoundComponent,
	errorComponent: ErrorComponent
});
function RootShell({ children }) {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("html", {
		lang: "en",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("head", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(HeadContent, {}) }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("body", { children: [children, /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Scripts, {})] })]
	});
}
function RootComponent() {
	const { queryClient } = Route$8.useRouteContext();
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)(QueryClientProvider, {
		client: queryClient,
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Outlet, {})
	});
}
var $$splitComponentImporter$7 = () => import("./story-CYdigG73.mjs");
var Route$7 = createFileRoute("/story")({
	head: () => ({
		meta: [
			{ title: "Story · Crucible" },
			{
				name: "description",
				content: "Why Crucible exists: ONNX Runtime is 50MB+, PyTorch is 750MB+, and neither fits in a browser tab. So I wrote it from scratch."
			},
			{
				property: "og:title",
				content: "Crucible — Why This Exists"
			},
			{
				property: "og:description",
				content: "A first-person build story: hand-writing an ONNX protobuf decoder and keeping a C++ engine and a Rust engine bit-for-bit identical."
			},
			{
				property: "og:url",
				content: "/story"
			}
		],
		links: [{
			rel: "canonical",
			href: "/story"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$7, "component")
});
var $$splitComponentImporter$6 = () => import("./roadmap-C9BR7jIW.mjs");
var Route$6 = createFileRoute("/roadmap")({
	head: () => ({
		meta: [
			{ title: "Roadmap · Crucible" },
			{
				name: "description",
				content: "What's shipped, in progress, and planned for Crucible — from the pure-Rust WASM build to biometric matching and OCR."
			},
			{
				property: "og:title",
				content: "Crucible — Roadmap"
			},
			{
				property: "og:description",
				content: "Active development: kernel parity, int8 quantization, biometrics, OCR, WebGPU."
			},
			{
				property: "og:url",
				content: "/roadmap"
			}
		],
		links: [{
			rel: "canonical",
			href: "/roadmap"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$6, "component")
});
var $$splitComponentImporter$5 = () => import("./playground-DG00rt3e.mjs");
var Route$5 = createFileRoute("/playground")({
	head: () => ({ meta: [
		{ title: "Playground · Crucible" },
		{
			name: "description",
			content: "Drop an ONNX model, feed a tensor, inspect the graph — all in the browser."
		},
		{
			property: "og:title",
			content: "Crucible WASM Playground"
		},
		{
			property: "og:description",
			content: "Interactive ONNX runtime running entirely client-side."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$5, "component")
});
var $$splitComponentImporter$4 = () => import("./fraud-DGF4fPaJ.mjs");
var Route$4 = createFileRoute("/fraud")({
	head: () => ({ meta: [
		{ title: "Fraud Detector · Crucible" },
		{
			name: "description",
			content: "Run fraud inference on transactions entirely in-browser via Crucible's WASM ONNX runtime."
		},
		{
			property: "og:title",
			content: "Privacy-First Fraud Detection · Crucible"
		},
		{
			property: "og:description",
			content: "Zero network bytes. Zero data leaks. ML in the tab."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$4, "component")
});
var $$splitComponentImporter$3 = () => import("./docs-BYl0jFa4.mjs");
var Route$3 = createFileRoute("/docs")({
	head: () => ({ meta: [
		{ title: "Operator Docs · Crucible" },
		{
			name: "description",
			content: "Reference documentation for every ONNX operator implemented in the Crucible engine."
		},
		{
			property: "og:title",
			content: "Crucible Operator Reference"
		},
		{
			property: "og:description",
			content: "Every kernel, attribute, and FFI signature."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$3, "component")
});
var $$splitComponentImporter$2 = () => import("./benchmark-Cdwdao10.mjs");
var Route$2 = createFileRoute("/benchmark")({
	head: () => ({ meta: [
		{ title: "Benchmark · Crucible" },
		{
			name: "description",
			content: "Head-to-head benchmarks: Crucible vs ONNX Runtime vs PyTorch across model sizes."
		},
		{
			property: "og:title",
			content: "Crucible Benchmark Console"
		},
		{
			property: "og:description",
			content: "Latency, footprint, cold-start — all measured, all in your browser."
		}
	] }),
	component: lazyRouteComponent($$splitComponentImporter$2, "component")
});
var $$splitComponentImporter$1 = () => import("./architecture-DaXdelex.mjs");
var Route$1 = createFileRoute("/architecture")({
	head: () => ({
		meta: [
			{ title: "Architecture · Crucible" },
			{
				name: "description",
				content: "How Crucible loads an ONNX model, decodes its protobuf bytes, and schedules kernels — both in native C++ and in Rust/WASM."
			},
			{
				property: "og:title",
				content: "Crucible — Architecture"
			},
			{
				property: "og:description",
				content: "Row-major NCHW tensors, Kahn's topological execution, and a hosted-vs-local data-path comparison."
			},
			{
				property: "og:url",
				content: "/architecture"
			}
		],
		links: [{
			rel: "canonical",
			href: "/architecture"
		}]
	}),
	component: lazyRouteComponent($$splitComponentImporter$1, "component")
});
var $$splitComponentImporter = () => import("./routes-CfmlYt7E.mjs");
var Route = createFileRoute("/")({ component: lazyRouteComponent($$splitComponentImporter, "component") });
var StoryRoute = Route$7.update({
	id: "/story",
	path: "/story",
	getParentRoute: () => Route$8
});
var RoadmapRoute = Route$6.update({
	id: "/roadmap",
	path: "/roadmap",
	getParentRoute: () => Route$8
});
var PlaygroundRoute = Route$5.update({
	id: "/playground",
	path: "/playground",
	getParentRoute: () => Route$8
});
var FraudRoute = Route$4.update({
	id: "/fraud",
	path: "/fraud",
	getParentRoute: () => Route$8
});
var DocsRoute = Route$3.update({
	id: "/docs",
	path: "/docs",
	getParentRoute: () => Route$8
});
var BenchmarkRoute = Route$2.update({
	id: "/benchmark",
	path: "/benchmark",
	getParentRoute: () => Route$8
});
var ArchitectureRoute = Route$1.update({
	id: "/architecture",
	path: "/architecture",
	getParentRoute: () => Route$8
});
var rootRouteChildren = {
	IndexRoute: Route.update({
		id: "/",
		path: "/",
		getParentRoute: () => Route$8
	}),
	ArchitectureRoute,
	BenchmarkRoute,
	DocsRoute,
	FraudRoute,
	PlaygroundRoute,
	RoadmapRoute,
	StoryRoute
};
var routeTree = Route$8._addFileChildren(rootRouteChildren)._addFileTypes();
var getRouter = () => {
	return createRouter({
		routeTree,
		context: { queryClient: new QueryClient() },
		scrollRestoration: true,
		defaultPreloadStaleTime: 0
	});
};
//#endregion
export { getRouter };
