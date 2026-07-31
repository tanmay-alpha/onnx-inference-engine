globalThis.__nitro_main__ = import.meta.url;
import { a as FastResponse, n as HTTPError, r as defineLazyEventHandler, t as H3Core } from "./_libs/h3+rou3+srvx.mjs";
import { t as HookableCore } from "./_libs/hookable.mjs";
//#region #nitro-vite-setup
function lazyService(loader) {
	let promise, mod;
	return { fetch(req) {
		if (mod) return mod.fetch(req);
		if (!promise) promise = loader().then((_mod) => mod = _mod.default || _mod);
		return promise.then((mod) => mod.fetch(req));
	} };
}
var services = { ["ssr"]: lazyService(() => import("./_ssr/ssr.mjs")) };
globalThis.__nitro_vite_envs__ = services;
//#endregion
//#region #nitro/virtual/public-assets-data
var public_assets_data_default = {
	"/assets/architecture-CdHy-NY3.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"1e97-+t7oqS76BUdS6SRCl6wIfi1jZlU\"",
		"mtime": "2026-07-31T13:22:22.296Z",
		"size": 7831,
		"path": "../public/assets/architecture-CdHy-NY3.js"
	},
	"/favicon.svg": {
		"type": "image/svg+xml",
		"etag": "\"318-NdMesMUtcPM9z4hGZiPA6IIYQy8\"",
		"mtime": "2026-07-04T12:58:14.549Z",
		"size": 792,
		"path": "../public/favicon.svg"
	},
	"/assets/benchmark-Cf55XP_5.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"2286-S29vosPWJqnVgIm44gSE8mvB2kI\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 8838,
		"path": "../public/assets/benchmark-Cf55XP_5.js"
	},
	"/assets/crucible-DzLgedlU.css": {
		"type": "text/css; charset=utf-8",
		"etag": "\"4bfc-jbiyxTKcWEKylhcHo67do34IaDg\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 19452,
		"path": "../public/assets/crucible-DzLgedlU.css"
	},
	"/assets/crucible-wasm-qKIlQbA1.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"dd1-6W7MmPm8SDt2Aer1WKE7ytWWVhY\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 3537,
		"path": "../public/assets/crucible-wasm-qKIlQbA1.js"
	},
	"/assets/fraud-VzRdXIck.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"277a-omTenaG610d4hL/LWiNOB6HUlOk\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 10106,
		"path": "../public/assets/fraud-VzRdXIck.js"
	},
	"/assets/docs-T3AMe3w7.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"1aa5-niaasz2MbYUhCikxbHaoiU/x3Lo\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 6821,
		"path": "../public/assets/docs-T3AMe3w7.js"
	},
	"/assets/dashboard-CSx0uBW5.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"ed27-2ZndDG93Ai7f+6RIJYOIIbRUZgQ\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 60711,
		"path": "../public/assets/dashboard-CSx0uBW5.js"
	},
	"/assets/BarChart-C6mJ3q2c.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"5a59d-hakHNDQW5Jrjf2Y95nbDB1MnKLE\"",
		"mtime": "2026-07-31T13:22:22.296Z",
		"size": 370077,
		"path": "../public/assets/BarChart-C6mJ3q2c.js"
	},
	"/assets/lock-B0e_Y9uR.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"c3-anEBz0OWLOhj8bwg3aYc0N/4UaU\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 195,
		"path": "../public/assets/lock-B0e_Y9uR.js"
	},
	"/assets/roadmap-B22sZNcN.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"e7a-RPVpOd4F1vunySOLW+ou58D2gKw\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 3706,
		"path": "../public/assets/roadmap-B22sZNcN.js"
	},
	"/assets/playground-DiDgCazH.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"3c69-r8cHQRjmbbOti6J5l3fTCAQMBFc\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 15465,
		"path": "../public/assets/playground-DiDgCazH.js"
	},
	"/assets/story-DHtM25C1.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"15ec-BYjm3brxlHr7e8RS0gplzjPtjQw\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 5612,
		"path": "../public/assets/story-DHtM25C1.js"
	},
	"/assets/routes-Dpa5MCRj.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"264c-ILrZQnPv2dOk/PYB+5WHc5s9MdU\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 9804,
		"path": "../public/assets/routes-Dpa5MCRj.js"
	},
	"/wasm/.gitignore": {
		"type": "text/plain; charset=utf-8",
		"etag": "\"1-31gkjEFPNCyB4Fa0C+4S0XoIv2E\"",
		"mtime": "2026-07-10T20:29:18.514Z",
		"size": 1,
		"path": "../public/wasm/.gitignore"
	},
	"/assets/styles-BY_3sGxz.css": {
		"type": "text/css; charset=utf-8",
		"etag": "\"11174-hJAYrcPjKan3lEs8UewItG5TsUQ\"",
		"mtime": "2026-07-31T13:22:22.299Z",
		"size": 70004,
		"path": "../public/assets/styles-BY_3sGxz.css"
	},
	"/wasm/crucible_wasm.d.ts": {
		"type": "video/mp2t",
		"etag": "\"82c-ywBXQTPZs2/NtkVOyYud2BtcLB0\"",
		"mtime": "2026-07-21T14:01:36.688Z",
		"size": 2092,
		"path": "../public/wasm/crucible_wasm.d.ts"
	},
	"/models/fraud_detector.onnx": {
		"type": "text/plain; charset=utf-8",
		"etag": "\"c3-Vgz+UCs4I8Ij9X3RacGiL1r6a7o\"",
		"mtime": "2026-07-04T09:36:05.531Z",
		"size": 195,
		"path": "../public/models/fraud_detector.onnx"
	},
	"/wasm/crucible_wasm.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"1e20-rhKMYGJ0uwdNT+/OtdOGurwLSnU\"",
		"mtime": "2026-07-10T20:31:05.837Z",
		"size": 7712,
		"path": "../public/wasm/crucible_wasm.js"
	},
	"/wasm/crucible_wasm_bg.wasm": {
		"type": "application/wasm",
		"etag": "\"104fd-aGYG+kykfB5hOqu/YKhpoZJNr2o\"",
		"mtime": "2026-07-10T20:29:22.007Z",
		"size": 66813,
		"path": "../public/wasm/crucible_wasm_bg.wasm"
	},
	"/wasm/crucible_wasm_bg.wasm.d.ts": {
		"type": "video/mp2t",
		"etag": "\"2c1-Lsb3WBDLxbaYL+pHsvhBn2DUfNs\"",
		"mtime": "2026-07-10T20:31:05.814Z",
		"size": 705,
		"path": "../public/wasm/crucible_wasm_bg.wasm.d.ts"
	},
	"/wasm/package.json": {
		"type": "application/json",
		"etag": "\"163-9AcQgPPiQbw5f/BXKdna165d76Q\"",
		"mtime": "2026-07-10T20:29:22.011Z",
		"size": 355,
		"path": "../public/wasm/package.json"
	},
	"/assets/index-BF_zNKC2.js": {
		"type": "text/javascript; charset=utf-8",
		"etag": "\"87168-0ByvUYLJ7lOf54MOeGgwO2Fjypw\"",
		"mtime": "2026-07-31T13:22:22.296Z",
		"size": 553320,
		"path": "../public/assets/index-BF_zNKC2.js"
	}
};
//#endregion
//#region #nitro/virtual/public-assets
var publicAssetBases = {};
function isPublicAssetURL(id = "") {
	if (public_assets_data_default[id]) return true;
	for (const base in publicAssetBases) if (id.startsWith(base)) return true;
	return false;
}
//#endregion
//#region node_modules/nitro/dist/runtime/internal/route-rules.mjs
var headers = ((m) => function headersRouteRule(event) {
	for (const [key, value] of Object.entries(m.options || {})) event.res.headers.set(key, value);
});
//#endregion
//#region #nitro/virtual/routing
var findRouteRules = /* @__PURE__ */ (() => {
	const $0 = [{
		name: "headers",
		route: "/assets/**",
		handler: headers,
		options: { "cache-control": "public, max-age=31536000, immutable" }
	}];
	return (m, p) => {
		let r = [];
		if (p.charCodeAt(p.length - 1) === 47) p = p.slice(0, -1) || "/";
		let s = p.split("/");
		if (s.length > 1) {
			if (s[1] === "assets") r.unshift({
				data: $0,
				params: { "_": s.slice(2).join("/") }
			});
		}
		return r;
	};
})();
var _lazy_POgIi_ = defineLazyEventHandler(() => import("./_chunks/ssr-renderer.mjs"));
var findRoute = /* @__PURE__ */ (() => {
	const data = {
		route: "/**",
		handler: _lazy_POgIi_
	};
	return ((_m, p) => {
		return {
			data,
			params: { "_": p.slice(1) }
		};
	});
})();
[].filter(Boolean);
//#endregion
//#region node_modules/nitro/dist/runtime/internal/error/prod.mjs
var errorHandler = (error, event) => {
	const res = defaultHandler(error, event);
	return new FastResponse(typeof res.body === "string" ? res.body : JSON.stringify(res.body, null, 2), res);
};
function defaultHandler(error, event) {
	const unhandled = error.unhandled ?? !HTTPError.isError(error);
	const { status = 500, statusText = "" } = unhandled ? {} : error;
	if (status === 404) {
		const url = event.url || new URL(event.req.url);
		const baseURL = "/";
		if (/^\/[^/]/.test(baseURL) && !url.pathname.startsWith(baseURL)) return {
			status: 302,
			headers: new Headers({ location: `${baseURL}${url.pathname.slice(1)}${url.search}` })
		};
	}
	const headers = new Headers(unhandled ? {} : error.headers);
	headers.set("content-type", "application/json; charset=utf-8");
	return {
		status,
		statusText,
		headers,
		body: {
			error: true,
			...unhandled ? {
				status,
				unhandled: true
			} : typeof error.toJSON === "function" ? error.toJSON() : {
				status,
				statusText,
				message: error.message
			}
		}
	};
}
//#endregion
//#region #nitro/virtual/error-handler
var errorHandlers = [errorHandler];
async function error_handler_default(error, event) {
	for (const handler of errorHandlers) try {
		const response = await handler(error, event, { defaultHandler });
		if (response) return response;
	} catch (error) {
		console.error(error);
	}
}
//#endregion
//#region #nitro/virtual/app
function createNitroApp() {
	const captureError = (error, errorCtx) => {
		if (errorCtx?.event) {
			const errors = errorCtx.event.req.context?.nitro?.errors;
			if (errors) errors.push({
				error,
				context: errorCtx
			});
		}
	};
	const h3App = createH3App({ onError(error, event) {
		return error_handler_default(error, event);
	} });
	let appHandler = (req) => {
		req.context ||= {};
		req.context.nitro = req.context.nitro || { errors: [] };
		return h3App.fetch(req);
	};
	return {
		fetch: appHandler,
		h3: h3App,
		hooks: void 0,
		captureError
	};
}
function createH3App(config) {
	const h3App = new H3Core(config);
	h3App["~findRoute"] = (event) => findRoute(event.req.method, event.url.pathname);
	h3App["~getMiddleware"] = (event, route) => {
		const pathname = event.url.pathname;
		const method = event.req.method;
		const middleware = [];
		const routeRules = getRouteRules(method, pathname);
		event.context.routeRules = routeRules?.routeRules;
		if (routeRules?.routeRuleMiddleware.length) middleware.push(...routeRules.routeRuleMiddleware);
		if (route?.data?.middleware?.length) middleware.push(...route.data.middleware);
		return middleware;
	};
	return h3App;
}
//#endregion
//#region node_modules/nitro/dist/runtime/internal/app.mjs
var APP_ID = "default";
function useNitroApp() {
	let instance = useNitroApp._instance;
	if (instance) return instance;
	instance = useNitroApp._instance = createNitroApp();
	globalThis.__nitro__ = globalThis.__nitro__ || {};
	globalThis.__nitro__[APP_ID] = instance;
	return instance;
}
function useNitroHooks() {
	const nitroApp = useNitroApp();
	const hooks = nitroApp.hooks;
	if (hooks) return hooks;
	return nitroApp.hooks = new HookableCore();
}
function getRouteRules(method, pathname) {
	const m = findRouteRules(method, pathname);
	if (!m?.length) return { routeRuleMiddleware: [] };
	const routeRules = {};
	for (const layer of m) for (const rule of layer.data) {
		const currentRule = routeRules[rule.name];
		if (currentRule) {
			if (rule.options === false) {
				delete routeRules[rule.name];
				continue;
			}
			if (typeof currentRule.options === "object" && typeof rule.options === "object") currentRule.options = {
				...currentRule.options,
				...rule.options
			};
			else currentRule.options = rule.options;
			currentRule.route = rule.route;
			currentRule.params = {
				...currentRule.params,
				...layer.params
			};
		} else if (rule.options !== false) routeRules[rule.name] = {
			...rule,
			params: layer.params
		};
	}
	const middleware = [];
	const orderedRules = Object.values(routeRules).sort((a, b) => (a.handler?.order || 0) - (b.handler?.order || 0));
	for (const rule of orderedRules) {
		if (rule.options === false || !rule.handler) continue;
		middleware.push(rule.handler(rule));
	}
	return {
		routeRules,
		routeRuleMiddleware: middleware
	};
}
//#endregion
//#region node_modules/nitro/dist/presets/cloudflare/runtime/_module-handler.mjs
function createHandler(hooks) {
	const nitroApp = useNitroApp();
	const nitroHooks = useNitroHooks();
	return {
		async fetch(request, env, context) {
			globalThis.__env__ = env;
			augmentReq(request, {
				env,
				context
			});
			const ctxExt = {};
			const url = new URL(request.url);
			if (hooks.fetch) {
				const res = await hooks.fetch(request, env, context, url, ctxExt);
				if (res) return res;
			}
			return await nitroApp.fetch(request);
		},
		scheduled(controller, env, context) {
			globalThis.__env__ = env;
			context.waitUntil(nitroHooks.callHook("cloudflare:scheduled", {
				controller,
				env,
				context
			}) || Promise.resolve());
		},
		email(message, env, context) {
			globalThis.__env__ = env;
			context.waitUntil(nitroHooks.callHook("cloudflare:email", {
				message,
				event: message,
				env,
				context
			}) || Promise.resolve());
		},
		queue(batch, env, context) {
			globalThis.__env__ = env;
			context.waitUntil(nitroHooks.callHook("cloudflare:queue", {
				batch,
				event: batch,
				env,
				context
			}) || Promise.resolve());
		},
		tail(traces, env, context) {
			globalThis.__env__ = env;
			context.waitUntil(nitroHooks.callHook("cloudflare:tail", {
				traces,
				env,
				context
			}) || Promise.resolve());
		},
		trace(traces, env, context) {
			globalThis.__env__ = env;
			context.waitUntil(nitroHooks.callHook("cloudflare:trace", {
				traces,
				env,
				context
			}) || Promise.resolve());
		}
	};
}
function augmentReq(cfReq, ctx) {
	const req = cfReq;
	req.ip = cfReq.headers.get("cf-connecting-ip") || void 0;
	req.runtime ??= { name: "cloudflare" };
	req.runtime.cloudflare = {
		...req.runtime.cloudflare,
		...ctx
	};
	req.waitUntil = ctx.context?.waitUntil.bind(ctx.context);
}
//#endregion
//#region node_modules/nitro/dist/presets/cloudflare/runtime/cloudflare-module.mjs
var cloudflare_module_default = createHandler({ fetch(cfRequest, env, context, url) {
	if (env.ASSETS && isPublicAssetURL(url.pathname)) return env.ASSETS.fetch(cfRequest);
} });
//#endregion
export { cloudflare_module_default as default };
