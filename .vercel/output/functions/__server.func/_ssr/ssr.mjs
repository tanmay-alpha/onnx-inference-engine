//#region node_modules/.nitro/vite/services/ssr/index.js
var lastCapturedError;
var TTL_MS = 5e3;
function record(error) {
	lastCapturedError = {
		error,
		at: Date.now()
	};
}
if (typeof globalThis.addEventListener === "function") {
	globalThis.addEventListener("error", (event) => record(event.error ?? event));
	globalThis.addEventListener("unhandledrejection", (event) => record(event.reason));
}
function consumeLastCapturedError() {
	if (!lastCapturedError) return void 0;
	if (Date.now() - lastCapturedError.at > TTL_MS) {
		lastCapturedError = void 0;
		return;
	}
	const { error } = lastCapturedError;
	lastCapturedError = void 0;
	return error;
}
/**
* Render an HTML error page returned by the SSR pipeline when an
* unhandled exception escapes a route loader. Self-contained — no
* React, no JS framework, just a static page the browser can render.
*
* No inline event handlers (no `onclick="..."`) — those break under
* strict Content-Security-Policy headers. Instead, attach the handler
* after DOMContentLoaded via a small inline <script>.
*/
function renderErrorPage() {
	return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>This page didn't load</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font: 15px/1.5 system-ui, -apple-system, sans-serif; background: #fafafa; color: #111; display: grid; place-items: center; min-height: 100vh; margin: 0; padding: 1.5rem; }
      .card { max-width: 28rem; width: 100%; text-align: center; padding: 2rem; }
      h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
      p { color: #4b5563; margin: 0 0 1.5rem; }
      .actions { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
      a, button { padding: 0.5rem 1rem; border-radius: 0.375rem; font: inherit; cursor: pointer; text-decoration: none; border: 1px solid transparent; }
      .primary { background: #111; color: #fff; }
      .secondary { background: #fff; color: #111; border-color: #d1d5db; }
      .muted { color: #6b7280; font-size: 0.8rem; margin-top: 1rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>This page didn't load</h1>
      <p>Something went wrong on our end. You can try refreshing or head back home.</p>
      <div class="actions">
        <button id="retry-btn" class="primary" type="button">Try again</button>
        <a class="secondary" href="/">Go home</a>
      </div>
      <p id="retry-note" class="muted" style="display:none"></p>
    </div>
    <script>
      // Attach the click handler after DOMContentLoaded so that strict
      // CSP (script-src 'self') doesn't break this page.
      // Limit retries to prevent an infinite reload loop when the server
      // is in a permanently broken state.
      (function () {
        var MAX_RETRIES = 3;
        var key = '__crucible_retry_count';
        var count = parseInt(sessionStorage.getItem(key) || '0', 10);
        var btn = document.getElementById('retry-btn');
        var note = document.getElementById('retry-note');
        if (count >= MAX_RETRIES) {
          if (btn) btn.disabled = true;
          if (btn) btn.textContent = 'Retries exhausted';
          if (note) { note.style.display = 'block'; note.textContent = 'The page failed to load after ' + MAX_RETRIES + ' attempts. Please try again later or go home.'; }
        } else {
          if (btn) btn.addEventListener('click', function () {
            sessionStorage.setItem(key, String(count + 1));
            location.reload();
          });
        }
      })();
    <\/script>
  </body>
</html>`;
}
var serverEntryPromise;
async function getServerEntry() {
	if (!serverEntryPromise) {
		serverEntryPromise = new Promise((resolve, reject) => {
			import("./server-CI0yXFus.mjs").then((m) => resolve(m.default ?? m)).catch(reject);
		});
		serverEntryPromise.catch((err) => {
			console.error("Server entry import failed, will retry on next request:", err);
			serverEntryPromise = void 0;
		});
	}
	return serverEntryPromise;
}
async function normalizeCatastrophicSsrResponse(response) {
	if (response.status < 500) return response;
	if (!(response.headers.get("content-type") ?? "").includes("application/json")) return response;
	const body = await response.clone().text();
	if (!isH3SwallowedErrorBody(body)) return response;
	console.error(consumeLastCapturedError() ?? /* @__PURE__ */ new Error(`h3 swallowed SSR error: ${body}`));
	return new Response(renderErrorPage(), {
		status: 500,
		headers: { "content-type": "text/html; charset=utf-8" }
	});
}
function isH3SwallowedErrorBody(body) {
	try {
		const payload = JSON.parse(body);
		return payload.unhandled === true && payload.message === "HTTPError";
	} catch {
		return false;
	}
}
var server_default = { async fetch(request, env, ctx) {
	try {
		return await normalizeCatastrophicSsrResponse(await (await getServerEntry()).fetch(request, env, ctx));
	} catch (error) {
		console.error(error);
		return new Response(renderErrorPage(), {
			status: 500,
			headers: { "content-type": "text/html; charset=utf-8" }
		});
	}
} };
//#endregion
export { server_default as default, renderErrorPage as t };
