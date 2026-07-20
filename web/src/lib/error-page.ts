/**
 * Render an HTML error page returned by the SSR pipeline when an
 * unhandled exception escapes a route loader. Self-contained — no
 * React, no JS framework, just a static page the browser can render.
 *
 * No inline event handlers (no `onclick="..."`) — those break under
 * strict Content-Security-Policy headers. Instead, attach the handler
 * after DOMContentLoaded via a small inline <script>.
 */
export function renderErrorPage(): string {
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
    </script>
  </body>
</html>`;
}
