# Crucible Web Frontend

React 19 + TanStack Start (Vite 8, Tailwind CSS v4) frontend for Crucible.

## What runs where

| Page                                 | Needs the API?                           | What it does                                                                                                                                                                                                                      |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/`                                  | no                                       | Loads `public/models/fraud_detector.onnx` and scores three sample transactions live with the WASM runtime.                                                                                                                        |
| `/fraud`                             | no (saving history needs login)          | Scores a transaction in the browser with the WASM runtime. Saving to history is opt-in and only offered when logged in.                                                                                                           |
| `/playground`                        | WASM mode: no · server mode: yes + login | Parses any `.onnx` in the browser (graph, inputs, opset), checks WASM compatibility, runs it with the WASM runtime and/or the native engine via `POST /infer`. Errors from either backend are shown; there is no silent fallback. |
| `/benchmark`                         | no                                       | Renders `public/benchmarks/results.json` (written by `python benchmarks/run_all.py`). Without the file it shows an empty state.                                                                                                   |
| `/models`, `/dashboard`, `/api-keys` | yes + login                              | Model registry, analytics, API keys. Client-side routes guarded by `requireAuth`.                                                                                                                                                 |

The WASM runtime (`public/wasm/`, built from `../wasm` with wasm-pack) is a
separate pure-Rust interpreter implementing MatMul, Add, Relu, Sigmoid and
Softmax. The native C++ engine (20 operators) runs only on the server.

When the API is unreachable, pages that need it show a "Backend offline"
notice (from `GET /health`); WASM features keep working.

## Configuration

`VITE_API_URL` — base URL of the Crucible FastAPI server, inlined at build time
(default `http://localhost:8000`). See `.env.example`.

Protected requests send `Authorization: Bearer <JWT>` from `POST /auth/login`.

## Development

```bash
npm ci                 # install exact lockfile versions
npm run dev            # Vite dev server on :5173
npm run type-check     # tsc
npm run lint           # eslint (+ prettier)
npm run build          # production build
```

## Deployment (Vercel)

Nitro selects its `vercel` preset automatically on Vercel (or set
`NITRO_PRESET=vercel` locally), writing the Build Output API directory to
`web/.vercel/output`. `scripts/postbuild.js` (run by `npm run build`) mirrors
it to `<repo>/.vercel/output`, which is where Vercel looks when the project
builds from the repository root (`cd web && npm run build`). Set
`VITE_API_URL` in the Vercel project's environment variables.
