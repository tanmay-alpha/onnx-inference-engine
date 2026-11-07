# Crucible API Reference

Base URL: `http://localhost:8000` (local) — interactive schema at `/docs` and `/openapi.json`.

## Authentication

Every endpoint except the public ones below requires **one** of:

| Header | Who |
| --- | --- |
| `Authorization: Bearer <jwt>` | a user; token from `POST /auth/login` (expires after `CRUCIBLE_TOKEN_EXPIRE_MINUTES`, default 60) |
| `X-API-Key: <key>` | a user; key from `POST /auth/api-key` (stored as SHA-256; revoked/expired keys are rejected) |
| `X-API-Key: <CRUCIBLE_API_KEY>` | the optional operator key from the server environment; sees every user's data; cannot manage user API keys |

Missing or invalid credentials → `401`. If `CRUCIBLE_API_KEY` is unset, operator access is simply disabled.

**Public:** `GET /health`, `GET /ready`, `GET /operators`, `GET /metrics`, `POST /auth/register`, `POST /auth/login`, `/docs`, `/openapi.json`.

**Ownership:** models, inference logs and fraud-check history belong to the authenticated user. Other users get `404` for them (list endpoints only return your own rows).

## Public endpoints

### `GET /health`
Liveness. Always `200` while the process serves requests.
```json
{"status": "ok", "engine": "crucible-native", "native_available": true, "version": "1.0.0"}
```
`engine` is `"unavailable"` (and `native_available` false) when the native module `crucible_py` is not loaded — inference then returns `503`.

### `GET /ready`
`200` only when the database answers and the native engine is loaded, otherwise `503`. Same body as `/health` plus `"database": true|false` and `status` `"ready"`/`"not_ready"`.

### `GET /operators`
```json
{"supported": ["Add", "AveragePool", "BatchNormalization", "..."], "count": 20}
```

### `GET /metrics`
Prometheus text format: `crucible_inference_requests_total{engine,status}`, `crucible_inference_duration_seconds{engine}` (engine-only time), `crucible_errors_total{error_type}`, `crucible_batch_jobs_total{status}`.

## Accounts and keys

- `POST /auth/register` — `{"email": "...", "password": "<8-128 chars>", "full_name": "optional"}` → `201` user; `409` if the email exists (case-insensitive).
- `POST /auth/login` — `{"email": "...", "password": "..."}` → `{"access_token": "...", "token_type": "bearer", "expires_in": 3600}`; `401` on bad credentials (unknown emails take the same bcrypt check as wrong passwords).
- `GET /auth/me` → `{"id", "email", "full_name", "is_active", "is_admin", "created_at"}`.
- `POST /auth/api-key` — `{"name": "ci", "expires_in_days": 90 | null}` → key info plus `full_key` (shown once).
- `GET /auth/api-keys` → `[{"id", "name", "key_prefix", "is_active", "rate_limit", "last_used", "created_at", "expires_at"}]`. `rate_limit` (default 100) is the key's budget in inference requests per minute on `/infer` and `/inference/batch` (each batch item counts); beyond it → `429` + `Retry-After`. JWT callers have no per-minute limit.
- `DELETE /auth/api-key/{key_id}` → `{"status": "revoked", "key_id": "..."}`.

The `/auth/me` and API-key routes need a user (the operator key gets `403`).

### Rate limits
`POST /auth/login` and `POST /auth/register` together allow `AUTH_RATE_LIMIT_PER_IP` (default 30) attempts per client IP per `AUTH_RATE_WINDOW_SEC` (default 300 s, sliding); logins additionally `AUTH_RATE_LIMIT_PER_IP_EMAIL` (default 10) per IP+email. Beyond → `429` with `Retry-After` (seconds). The client IP is the TCP peer, or the first `X-Forwarded-For` entry when `TRUST_PROXY_HEADERS=true` (enable only behind a proxy that sets that entry itself, such as Render's; otherwise clients can spoof it).

All limits are kept **in process memory**: they reset on restart and are not shared between instances or workers (each enforces its own).

## Models

### `POST /convert`
Multipart: `model_file` (ONNX file, ≤ `MAX_UPLOAD_BYTES`, default 50 MB — parsing peaks at roughly twice the file size). `input_shape` is accepted for backward compatibility and ignored — signatures are read from the model.

The file is saved as `<uuid>.onnx`, checked with `onnx.checker` and, when available, the native loader (`crucible_py.load_model` + `check_model` + `get_model_info`). Models Crucible cannot run are **still stored** and flagged.
```json
{
  "onnx_model_id": "b7cb...", "name": "fraud_detector.onnx", "size_bytes": 195, "opset": 13,
  "operators_used": ["MatMul", "Add", "Sigmoid"], "all_supported": true,
  "unsupported_ops": [], "unsupported_reasons": [],
  "inputs":  [{"name": "input", "shape": [-1, 7], "dtype": "float32"}],
  "outputs": [{"name": "prob",  "shape": [-1, 1], "dtype": "float32"}]
}
```
`shape` uses `-1` for dynamic dims and `null` for unknown rank.

| Status | When |
| --- | --- |
| `400` | malformed ONNX; graph with more than `MAX_GRAPH_NODES` (default 2000) nodes (not stored) |
| `409` | you already store `MAX_MODELS_PER_USER` (default 20) models |
| `413` | file > `MAX_UPLOAD_BYTES`, or it would push your stored bytes past `MAX_STORAGE_BYTES_PER_USER` (default 200 MB) |
| `503` | `MAX_CONCURRENT_ANALYSES` (default 1) analyses already running (`Retry-After: 2`) |

Quotas don't apply to the operator key.

### `POST /validate`
Multipart with exactly one of `model_file` (analysed, not stored) or `model_id` (one of yours). Returns
`{"valid", "operators", "unsupported", "unsupported_reasons", "opset", "inputs", "outputs"}`.
A graph over `MAX_GRAPH_NODES` nodes comes back `valid: false` with the node count in `unsupported_reasons`. Malformed ONNX → `400`; busy → `503` as for `/convert`.

### `GET /models`
`{"models": [ModelItem], "count": n}` — your models only.

### `GET /models/{id}` / `DELETE /models/{id}`
`ModelItem` = `{"id", "name", "size_bytes", "created_at", "opset", "operators_used", "all_supported", "unsupported_ops", "unsupported_reasons", "inputs", "outputs"}`.
`DELETE` removes the row and the file (only if it lies inside the model directory) → `{"status": "deleted", "model_id": "..."}`.

## Inference

Runs **only** on the native C++ engine; there is no fallback.

### `POST /infer`
```json
{"model_id": "b7cb...", "input": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "input_shape": [1, 7]}
```
`input` is the flattened float32 tensor in row-major order; its length must equal the product of `input_shape`.
```json
{"model_id": "b7cb...", "output": [0.93], "output_shape": [1, 1],
 "inference_time_ms": 0.16, "backend": "crucible-native", "engine": "crucible-native"}
```
`output` is the model's first output. `inference_time_ms` times `crucible_py.run` only (no HTTP, JSON or model loading). Loaded models are cached (LRU, `MODEL_CACHE_SIZE`).

Each run is bounded by the engine: no single tensor above `MAX_TENSOR_ELEMENTS` (default 16M), live intermediates together ≤ `MAX_TOTAL_ELEMENTS` (default 64M = 256 MB float32), and a deadline of `INFERENCE_TIMEOUT_SEC`.

| Status | When |
| --- | --- |
| `404` | model unknown or not yours |
| `410` | model file no longer on disk (ephemeral hosting) — upload again |
| `413` | more than `MAX_INPUT_ELEMENTS` (default 4,000,000) input elements, output larger than `MAX_OUTPUT_ELEMENTS` (default 1,000,000), or body > `MAX_REQUEST_BODY_BYTES` |
| `422` | value count ≠ product(`input_shape`); shape doesn't match the model's static dims; NaN/Inf input; model not supported; multi-input or non-float32 model; engine rejected the input; a tensor or the run's intermediates exceed the element limits; NaN/Inf output |
| `429` | API key over its `rate_limit` per minute (`Retry-After`) |
| `503` | native engine unavailable, or all `MAX_CONCURRENT_INFERENCES` slots busy (`Retry-After: 1`) |
| `504` | run exceeded `INFERENCE_TIMEOUT_SEC`. The engine checks the deadline between nodes and inside Conv/pooling/MatMul outer loops, so the run stops shortly after it (best effort: one node or outer-loop step) and frees its slot |

### `POST /inference/batch`
Body: a JSON **array** of `/infer` requests (≤ `MAX_BATCH_SIZE`, default 32; total input elements ≤ `MAX_INPUT_ELEMENTS`). Items run sequentially.
```json
{"batch_size": 2, "success_count": 1, "error_count": 1, "results": [
  {"model_id": "...", "status": "success", "output": [...], "output_shape": [...], "inference_time_ms": 0.2, "backend": "crucible-native"},
  {"model_id": "...", "status": "error", "error": "Model ... not found", "status_code": 404, "backend": "crucible-native"}
]}
```
Whole-batch errors: `503` when the native engine is unavailable; `413` when the outputs together exceed `MAX_OUTPUT_ELEMENTS` (processing stops there); `429` when an API key lacks budget for every item.

### `GET /inference/logs?limit=50`
`{"logs": [{"id", "model_id", "latency_ms", "backend", "input_shape", "output_shape", "status", "created_at"}], "count": n}` — your runs; `status` is `success`, `error` or `timeout`.

## Fraud-check history

Scores are computed client-side (the browser runs the fraud model in WASM); the server only records them.

- `POST /fraud/log` — `{"tx_type", "amount", "orig_before", "orig_after", "dest_before", "dest_after", "probability" (0–1), "verdict", "execution_mode", "latency_ms"}` → the stored item plus `id`, `created_at`.
- `GET /fraud/history?limit=50` → `{"history": [...], "count": n}` — yours only.

## Analytics (your data only)

- `GET /analytics/inference?days=7` → `{"period_days", "data": [{"date", "count", "avg_latency_ms", "min_latency_ms", "max_latency_ms"}]}` (successful runs).
- `GET /analytics/fraud?days=7` → `{"period_days", "data": [{"date", "total", "fraud_count", "avg_probability"}]}` (`fraud_count` = verdicts containing "high").
- `GET /analytics/models` → `{"models": [{"id", "name", "usage_count", "inference_count", "avg_latency_ms", "last_used"}]}`.

## Benchmarks

There is no benchmark API. Published numbers live in the static file `web/public/benchmarks/results.json`, produced by `python benchmarks/run_all.py`.

## Errors

Errors use FastAPI's envelope: `{"detail": "..."}` (validation errors: `{"detail": [ ... ]}` with `422`). Unexpected failures return `500` with `"Internal error (trace <id>)."`; the traceback is logged server-side under that id.
