# Crucible API Reference

Base URL: `http://localhost:8000`

## Authentication

All protected endpoints require either:
- `Authorization: Bearer <jwt_token>` header, OR
- `X-API-Key: <api_key>` header

Public endpoints: `/health`, `/operators`, `/models`, `/fraud/history`, `/benchmarks`, `/docs`, `/openapi.json`

## Endpoints

### Health & Info
- `GET /health` — Liveness probe (public)
- `GET /operators` — List supported ONNX ops (public)
- `GET /metrics` — Prometheus metrics (public)

### Authentication
- `POST /auth/register` — Create user account
  ```json
  {"email": "user@example.com", "password": "secure123", "full_name": "John Doe"}
  ```
- `POST /auth/login` — Get JWT token
  ```json
  {"email": "user@example.com", "password": "secure123"}
  ```
  Returns: `{"access_token": "...", "token_type": "bearer", "expires_in": 3600}`
- `GET /auth/me` — Get current user info (requires auth)
- `POST /auth/api-key` — Generate new API key (requires auth)
  ```json
  {"name": "Production Server", "expires_in_days": 90}
  ```
- `GET /auth/api-keys` — List my API keys (requires auth)
- `DELETE /auth/api-key/{key_id}` — Revoke API key (requires auth)

### Model Management
- `POST /convert` — Upload ONNX model (requires auth)
  - Form data: `model_file` (.onnx), `input_shape` (JSON array)
- `GET /models` — List all models (public)
- `GET /models/{model_id}` — Get model info (public)
- `DELETE /models/{model_id}` — Delete model (requires auth)

### Inference
- `POST /infer` — Run single inference (requires auth)
  ```json
  {"model_id": "abc123", "input": [0.1, 0.2, ...], "input_shape": [1, 3, 224, 224]}
  ```
- `POST /inference/batch` — Batch inference (requires auth)
  ```json
  {"requests": [{"model_id": "...", "input": [...], "input_shape": [...]}, ...]}
  ```

### Validation
- `POST /validate` — Validate ONNX model (requires auth)
  - Form data: `model_file` OR `model_id`

### Analytics (requires auth)
- `GET /analytics/inference?days=7` — Inference volume stats
- `GET /analytics/fraud?days=7` — Fraud detection stats
- `GET /analytics/models` — Model usage statistics

### Fraud Detection
- `POST /fraud/log` — Log fraud check (public)
  ```json
  {"tx_type": "TRANSFER", "amount": 50000, "orig_before": 60000, ...}
  ```
- `GET /fraud/history?limit=50` — Get fraud history (public)

### Benchmarks
- `POST /benchmarks` — Log benchmark (public)
- `GET /benchmarks?limit=50` — Get benchmarks (public)

## Response Format

### Success Response
```json
{
  "output": [0.1, 0.2, ...],
  "output_shape": [1, 10],
  "inference_time_ms": 12.5,
  "engine": "crucible-cpp"
}
```

### Error Response
```json
{
  "detail": "Error description",
  "error_code": "MODEL_NOT_FOUND",
  "trace_id": "abc123def456"
}
```

## Status Codes
- `200` — Success
- `201` — Created
- `400` — Bad request (invalid input)
- `401` — Unauthorized (missing/invalid auth)
- `403` — Forbidden (insufficient permissions)
- `404` — Not found
- `413` — Payload too large
- `429` — Rate limited
- `500` — Internal server error
- `503` — Service not configured
- `504` — Gateway timeout
