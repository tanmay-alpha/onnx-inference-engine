# Crucible Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Crucible Platform                         │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  C++ Engine  │  WASM Module │ Python Server│   Web Dashboard   │
│  engine/     │   wasm/      │   server/    │      web/         │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ ONNX Parser  │ Pure-Rust    │ FastAPI      │ React 19 + TanStack│
│ Tensor Lib   │ reimplementation│ REST API  │ Router + Tailwind │
│ 20+ Ops      │ WASM bindgen │ PostgreSQL   │ Chart.js dashboards│
│ Eigen SIMD   │ Fraud model  │ Auth (JWT)   │ Realtime metrics   │
│ pybind11     │ (client-side)│ Batch infer  │                   │
│ CLI (Rust)   │              │ Prometheus   │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

## Data Flow

### Inference Flow (Server)
```
Client → POST /infer → Auth Middleware → Body Size Check →
  Timeout Wrapper → Model Lookup → C++ Engine → Log to DB → Response
```

### Inference Flow (WASM — Browser)
```
User Input → WASM Module (in-browser) → Local Execution →
  Result Display → Optional: POST /fraud/log (server)
```

### Fraud Detection Pipeline
```
Transaction Data → Input Validation → Feature Extraction →
  Model Inference (WASM or Server) → Risk Scoring →
  Database Log → Optional: Webhook Alert → UI Display
```

## Component Details

### C++ Engine (`engine/`)
- **Tensor**: Row-major float32/int64 storage with shape/strides
- **ONNX Parser**: Hand-written protobuf decoder (varint, fixed32, length-delimited)
- **Executor**: Kahn's algorithm topological sort + visitor pattern for ops
- **Ops**: Conv2D (im2col + Eigen GEMM), MatMul/Gemm, activations, pooling, norm
- **Bindings**: pybind11 for Python, C-API for Rust FFI

### WASM Module (`wasm/`)
- **Pure-Rust reimplementation** of ONNX runtime
- **fraud_model.rs**: 2-layer feedforward network (29→16→1) with hardcoded weights
- **ONNX Parser**: Rust implementation for parsing ONNX model files
- **Memory**: Explicit allocator with reference counting

### Python Server (`server/`)
- **FastAPI** with async support
- **Database**: SQLAlchemy 2.0 async ORM (SQLite dev, PostgreSQL prod)
- **Auth**: JWT bearer tokens + API keys (bcrypt hashing)
- **Endpoints**: 20+ REST endpoints covering inference, auth, analytics, batch
- **Metrics**: Prometheus client for monitoring

### Web Frontend (`web/`)
- **React 19** with TanStack Router (App Router)
- **Tailwind CSS 4** with custom design system
- **WASM Integration**: Dynamic loading with error boundary
- **Dashboard**: Recharts for analytics visualization

## Deployment Architecture

### Docker Compose
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ PostgreSQL│    │  Redis   │    │  Server  │    │    Web   │
│  :5432   │◄───│  :6379   │◄───│  :8000   │◄───│  :3000   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     ▲                ▲
                ┌──────────┐    ┌──────────┐
                │ Prometheus│    │ Grafana  │
                │  :9090   │    │  :3001   │
                └──────────┘    └──────────┘
```

### Vercel Deployment
- Frontend deployed as static export
- API routes proxy to Python backend
- WASM module bundled as static asset

## Security Model
- JWT authentication for user accounts
- API keys for programmatic access
- Rate limiting per user/API key
- Request body size limits
- Inference timeouts
- HMAC webhook signatures
- Non-root Docker execution
- SQL injection prevention (ORM only)
