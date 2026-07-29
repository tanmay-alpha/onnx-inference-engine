# onnx-inference-engine

A high-performance ONNX model inference engine with a FastAPI backend, TanStack Start frontend, and a C++17 execution core. Supports fraud detection, image classification (MobileNetV2, ResNet18), and custom ONNX models. Deploys on Render + Vercel + Supabase.

## What It Is

onnx-inference-engine (internally codenamed **Crucible**) provides a complete pipeline for training, converting, validating, and serving ONNX models at production scale. A custom C++ inference engine delivers low-latency execution with a Python fallback for environments where the native bindings are unavailable. The web frontend exposes an interactive playground, analytics dashboard, fraud detection history, and architecture visualizations. A pure-Rust WASM reimplementation enables zero-latency, privacy-preserving inference directly in the browser.

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, SQLAlchemy 2.0 (async), Alembic, Prometheus Client, structlog |
| **Frontend** | TanStack Start, React, TanStack Router, Tailwind CSS, shadcn/ui, Chart.js |
| **Database** | Supabase (PostgreSQL) with asyncpg; SQLite for local development |
| **ML Training** | scikit-learn, XGBoost, PyTorch (model export), imbalanced-learn (SMOTE) |
| **Inference Engine** | C++17 core with Eigen, protobuf ONNX parser, pybind11 Python bindings |
| **WASM Runtime** | Pure Rust WASM reimplementation for browser-side inference |
| **CLI** | Rust + Clap (FFI bridge to C++ engine) |
| **Deployment** | Render (API), Vercel (frontend), Supabase (DB + Storage) |

## Features

- Upload and serve ONNX models via REST API with multipart upload
- Real-time inference through a custom C++ execution engine (numpy fallback available)
- Fraud detection model trained on the Kaggle CreditCard dataset (GradientBoosting / XGBoost)
- Image classification support (MobileNetV2, ResNet18)
- JWT-based authentication with user registration, login, and API key management
- Analytics dashboard with inference history, fraud transaction logs, and benchmark results
- Model management: upload, list, validate, and delete ONNX models
- Prometheus metrics endpoint (`GET /metrics`)
- Supabase Storage integration for model file persistence
- Batch inference (up to 100 requests per call)
- Structured logging with structlog
- Comprehensive input validation and size limits to prevent abuse

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (npm or pnpm)
- Supabase account (for production database)
- CMake 3.25+ and Ninja (optional, for C++ engine build)
- Rust 1.78+ and wasm-pack (optional, for WASM engine)

### Setup

```bash
# 1. Clone the repository (include submodules for C++ deps)
git clone --recursive https://github.com/tanmay-alpha/onnx-inference-engine.git
cd onnx-inference-engine

# If cloned without --recursive, fetch submodules:
git submodule update --init --recursive

# 2. Copy environment template and configure
cp server/.env.example server/.env
# Edit server/.env with your Supabase credentials and settings

# 3. Install Python dependencies
pip install -r server/requirements.txt

# 4. Install frontend dependencies
cd web && npm install
cd ..

# 5. Apply database migrations
cd server && alembic upgrade head && cd ..

# 6. (Optional) Seed the database with demo data
cd server && python seed_db.py && cd ..

# 7. Start the backend server
cd server && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 8. In a new terminal, start the frontend dev server
cd web && npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Enable the `pgcrypto` extension in the SQL Editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS pgcrypto;
   ```
3. Run the Alembic migrations against your Supabase database (set `DATABASE_URL` in `.env` to your Supabase connection string):
   ```bash
   cd server && alembic upgrade head
   ```
4. Create a Storage bucket named `models` in the Supabase dashboard (optional, for Supabase Storage integration)
5. Set the following environment variables:
   - `DATABASE_URL` — Supabase PostgreSQL connection string
   - `SUPABASE_URL` — Your Supabase project URL (e.g. `https://xxx.supabase.co`)
   - `SUPABASE_SERVICE_ROLE_KEY` — Supabase service role key
   - `CRUCIBLE_SECRET_KEY` — A random 32+ character string for JWT signing
   - `CRUCIBLE_API_KEY` — Master API key for protected endpoints

## API Endpoints

All endpoints are prefixed with the server base URL (default `http://localhost:8000`).

### Health & Info
| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | Liveness probe; returns engine name and version |
| `GET` | `/operators` | No | Catalogue of supported ONNX operator types |
| `GET` | `/metrics` | No | Prometheus metrics exposition |

### Authentication
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | No | Register a new user account |
| `POST` | `/auth/login` | No | Authenticate and receive a JWT bearer token |
| `GET` | `/auth/me` | JWT | Get current authenticated user info |
| `POST` | `/auth/api-key` | JWT | Generate a new API key |
| `GET` | `/auth/api-keys` | JWT | List all API keys for the current user |
| `DELETE` | `/auth/api-key/{key_id}` | JWT | Revoke an API key |

### Model Management
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/convert` | API Key | Upload and validate an ONNX model file |
| `GET` | `/models` | No | List all registered models |
| `GET` | `/models/{model_id}` | No | Get metadata for a specific model |
| `DELETE` | `/models/{model_id}` | API Key | Delete a registered model |
| `POST` | `/validate` | API Key | Validate operator support for an ONNX model (file or model_id) |

### Inference
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/infer` | API Key | Run single-model inference |
| `POST` | `/inference/batch` | JWT | Run batch inference (up to 100 requests) |
| `GET` | `/inference/logs` | No | Get recent inference execution logs |

### Fraud Detection
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/fraud/log` | No | Log a fraud detection evaluation |
| `GET` | `/fraud/history` | No | Retrieve fraud transaction history |

### Benchmarks
| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/benchmarks` | No | Log a benchmark result |
| `GET` | `/benchmarks` | No | Retrieve recorded benchmarks |

### Analytics
| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/analytics/inference` | JWT | Inference volume and latency trends (1–90 days) |
| `GET` | `/analytics/fraud` | JWT | Fraud detection statistics (1–90 days) |
| `GET` | `/analytics/models` | JWT | Per-model usage and latency statistics |

## Frontend Routes

| Route | Description |
|---|---|
| `/` | Home page with live inference demo and recent fraud alerts |
| `/architecture` | Interactive architecture overview with animated computation graph |
| `/docs` | API documentation and usage guide |
| `/roadmap` | Project roadmap with shipped, in-progress, and planned features |
| `/story` | Project story and background |
| `/fraud` | Fraud detection history and real-time alerts |
| `/benchmark` | Benchmark comparison (Crucible vs ONNX Runtime vs PyTorch) |
| `/playground` | ONNX model playground with WASM-powered in-browser inference |

## Project Structure

```
onnx-inference-engine/
├── server/                          # FastAPI backend
│   ├── main.py                      # Application entry point, all route handlers
│   ├── auth.py                      # JWT + API key authentication
│   ├── database.py                  # SQLAlchemy ORM models and CRUD operations
│   ├── config.py                    # Pydantic Settings with all env vars
│   ├── schemas.py                   # Pydantic v2 request/response models
│   ├── validator.py                 # ONNX validation and operator extraction
│   ├── converter.py                 # ONNX upload acceptance and validation
│   ├── metrics.py                   # Prometheus metrics collection
│   ├── logging_config.py            # Structlog configuration
│   ├── webhooks.py                  # Webhook notification system
│   ├── seed_db.py                   # Database seeder for demo data
│   ├── requirements.txt             # Pinned Python dependencies
│   ├── alembic.ini                  # Alembic configuration
│   ├── .env.example                 # Environment variable template
│   ├── Dockerfile                   # Production Docker image
│   ├── alembic/                     # Database migration scripts
│   │   └── versions/
│   ├── ml/                          # ML training utilities and pipeline
│   └── tests/                       # Server unit and integration tests
├── engine/                          # C++17 inference engine
│   ├── CMakeLists.txt               # Top-level CMake build configuration
│   ├── CMakePresets.json            # Debug & Release presets
│   ├── include/crucible/            # Public C++ headers
│   │   ├── tensor.hpp               # Row-major float32 Tensor class
│   │   ├── onnx_parser.hpp          # Protobuf ONNX compute graph decoder
│   │   ├── executor.hpp             # Kahn's algorithm DAG executor
│   │   ├── c_api.h                  # C-FFI export boundary (for Rust CLI / WASM)
│   │   └── ops/                     # Operator kernels (Conv2D, MatMul, Gemm, ReLU, etc.)
│   ├── src/                         # C++ source code
│   │   ├── tensor.cpp               # Core tensor implementation
│   │   ├── onnx_parser.cpp          # ONNX protobuf decoder
│   │   ├── executor.cpp             # DAG executor
│   │   ├── c_api.cpp                # C ABI for FFI consumers
│   │   └── ops/                     # Operator implementations
│   │       ├── linear.cpp           # MatMul, Gemm
│   │       ├── activations.cpp      # Relu, Sigmoid, etc.
│   │       ├── conv2d.cpp           # Convolution
│   │       ├── pooling.cpp          # MaxPool, AvgPool
│   │       └── norm.cpp             # BatchNormalization
│   ├── bindings/python/             # pybind11 Python bindings (crucible_py)
│   ├── tests/                       # C++ unit tests (GoogleTest)
│   │   └── fixtures/                # ONNX test model fixtures
│   └── third_party/                 # Git submodule dependencies
│       ├── eigen/                   # Eigen linear algebra
│       ├── protobuf/                # Protocol Buffers (ONNX wire format)
│       ├── googletest/              # GoogleTest framework
│       ├── google-benchmark/        # Google Benchmark
│       └── pybind11/                # pybind11 bindings library
├── cli/                             # Rust CLI binary (FFI bridge to C++ engine)
│   ├── Cargo.toml
│   └── src/                         # Clap CLI commands (run, bench, info)
├── wasm/                            # Pure-Rust WebAssembly module
│   ├── Cargo.toml
│   └── src/lib.rs                   # In-browser WASM inference engine
├── models/                          # ML model assets and training code
│   ├── mobilenet_v2.onnx            # Pre-trained MobileNetV2
│   ├── resnet18.onnx                # Pre-trained ResNet18
│   ├── fraud/
│   │   ├── fraud_detector.onnx      # Production fraud detection model
│   │   ├── config.yaml              # Training configuration
│   │   ├── train_fraud_model.py     # Training script
│   │   ├── evaluate.py              # Model evaluation
│   │   ├── features.py              # Feature engineering
│   │   └── experiments/             # Experiment tracking
│   ├── image_classifier/
│   │   ├── config.yaml
│   │   ├── train.py                 # Image classifier training
│   │   └── evaluate.py
│   ├── data/                        # Raw and processed datasets
│   ├── registry/                    # Production model registry metadata
│   └── scripts/                     # Retraining and model comparison scripts
├── web/                             # TanStack Start frontend
│   ├── src/
│   │   ├── routes/                  # TanStack Router file-based routes
│   │   │   ├── __root.tsx           # Root layout
│   │   │   ├── index.tsx            # Home page
│   │   │   ├── fraud.tsx            # Fraud detection page
│   │   │   ├── benchmark.tsx        # Benchmark comparison page
│   │   │   ├── playground.tsx       # ONNX model playground
│   │   │   ├── docs.tsx             # Documentation page
│   │   │   ├── architecture.tsx     # Architecture visualization
│   │   │   ├── roadmap.tsx          # Roadmap page
│   │   │   └── story.tsx            # Project story
│   │   ├── components/              # Reusable UI components
│   │   │   ├── crucible/            # Layout and shared components
│   │   │   └── ui/                  # shadcn/ui primitives
│   │   ├── lib/                     # API client, utilities, WASM loader
│   │   │   ├── crucible-wasm.ts     # WASM inference bindings
│   │   │   └── utils.ts             # Helper functions
│   │   ├── data/                    # Static typed data
│   │   │   ├── graph.ts             # Computation graph for visualization
│   │   │   └── roadmap.ts           # Roadmap data
│   │   └── styles/                  # Global styles
│   ├── public/
│   │   ├── wasm/                    # Compiled WASM artifacts
│   │   └── models/                  # Bundled ONNX model files
│   ├── package.json                 # Frontend dependencies and scripts
│   ├── tsconfig.json                # TypeScript configuration
│   └── vite.config.ts               # Vite build configuration
├── benchmarks/                      # Benchmark result storage
├── scripts/                         # Helper build scripts (build-wasm.sh)
├── docker-compose.yml               # Full-stack Docker Compose setup
├── Makefile                         # Build automation and dev shortcuts
├── render.yaml                      # Render deployment configuration
├── Dockerfile                       # Multi-stage container build
├── .env.example                     # (in server/) Environment variable template
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Environment Variables

All server-side configuration is managed through environment variables. See `server/.env.example` for the full template.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./crucible.db` | Async SQLAlchemy database URL (Supabase Postgres in production) |
| `ENVIRONMENT` | No | `development` | App environment: `development`, `staging`, `production` |
| `APP_NAME` | No | `Crucible AI Inference Engine` | Application display name |
| `APP_VERSION` | No | `1.0.0` | Application version |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8000` | Server port |
| `DEBUG` | No | `False` | Enable debug mode |
| `CRUCIBLE_SECRET_KEY` | Yes | `dev-secret-key-...` | JWT signing secret (32+ chars in production) |
| `CRUCIBLE_API_KEY` | Yes | `crucible-development-api-key` | Master API key for `X-API-Key` header |
| `CRUCIBLE_CORS_ORIGINS` | No | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed CORS origins |
| `CRUCIBLE_MODEL_DIR` | No | `/tmp/models` | Directory for ONNX model file storage |
| `CRUCIBLE_LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CRUCIBLE_LOG_FORMAT` | No | `console` | Log format: `console` or `json` |
| `CRUCIBLE_TOKEN_EXPIRE_MINUTES` | No | `60` | JWT token validity duration |
| `INFERENCE_TIMEOUT_SEC` | No | `60` | Hard timeout for model inference execution |
| `MAX_UPLOAD_BYTES` | No | `209715200` (200 MB) | Maximum ONNX model file upload size |
| `MAX_REQUEST_BODY_BYTES` | No | `10485760` (10 MB) | Maximum JSON request body size |
| `MAX_INPUT_ELEMENTS` | No | `50000000` (50M) | Maximum product of input tensor dimensions |
| `SUPABASE_URL` | Production | — | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Production | — | Supabase service-role key (bypasses RLS) |
| `SUPABASE_STORAGE_BUCKET` | No | `models` | Supabase Storage bucket name for model files |
| `WEBHOOK_TIMEOUT_SEC` | No | `10` | Webhook HTTP request timeout |
| `WEBHOOK_MAX_RETRIES` | No | `3` | Maximum webhook delivery retries |
| `FRAUD_HIGH_RISK_THRESHOLD` | No | `0.7` | High-risk fraud probability threshold |
| `FRAUD_CRITICAL_RISK_THRESHOLD` | No | `0.9` | Critical-risk fraud probability threshold |

### Security Notes

- Never commit `.env` to version control.
- In production, `CRUCIBLE_SECRET_KEY` must be at least 32 random characters.
- The server performs fail-fast validation on startup and will refuse to start if production defaults are detected.

## Building the C++ Engine

The C++ inference engine (`engine/`) is optional. The Python server falls back to a numpy-based inference stub if the native bindings are unavailable. To build the full engine:

```bash
# Initialize git submodules (Eigen, protobuf, googletest, google-benchmark, pybind11)
git submodule update --init --recursive

# Configure and build in Release mode
cd engine
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON \
         -DCRUCIBLE_ENABLE_TESTS=ON
cmake --build . -j$(nproc)

# Run C++ tests
ctest --output-on-failure
```

The Python bindings (`crucible_py`) will be built into `engine/build/`. Add the build directory to your `PYTHONPATH` before starting the server:

```bash
export PYTHONPATH="/path/to/engine/build:$PYTHONPATH"
```

## Building the WASM Engine

A pure-Rust WASM reimplementation of the inference engine is available for browser-side execution:

```bash
cd wasm
wasm-pack build --target web --out-dir ../web/public/wasm
```

The compiled artifact (`crucible_wasm_bg.wasm`) is loaded by the frontend at `web/src/lib/crucible-wasm.ts`.

## Building the Rust CLI

```bash
cd cli
cargo build --release
```

The CLI communicates with the C++ engine via the C ABI exported in `engine/src/c_api.cpp`.

## Deployment

### Render (Backend API)

1. Connect your GitHub repository to Render.
2. Render will detect `render.yaml` and set up two services automatically:
   - **crucible-api**: Docker-based FastAPI server
   - **crucible-web**: Static site from Vite/TanStack Start build output
3. Set required environment variables (`DATABASE_URL`, `CRUCIBLE_SECRET_KEY`, `CRUCIBLE_API_KEY`, `CRUCIBLE_CORS_ORIGINS`).
4. The API service will be reachable at the Render-assigned domain.

### Vercel (Frontend)

1. Import the `web/` directory as a new Vercel project.
2. Set `VITE_API_URL` to your Render API service URL.
3. Set `NODE_VERSION` to `20`.
4. Deploy. Vercel will run `npm install && npm run build` automatically.

### Docker Compose (Full Stack Locally)

```bash
docker compose up -d
# Frontend: http://localhost:3000
# API: http://localhost:8000
```

### Makefile Shortcuts

```bash
make help            # Show all available targets
make install         # Install all dependencies (server + web)
make install-server  # Install Python dependencies
make install-web     # Install frontend dependencies
make install-engine  # Build C++ engine
make dev-server      # Run FastAPI dev server (port 8000)
make dev-web         # Run frontend dev server (port 5173)
make dev-all         # Run both dev servers concurrently
make test            # Run all tests
make test-server     # Run Python server tests
make test-engine     # Run C++ engine tests
make db-upgrade      # Apply Alembic migrations
make docker-up       # Start full stack with Docker Compose
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository and create a feature branch from `main`.
2. Ensure all tests pass: `make test`.
3. Run linters: `make lint` and `make format`.
4. If your change touches the database schema, generate a new Alembic migration: `make db-migrate <description>`.
5. If your change touches C++ code, ensure engine tests pass: `make test-engine`.
6. Submit a pull request with a clear description of the change and any relevant issue references.

## Roadmap

See the `/roadmap` page in the frontend or `web/src/data/roadmap.ts` for the current roadmap. Key shipped milestones:

- C++17 core with custom tensor class (row-major float32, Eigen-backed linear algebra)
- Hand-written ONNX protobuf decoder (no protoc dependency)
- Kahn's algorithm DAG executor with topological scheduling
- Python pybind11 bindings and numpy fallback
- Pure-Rust WASM reimplementation for browser inference
- Full FastAPI backend with auth, analytics, and Prometheus metrics
- TanStack Start frontend with interactive architecture visualization

## License

MIT License — Copyright (c) 2025-2026 Tanmay

See [LICENSE](LICENSE) for the full license text.
