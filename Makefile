.PHONY: help install install-server install-web install-engine submodules test test-server test-web test-engine lint format clean db-upgrade db-migrate docker-up docker-down build-all train-model

# Engine build directory (Release, tests + Python bindings + shared lib).
ENGINE_BUILD ?= build/release
PYTHON ?= python
ENGINE_PY_DIR = $(ENGINE_BUILD)/python

# Default target
help:
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║          🔥 Crucible — Build Automation          ║"
	@echo "╚══════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies"
	@echo "  make install-server   Install Python server deps"
	@echo "  make install-web      Install web frontend deps"
	@echo "  make install-engine   Build C++ engine, bindings, shared lib (CMake)"
	@echo ""
	@echo "Development:"
	@echo "  make dev-server       Run FastAPI dev server (port 8000)"
	@echo "  make dev-web          Run Next.js dev server (port 3000)"
	@echo "  make dev-all          Run both servers"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-server      Run server tests against the native engine"
	@echo "  make test-web         Type-check + lint the web app"
	@echo "  make test-engine      Run C++ ctest + Python parity tests vs ORT"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run all linters"
	@echo "  make lint-server      Lint Python code"
	@echo "  make lint-web         Lint TypeScript"
	@echo "  make format           Format all code"
	@echo ""
	@echo "Database:"
	@echo "  make db-upgrade       Apply pending migrations"
	@echo "  make db-migrate       Generate new migration"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start full stack with Docker Compose"
	@echo "  make docker-down      Stop Docker Compose stack"
	@echo "  make docker-build     Build all Docker images"
	@echo ""
	@echo "Fraud model:"
	@echo "  make train-model      Retrain the fraud model into /tmp/fraud (compare with models/fraud/)"
	@echo ""
	@echo "WASM & CLI:"
	@echo "  make wasm             Build the browser WASM runtime (wasm-pack) into web/public/wasm + wasm/pkg"
	@echo "  make wasm-fixtures    Regenerate WASM ORT parity fixtures (needs onnx + onnxruntime)"
	@echo "  make wasm-test        cargo test + clippy (native and wasm32) for wasm/"
	@echo "  make cli              Build the Rust CLI against the engine shared library"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts"

# ─── Installation ────────────────────────────────────────────────────────────

install: install-server install-web
	@echo "✓ All dependencies installed"

install-server:
	cd server && pip install -r requirements.txt
	@echo "✓ Server dependencies installed"

install-web:
	cd web && npm install
	@echo "✓ Web dependencies installed"

submodules:
	git submodule update --init --depth 1 engine/third_party/eigen engine/third_party/googletest engine/third_party/pybind11

install-engine: submodules
	cmake -S engine -B $(ENGINE_BUILD) -DCMAKE_BUILD_TYPE=Release \
		-DCRUCIBLE_ENABLE_TESTS=ON -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON -DCRUCIBLE_BUILD_SHARED=ON \
		-DPython_EXECUTABLE="$$(command -v $(PYTHON))"
	cmake --build $(ENGINE_BUILD) --config Release --parallel
	@echo "✓ C++ engine built in $(ENGINE_BUILD)"

# ─── Development Servers ─────────────────────────────────────────────────────

dev-server:
	cd server && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	cd web && npm run dev

dev-all:
	@echo "Starting all dev servers..."
	@make -j2 dev-server dev-web

# ─── Testing ─────────────────────────────────────────────────────────────────

test: test-server test-engine
	@echo "✓ All tests passed"

test-server: install-engine
	CRUCIBLE_REQUIRE_NATIVE=1 PYTHONPATH="$(ENGINE_PY_DIR):." $(PYTHON) -m pytest server -q

test-web:
	cd web && npm run type-check && npm run lint

test-engine: install-engine
	ctest --test-dir $(ENGINE_BUILD) -C Release --output-on-failure
	CRUCIBLE_PY_DIR="$(ENGINE_PY_DIR)" $(PYTHON) -m pytest engine/tests/python/test_parity.py -q

test-all:
	@echo "Running full test suite..."
	@make -j2 test-server test-engine test-web

# ─── Linting ─────────────────────────────────────────────────────────────────

lint: lint-server lint-web
	@echo "✓ All linting passed"

lint-server:
	cd server && ruff check . || true

lint-web:
	cd web && npm run lint

# ─── Formatting ──────────────────────────────────────────────────────────────

format:
	cd server && ruff format . || true
	cd web && npm run format

# ─── Database ────────────────────────────────────────────────────────────────

db-upgrade:
	cd server && alembic upgrade head

db-migrate:
	cd server && alembic revision --autogenerate -m "$(filter-out $@,$(MAKECMDGOALS))"
	@echo "✓ Migration created — edit the file in server/alembic/versions/ before applying"

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-up:
	docker compose up -d
	@echo "✓ Stack started — web: http://localhost:3000, api: http://localhost:8000"

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

# ─── Fraud model ─────────────────────────────────────────────────────────────

train-model:
	python models/fraud/train_fraud_model.py --out-dir /tmp/fraud

# ─── Build ───────────────────────────────────────────────────────────────────

build-all: install-engine
	cd server && pip install -r requirements.txt
	cd web && npm run build
	@echo "✓ All builds complete"

# ─── WASM & CLI ──────────────────────────────────────────────────────────────

.PHONY: wasm wasm-fixtures wasm-test cli cli-test

wasm:
	bash scripts/build-wasm.sh

wasm-fixtures:
	$(PYTHON) wasm/tests/gen_fixtures.py

wasm-test:
	cd wasm && cargo test --release
	cd wasm && cargo clippy --all-targets -- -D warnings
	cd wasm && cargo clippy --lib --target wasm32-unknown-unknown -- -D warnings

# The CLI links libcrucible from $(ENGINE_BUILD) (see cli/build.rs); run it with
# LD_LIBRARY_PATH=$(ENGINE_BUILD), e.g.
#   LD_LIBRARY_PATH=build/release cli/target/release/crucible info --model models/resnet18.onnx
cli: install-engine
	CRUCIBLE_LIB_DIR=$(abspath $(ENGINE_BUILD)) cargo build --release --manifest-path cli/Cargo.toml

cli-test: install-engine
	CRUCIBLE_LIB_DIR=$(abspath $(ENGINE_BUILD)) LD_LIBRARY_PATH=$(abspath $(ENGINE_BUILD)) \
		cargo test --release --manifest-path cli/Cargo.toml

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(ENGINE_BUILD) engine/build/
	cd server && rm -rf __pycache__ .pytest_cache .mypy_cache alembic/versions/__pycache__/
	cd web && rm -rf dist .next node_modules/.cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"
