.PHONY: help install install-server install-web install-engine test test-server test-web test-engine lint format clean db-upgrade db-migrate docker-up docker-down build-all

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
	@echo "  make install-engine   Build C++ engine (requires CMake)"
	@echo ""
	@echo "Development:"
	@echo "  make dev-server       Run FastAPI dev server (port 8000)"
	@echo "  make dev-web          Run Next.js dev server (port 3000)"
	@echo "  make dev-all          Run both servers"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-server      Run Python server tests"
	@echo "  make test-web         Run web tests"
	@echo "  make test-engine      Run C++ engine tests"
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
	@echo "ML Pipeline:"
	@echo "  make prepare-data     Prepare fraud detection dataset"
	@echo "  make train-model      Train fraud detection model"
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

install-engine:
	cd engine && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(shell nproc 2>/dev/null || echo 4)
	@echo "✓ C++ engine built"

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

test-server:
	cd server && pytest tests/ -v --tb=short

test-web:
	cd web && npm run type-check && npm run lint

test-engine:
	cd engine && mkdir -p build && cd build && cmake .. && ctest --output-on-failure

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

# ─── ML Pipeline ─────────────────────────────────────────────────────────────

prepare-data:
	cd server && python -m server.ml.prepare_dataset --synthetic --samples 50000
	@echo "✓ Dataset prepared"

train-model:
	cd server && python -m server.ml.training --synthetic
	@echo "✓ Model trained and exported"

# ─── Build ───────────────────────────────────────────────────────────────────

build-all: install-engine
	cd server && pip install -r requirements.txt
	cd web && npm run build
	@echo "✓ All builds complete"

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	cd engine && rm -rf build/
	cd server && rm -rf __pycache__ .pytest_cache .mypy_cache alembic/versions/__pycache__/
	cd web && rm -rf dist .next node_modules/.cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"
