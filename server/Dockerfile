# Crucible FastAPI server — Dockerfile (Issue #13)
#
# Two-stage build:
#   Stage 1 (cpp-builder) builds the C++ engine and the pybind11
#                       module so the Python image doesn't need
#                       a C++ toolchain.
#   Stage 2 (runtime)    ships only the wheel + the Python server,
#                       giving a much smaller attack surface and
#                       a smaller image.
#
# Build (from repo root):
#   docker build -f server/Dockerfile -t crucible-server .
# Run:
#   docker run --rm -p 8000:8000 crucible-server
#
# The plan §13 calls for "build and run the FastAPI Dockerfile,
# verify it starts on port 8000" — that's the smoke test this
# image is designed to pass.

# ============================================================================
# Stage 1: C++ builder
# ============================================================================
# We use a full Debian-based image here because we need cmake,
# ninja, a C++ compiler, and git (for submodules). The runtime
# stage uses slim, so the extra ~600 MB stays out of the shipped
# artifact.
FROM debian:bookworm-slim AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /crucible

# Copy only what's needed to configure + build. We deliberately
# leave tests, benchmarks, and models out of the build context
# to keep this layer small and fast.
COPY engine/CMakeLists.txt             engine/CMakeLists.txt
COPY engine/CMakePresets.json          engine/CMakePresets.json
COPY engine/include/                   engine/include/
COPY engine/src/                       engine/src/
COPY engine/bindings/                  engine/bindings/
COPY engine/third_party/               engine/third_party/
COPY server/                           server/

# Build in Release mode. The cpp-builder image is throwaway, so
# we don't bother stripping or splitting artifacts.
RUN cmake --preset release -S engine -B build/release \
    && cmake --build build/release --target crucible_py --parallel

# ============================================================================
# Stage 2: runtime
# ============================================================================
# Slim Python 3.11 base. The crucible_py .so lives in
# /crucible/build/release/python; we COPY it directly so the
# runtime image doesn't need CMake.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CRUCIBLE_MODEL_DIR=/crucible/models

# Install only the runtime Python deps. We deliberately leave
# pytest out — it's a dev dep and would add 50 MB to the image.
COPY server/requirements.txt /crucible/server/requirements.txt
RUN pip install --no-cache-dir -r /crucible/server/requirements.txt

# Copy the server source.
COPY server/ /crucible/server/

# Copy the pre-built crucible_py extension from the builder stage.
# The pybind11_add_module target emits to ${CMAKE_BINARY_DIR}/python/
# by default (see engine/bindings/python/CMakeLists.txt), so the
# .so lands at /crucible/build/release/python/crucible_py*.so.
COPY --from=cpp-builder /crucible/build/release/python/ /crucible/build/release/python/

# Where uploaded ONNX models get saved. Persist across container
# restarts by mounting a volume here.
RUN mkdir -p /crucible/models

WORKDIR /crucible

# PYTHONPATH lets `uvicorn app.main:app` resolve crucible_py without
# the user setting the env var manually.
ENV PYTHONPATH=/crucible/build/release/python:/crucible

EXPOSE 8000

# uvicorn with 1 worker is fine for a demo server. For production
# scale-out, replace with `uvicorn.workers.UvicornWorker` behind
# gunicorn or use --workers N.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]