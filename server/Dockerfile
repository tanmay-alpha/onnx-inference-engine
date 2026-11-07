# Crucible FastAPI server image (used by render.yaml and docker-compose.yml).
#
# Build from the repo root:
#   docker build -f server/Dockerfile -t crucible-server .
# Run:
#   docker run --rm -p 8000:8000 -e CRUCIBLE_SECRET_KEY=... crucible-server
#
# Stage 1 compiles only the pybind11 module `crucible_py` (portable Release
# flags: no -march=native, so the image runs on any x86-64 host).
# Stage 2 is a slim Python runtime with the module on PYTHONPATH; the build
# fails if the module cannot be imported there.

# ============================================================================
# Stage 1: build crucible_py
# ============================================================================
FROM debian:bookworm-slim AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake g++ git ninja-build python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /crucible
COPY engine/CMakeLists.txt engine/CMakeLists.txt
COPY engine/include/       engine/include/
COPY engine/src/           engine/src/
COPY engine/bindings/      engine/bindings/
# Only eigen + pybind11 are needed (tests/benchmarks are off); the other
# submodules are excluded by .dockerignore.
COPY engine/third_party/   engine/third_party/

# Hosts that build from a plain checkout (e.g. Render) may not have
# submodules populated: fetch the two we need at their pinned tags.
RUN if [ ! -f engine/third_party/eigen/CMakeLists.txt ]; then \
        rm -rf engine/third_party/eigen && \
        git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git engine/third_party/eigen; \
    fi \
    && if [ ! -f engine/third_party/pybind11/CMakeLists.txt ]; then \
        rm -rf engine/third_party/pybind11 && \
        git clone --depth 1 --branch v2.12.0 https://github.com/pybind/pybind11.git engine/third_party/pybind11; \
    fi

RUN cmake -S engine -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCRUCIBLE_ENABLE_PYTHON_BINDINGS=ON \
        -DCRUCIBLE_ENABLE_TESTS=OFF \
        -DCRUCIBLE_ENABLE_BENCHMARKS=OFF \
        -DCRUCIBLE_MARCH_NATIVE=OFF \
        -DPython_EXECUTABLE=/usr/bin/python3 \
    && cmake --build build --target crucible_py --parallel \
    && PYTHONPATH=build/python python3 -c "import crucible_py"

# ============================================================================
# Stage 2: runtime
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CRUCIBLE_MODEL_DIR=/crucible/models \
    PYTHONPATH=/crucible/native:/crucible

COPY server/requirements.txt /crucible/server/requirements.txt
RUN pip install --no-cache-dir -r /crucible/server/requirements.txt

COPY server/ /crucible/server/
COPY --from=cpp-builder /crucible/build/python/ /crucible/native/

# Fail the build if the shipped module does not load in this image.
RUN python -c "import crucible_py; print('crucible_py ops:', len(crucible_py.supported_ops()))"

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin crucible \
    && mkdir -p /crucible/models \
    && chown -R crucible:crucible /crucible/models

WORKDIR /crucible
USER crucible
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
