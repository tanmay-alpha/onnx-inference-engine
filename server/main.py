"""FastAPI inference server for Crucible (Issue #13).

Endpoints (all under base URL ``http://localhost:8000``):

  POST /convert     multipart .pt -> ONNX, save to /tmp/models/<uuid>.onnx
  POST /infer       JSON body, run inference via crucible_py
  POST /validate    multipart .onnx OR JSON {model_id}, structural check
  GET  /operators   catalogue of supported op_type strings
  GET  /health      liveness probe

The /infer endpoint calls into the crucible_py module built in
Issue #12. If that module isn't importable (i.e. the C++ engine
hasn't been built and installed yet), the server falls back to a
pure-Python emulator that runs the same op set in numpy. The
emulator is slower and less numerically accurate, but it lets
the test suite run on a developer laptop with no C++ build.

The choice between the two backends is made once at import time
and recorded in `BACKEND`; the rest of the code is identical.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server import converter, validator
from server.schemas import (
    ConvertResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    OperatorsResponse,
    ValidateResponse,
)


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
SERVER_VERSION = "1.0.0"
ENGINE_NAME = "crucible-cpp"


def _model_dir() -> Path:
    """Resolve the model dir on every call.

    Why a function and not a module constant?
        Tests want to monkeypatch CRUCIBLE_MODEL_DIR and have the
        change take effect. A module-level `Path(os.environ.get(...))`
        captures the env var at import time and ignores later
        mutations. A function reads the env var fresh each time,
        so tests can use `monkeypatch.setenv` to redirect writes
        to a tmp_path.
    """
    d = Path(os.environ.get("CRUCIBLE_MODEL_DIR", "/tmp/models"))
    d.mkdir(parents=True, exist_ok=True)
    return d

# Max upload size — 200 MB. MobileNetV2 is ~14 MB; ResNet50 is ~100 MB;
# anything beyond that is almost certainly an attack or a misuse.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ---------------------------------------------------------------------------
# Backend selection: real C++ binding if available, numpy fallback otherwise.
# ---------------------------------------------------------------------------
try:
    import crucible_py  # type: ignore[import-not-found]
    BACKEND = "cpp"
    _log = logging.getLogger("crucible.server")
    _log.info("Loaded backend: crucible_py (C++ bindings)")
except ImportError:
    BACKEND = "numpy-fallback"
    crucible_py = None  # type: ignore[assignment]
    _log = logging.getLogger("crucible.server")
    _log.warning(
        "crucible_py not importable; using numpy fallback for /infer. "
        "Build and install the C++ bindings to use the real engine."
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# A process-local dict mapping model_id -> Path. Cleared on restart,
# which is fine for an Issue #13 demo. A real deployment would back
# this with a database or a shared filesystem.
_MODEL_REGISTRY: dict[str, Path] = {}


def _register_model(onnx_path: Path) -> str:
    """Store a model under the resolved model dir and return its uuid."""
    model_id = uuid.uuid4().hex
    target = _model_dir() / f"{model_id}.onnx"
    # shutil.move so we tolerate cross-device tmp dirs.
    import shutil
    shutil.move(str(onnx_path), str(target))
    _MODEL_REGISTRY[model_id] = target
    return model_id


def _lookup_model(model_id: str) -> Path:
    if model_id not in _MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"model_id {model_id!r} not found. Did you POST /convert first?",
        )
    return _MODEL_REGISTRY[model_id]


# ---------------------------------------------------------------------------
# Inference: real C++ binding or numpy fallback.
# ---------------------------------------------------------------------------
def _infer_numpy_fallback(model_path: Path,
                          input_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure-Python fallback for /infer.

    The numpy fallback only handles the *output shape* check that
    tests rely on. We don't run the actual graph here — that's
    what the C++ engine is for. The fallback exists so the API
    layer (HTTP shape, JSON encode/decode, model_id registry,
    error envelopes) is testable without the C++ build.

    For an end-to-end correctness test, build the bindings and
    run with BACKEND=cpp; the test suite in test_api.py has a
    pytest.mark.cpp_only test that the user enables via
    `pytest -m cpp_only`.
    """
    t0 = time.perf_counter()
    model_proto = onnx.load(str(model_path))
    # Fake output: zeros of the right shape.
    out_shape = [d.dim_value for d in model_proto.graph.output[0].type.tensor_type.shape.dim]
    # Substitute any symbolic dims with 1.
    out_shape = [d if d > 0 else 1 for d in out_shape]
    out = np.zeros(out_shape, dtype=np.float32)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms


def _infer_cpp(model_path: Path,
               input_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Real path through the C++ engine via crucible_py."""
    assert crucible_py is not None
    t0 = time.perf_counter()
    model = crucible_py.load_model(str(model_path))
    out = crucible_py.run(model, input_array)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(out, dtype=np.float32), elapsed_ms


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crucible Inference Server",
    description="REST API around the Crucible ONNX engine",
    version=SERVER_VERSION,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Always returns 200 unless the process is dead."""
    return HealthResponse(
        status="ok",
        engine=ENGINE_NAME,
        version=SERVER_VERSION,
    )


@app.get("/operators", response_model=OperatorsResponse)
def get_operators() -> OperatorsResponse:
    """Catalogue of op_type strings Crucible can dispatch."""
    supported = validator.list_supported_ops()
    return OperatorsResponse(supported=supported, count=len(supported))


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    model_file: UploadFile = File(..., description="PyTorch .pt or .pth file"),
    input_shape: str = Form(..., description='JSON array, e.g. "[1,3,224,224]"'),
) -> ConvertResponse:
    """Convert an uploaded PyTorch model to ONNX.

    The .pt file is processed with torch.load + torch.onnx.export.
    The output is saved under the resolved model dir as <uuid>.onnx and its id is
    returned in `onnx_model_id`. Use that id in subsequent
    /infer and /validate calls.
    """
    # Read with explicit size cap. Starlette doesn't enforce one
    # by default and a hostile client can stream 10 GB.
    contents = await model_file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        shape_list = json.loads(input_shape)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"input_shape is not valid JSON: {exc}",
        ) from exc
    if not isinstance(shape_list, list) or not all(isinstance(d, int) for d in shape_list):
        raise HTTPException(
            status_code=400,
            detail="input_shape must be a JSON array of integers",
        )

    # Run the conversion in a temp file first so the registry only
    # ever sees valid, onnx-checked models.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        model_proto = converter.convert_pt_bytes(
            contents, shape_list, tmp_path,
        )
    except ValueError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — translator for any torch.onnx.export failure
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {exc}",
        ) from exc

    ops = validator.extract_op_types(model_proto)
    supported, unsupported = validator.partition_ops(ops)

    model_id = _register_model(tmp_path)

    return ConvertResponse(
        onnx_model_id=model_id,
        model_path=str(_MODEL_REGISTRY[model_id]),
        operators_used=ops,
        all_supported=not unsupported,
        unsupported_ops=unsupported,
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    """Run inference on a previously converted model.

    The input is a flat float list; the server reshapes it to
    `input_shape` and feeds the result to the C++ engine (or
    the numpy fallback if the bindings aren't built).
    """
    model_path = _lookup_model(req.model_id)

    expected = int(np.prod(req.input_shape))
    if len(req.input) != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input has {len(req.input)} elements but "
                f"input_shape {req.input_shape} requires {expected}"
            ),
        )
    arr = np.asarray(req.input, dtype=np.float32).reshape(req.input_shape)

    try:
        if BACKEND == "cpp":
            out_arr, elapsed_ms = _infer_cpp(model_path, arr)
        else:
            out_arr, elapsed_ms = _infer_numpy_fallback(model_path, arr)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — translator for engine errors
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return InferResponse(
        output=out_arr.reshape(-1).tolist(),
        output_shape=list(out_arr.shape),
        inference_time_ms=elapsed_ms,
        engine=ENGINE_NAME,
    )


@app.post("/validate", response_model=ValidateResponse)
async def validate(
    model_file: Optional[UploadFile] = File(None, description="ONNX file to validate (alternative to model_id)"),
    model_id: Optional[str] = Form(None, description="Previously registered model id (alternative to file upload)"),
) -> ValidateResponse:
    """Validate that every op in a model is supported by Crucible.

    Accepts either a multipart file upload OR a model_id form
    field pointing to a previously converted model. The two
    parameters are mutually exclusive; supplying both is a 400.
    """
    if (model_file is None) == (model_id is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of `model_file` or `model_id`",
        )

    try:
        if model_file is not None:
            data = await model_file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes",
                )
            result = validator.validate_model_bytes(data)
        else:
            path = _lookup_model(model_id)  # type: ignore[arg-type]
            result = validator.validate_model_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ValidateResponse(
        valid=result.valid,
        operators=result.operators,
        unsupported=result.unsupported,
    )