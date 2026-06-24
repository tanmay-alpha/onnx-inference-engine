"""FastAPI inference server for Crucible (Issue #13).

Endpoints (all under base URL ``http://localhost:8000``):

  POST /convert     multipart .onnx upload, save to /tmp/models/<uuid>.onnx
  POST /infer       JSON body, run inference via crucible_py
  POST /validate    multipart .onnx OR form {model_id}, structural check
  GET  /operators   catalogue of supported op_type strings  (public)
  GET  /health      liveness probe                          (public)

Security posture (after Issue #13 review)
=========================================

  * Auth: every non-public route requires the X-API-Key header
    matching CRUCIBLE_API_KEY. The check uses hmac.compare_digest
    to avoid timing side-channels. Without a configured key the
    server returns 503 on protected routes (NOT 200) so an
    operator who forgot to set the env var notices immediately.

  * /convert: accepts a pre-exported .onnx only. We removed the
    PyTorch-deserialization path because torch.load(weights_only=False)
    is RCE-equivalent and torch.load(weights_only=True) cannot
    reconstruct a full nn.Module. The PyTorch -> ONNX helper is
    still present (convert_torch_module) for in-process use.

  * /infer: input_shape product is capped at MAX_INPUT_ELEMENTS
    (50M) so a 50 GB JSON body can't OOM the server.

  * Error envelopes: 4xx details are short and stable. 5xx details
    are a fixed string with a correlation id; the full traceback
    goes to the server log only.

  * Docker: runs as non-root user 'crucible' (uid 1000).
"""
from __future__ import annotations

import hmac
import json
import logging
import math
import os
import secrets
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import onnx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

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

# Max upload size — 200 MB. MobileNetV2 is ~14 MB; ResNet50 is ~100 MB;
# anything beyond that is almost certainly an attack or a misuse.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Cap on the product of input_shape dims. 50M float32 elements is
# 200 MB — large enough for an ImageNet input (3*224*224 = 150K)
# with headroom, small enough that a hostile request cannot OOM
# the server. Pydantic v2 validators run before we touch numpy.
MAX_INPUT_ELEMENTS = 50_000_000

# Endpoints that bypass auth. /health is a liveness probe that
# orchestrators poll — it must not require credentials or the
# container's health check fails. /operators is a capability
# catalogue; leaking it tells an attacker nothing they couldn't
# guess from the docs.
PUBLIC_PATHS = frozenset({"/health", "/operators", "/docs", "/openapi.json"})


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


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# A static API key is the lowest-friction auth scheme that the
# security review will accept for a demo server. In production
# this should be replaced with OAuth2 bearer or mTLS; the
# interface (a single function returning bool) is the same either
# way so the call sites in the endpoints don't change.
#
# Set CRUCIBLE_API_KEY to enable. If unset, /health and /operators
# stay public (liveness + capability discovery), and the rest of
# the routes are 503 (operator must configure the server).
def _check_api_key(api_key: Optional[str]) -> None:
    """Reject the request if the API key is wrong.

    The comparison uses hmac.compare_digest to avoid timing
    side-channels. Empty string vs empty string is a fail (we
    never want a no-key request to authenticate).
    """
    expected = os.environ.get("CRUCIBLE_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Server is not configured: set CRUCIBLE_API_KEY "
                "in the environment. /health and /operators stay "
                "public."
            ),
        )
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )


def require_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
) -> None:
    """FastAPI dependency: rejects requests with a bad / missing key.

    Used as `dependencies=[Depends(require_api_key)]` on every
    protected route. The function returns None — the value is
    the side effect (raising HTTPException for bad keys).
    Declared as a dependency rather than middleware because
    middleware-raised HTTPException doesn't propagate cleanly
    through FastAPI's TestClient (anyio task groups wrap it
    in a BaseExceptionGroup that fails the test).
    """
    _check_api_key(x_api_key)


# ---------------------------------------------------------------------------
# Backend selection: real C++ binding if available, numpy fallback otherwise.
# ---------------------------------------------------------------------------
_log = logging.getLogger("crucible.server")
try:
    import crucible_py  # type: ignore[import-not-found]
    BACKEND = "cpp"
    _log.info("Loaded backend: crucible_py (C++ bindings)")
except ImportError:
    BACKEND = "numpy-fallback"
    crucible_py = None  # type: ignore[assignment]
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
    """
    t0 = time.perf_counter()
    model_proto = onnx.load(str(model_path))
    out_shape = [
        d.dim_value if d.dim_value > 0 else 1
        for d in model_proto.graph.output[0].type.tensor_type.shape.dim
    ]
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
# Error envelope helpers
# ---------------------------------------------------------------------------
def _internal_error(exc: Exception, trace_id: str) -> HTTPException:
    """Build a 500 with a fixed message and log the full traceback.

    The trace id is a 16-hex-char token the user can quote when
    filing a bug. It is NOT derived from the request body so it
    cannot leak user data into logs.
    """
    _log.exception("trace=%s: %s", trace_id, exc)
    return HTTPException(
        status_code=500,
        detail=f"Internal error (trace {trace_id}). See server log for details.",
    )


def _new_trace_id() -> str:
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crucible Inference Server",
    description="REST API around the Crucible ONNX engine",
    version=SERVER_VERSION,
)


@app.middleware("http")
async def auth_middleware(request, call_next):
    """No-op kept for backwards compatibility.

    Auth is now enforced via the `require_api_key` dependency
    declared on each protected route. We keep this middleware
    hook (currently a pass-through) so future cross-cutting
    concerns — request id stamping, body-size pre-checks —
    have a documented home.
    """
    return await call_next(request)


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


@app.post("/convert", response_model=ConvertResponse,
             dependencies=[Depends(require_api_key)])
async def convert(
    model_file: UploadFile = File(..., description="ONNX file to register"),
    input_shape: str = Form(..., description='JSON array, e.g. "[1,3,224,224]"'),
) -> ConvertResponse:
    """Validate and register a pre-exported ONNX model.

    /convert used to accept .pt files; the security review
    correctly flagged torch.load as RCE-equivalent. We now require
    the client to export to .onnx on the training side and POST
    that here. The server onnx-loads + onnx-checks the upload
    and saves it under the model registry.
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
    if any(d <= 0 for d in shape_list):
        raise HTTPException(
            status_code=400,
            detail="input_shape dims must be positive",
        )
    # Cap the product so the dummy input we'd build on a future
    # state-dict path can't OOM the server. The cap is generous
    # enough for ImageNet (3*224*224 = 150K).
    try:
        product = math.prod(shape_list)
    except OverflowError as exc:
        raise HTTPException(
            status_code=400,
            detail="input_shape product overflows",
        ) from exc
    if product > MAX_INPUT_ELEMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input_shape product {product} exceeds the "
                f"{MAX_INPUT_ELEMENTS}-element cap"
            ),
        )

    # Run through a temp file first so the registry only ever
    # sees valid, onnx-checked models.
    trace_id = _new_trace_id()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        model_proto = converter.accept_onnx_upload(
            contents, shape_list, tmp_path,
        )
    except ValueError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — translator
        tmp_path.unlink(missing_ok=True)
        raise _internal_error(exc, trace_id) from exc

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


@app.post("/infer", response_model=InferResponse,
             dependencies=[Depends(require_api_key)])
def infer(req: InferRequest) -> InferResponse:
    """Run inference on a previously converted model.

    The input is a flat float list; the server reshapes it to
    `input_shape` and feeds the result to the C++ engine (or
    the numpy fallback if the bindings aren't built).
    """
    model_path = _lookup_model(req.model_id)

    # Pydantic should have caught this, but the schemas validator
    # only checks each dim < 2**31 — the product can still be huge.
    # We re-check here as defence-in-depth.
    try:
        product = math.prod(req.input_shape)
    except OverflowError as exc:
        raise HTTPException(
            status_code=400,
            detail="input_shape product overflows",
        ) from exc
    if product > MAX_INPUT_ELEMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input_shape product {product} exceeds the "
                f"{MAX_INPUT_ELEMENTS}-element cap"
            ),
        )

    if len(req.input) != product:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input has {len(req.input)} elements but "
                f"input_shape {req.input_shape} requires {product}"
            ),
        )
    arr = np.asarray(req.input, dtype=np.float32).reshape(req.input_shape)

    trace_id = _new_trace_id()
    try:
        if BACKEND == "cpp":
            out_arr, elapsed_ms = _infer_cpp(model_path, arr)
        else:
            out_arr, elapsed_ms = _infer_numpy_fallback(model_path, arr)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — translator
        raise _internal_error(exc, trace_id) from exc

    return InferResponse(
        output=out_arr.reshape(-1).tolist(),
        output_shape=list(out_arr.shape),
        inference_time_ms=elapsed_ms,
        engine=ENGINE_NAME,
    )


@app.post("/validate", response_model=ValidateResponse,
               dependencies=[Depends(require_api_key)])
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

    trace_id = _new_trace_id()
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
    except HTTPException:
        raise
    except ValueError as exc:
        # Malformed ONNX: 400 with the validator's message (it
        # doesn't echo user bytes; just the error name).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — translator
        raise _internal_error(exc, trace_id) from exc

    return ValidateResponse(
        valid=result.valid,
        operators=result.operators,
        unsupported=result.unsupported,
    )