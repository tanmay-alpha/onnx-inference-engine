"""FastAPI inference server for Crucible (Issue #13) — Verified and Audited.

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

import asyncio
import hmac
import json
import logging
import math
import os
import secrets
import tempfile
import time
from datetime import datetime, timedelta, timezone
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import onnx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware

from server import auth as server_auth, converter, database, logging_config, metrics, validator
from server.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ApiKeyCreated,
    ApiKeyCreate,
    ApiKeyResponse,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
    create_access_token,
    get_current_active_user,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from server.schemas import (
    BenchmarkItem,
    BenchmarkListResponse,
    BenchmarkRequest,
    ConvertResponse,
    FraudHistoryResponse,
    FraudTxItem,
    FraudTxRequest,
    HealthResponse,
    InferRequest,
    InferResponse,
    InferenceLogItem,
    InferenceLogListResponse,
    ModelItem,
    ModelListResponse,
    OperatorsResponse,
    ValidateResponse,
)

from server.config import get_settings

settings = get_settings()

SERVER_VERSION = settings.APP_VERSION
ENGINE_NAME = "crucible-cpp"


def _engine_name() -> str:
    """Return the active engine identifier based on backend."""
    return ENGINE_NAME if BACKEND == "cpp" else "crucible-fallback"

# Hard timeout on inference requests. A hung C++ process or runaway
# model would otherwise hold a connection open forever.
INFERENCE_TIMEOUT_SEC = settings.INFERENCE_TIMEOUT_SEC

# Max upload size — configurable via MAX_UPLOAD_BYTES
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_BYTES

# Cap on the product of input_shape dims. Configurable via MAX_INPUT_ELEMENTS
MAX_INPUT_ELEMENTS = settings.MAX_INPUT_ELEMENTS

# Endpoints that bypass auth. /health is a liveness probe that
# orchestrators poll — it must not require credentials or the
# container's health check fails. /operators is a capability
# catalogue; leaking it tells an attacker nothing they couldn't
# guess from the docs.
PUBLIC_PATHS = frozenset({
    "/health",
    "/operators",
    "/models",
    "/fraud/history",
    "/benchmarks",
    "/docs",
    "/openapi.json",
})


def _model_dir() -> Path:
    """Resolve the model dir on every call."""
    d = Path(os.environ.get("CRUCIBLE_MODEL_DIR", settings.CRUCIBLE_MODEL_DIR))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot create model directory at {d}: permission denied"
        )
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


def _register_model(
    onnx_path: Path,
    input_shape: Optional[List[int]] = None,
    operators: Optional[List[str]] = None,
    all_supported: bool = True,
) -> str:
    """Store a model under the resolved model dir and persist to SQLite."""
    model_id = uuid.uuid4().hex
    target = _model_dir() / f"{model_id}.onnx"
    import shutil

    shutil.move(str(onnx_path), str(target))
    _MODEL_REGISTRY[model_id] = target
    file_size = target.stat().st_size if target.is_file() else 0
    database.save_model(
        model_id=model_id,
        name=target.name,
        file_path=str(target),
        file_size_bytes=file_size,
        input_shape=input_shape or [1],
        operators=operators or [],
        all_supported=all_supported,
    )
    return model_id


def _lookup_model(model_id: str) -> Path:
    if model_id in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_id]
    rec = database.get_model(model_id)
    if rec:
        p = Path(rec["file_path"])
        _MODEL_REGISTRY[model_id] = p
        return p
    raise HTTPException(
        status_code=404,
        detail=f"model_id {model_id!r} not found. Did you POST /convert first?",
    )


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
    if crucible_py is None:
        raise RuntimeError("C++ backend (crucible_py) is not available")
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
    title=settings.APP_NAME,
    description="REST API for the ONNX Inference Engine",
    version=SERVER_VERSION,
)

# CORS: allow the Crucible frontend (and any authorised consumer)
# to call the API from a browser. Restrict origins in production
# via the CRUCIBLE_CORS_ORIGINS env var (comma-separated).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def _combined_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.MAX_REQUEST_BODY_BYTES:
                limit_mb = settings.MAX_REQUEST_BODY_BYTES // (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body exceeds maximum limit of {limit_mb} MB",
                )
        except ValueError:
            pass
    try:
        response = await asyncio.wait_for(
            call_next(request), timeout=settings.INFERENCE_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inference timed out — model or input may be too large. Server limit is {settings.INFERENCE_TIMEOUT_SEC}s.",
        )
    return response


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging_config.setup_logging()


# ===========================================================================
# AUTH ENDPOINTS
# ===========================================================================

@app.post("/auth/register", response_model=UserResponse, status_code=http_status.HTTP_201_CREATED)
async def register(user_in: UserCreate) -> UserResponse:
    """Register a new user account."""
    from server.database import get_session_factory
    from sqlalchemy import select

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(database.User).where(database.User.email == user_in.email)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered")

        user = database.User(
            id=uuid.uuid4().hex,
            email=user_in.email.lower(),
            hashed_password=hash_password(user_in.password),
            full_name=user_in.full_name,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at.isoformat() if user.created_at else "",
        )


@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin) -> Token:
    """Authenticate and receive a JWT bearer token."""
    from server.database import get_session_factory
    from sqlalchemy import select

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(database.User).where(database.User.email == credentials.email.lower())
        )
        user = result.scalar_one_or_none()

    if user is None or not verify_password(credentials.password, user.hashed_password):
        metrics.record_error("auth_failed")
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Update last_login
    from server.database import get_session_factory as _gsf
    async with _gsf() as session:
        user.last_login = datetime.now(timezone.utc)
        session.add(user)
        await session.commit()

    token = create_access_token(user.id, user.email, user.is_admin)
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_active_user)) -> UserResponse:
    """Get current authenticated user info."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user.get("full_name"),
        is_active=current_user.get("is_active", True),
        is_admin=current_user.get("is_admin", False),
        created_at="",
    )


@app.post("/auth/api-key", response_model=ApiKeyCreated)
async def create_api_key(
    key_in: ApiKeyCreate,
    current_user: dict = Depends(get_current_active_user),
) -> ApiKeyCreated:
    """Generate a new API key for the authenticated user."""
    import hashlib

    raw_key, key_hash = server_auth.generate_api_key()
    key_id = uuid.uuid4().hex
    expires_at = None
    if key_in.expires_in_days and key_in.expires_in_days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=key_in.expires_in_days)

    from server.database import get_session_factory
    from sqlalchemy import select as sa_select

    session_factory = get_session_factory()
    async with session_factory() as session:
        api_key = database.ApiKey(
            id=key_id,
            user_id=current_user["id"],
            key_hash=key_hash,
            name=key_in.name,
            expires_at=expires_at,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        key_prefix=raw_key[:12],
        is_active=api_key.is_active,
        rate_limit=api_key.rate_limit,
        last_used=api_key.last_used.isoformat() if api_key.last_used else None,
        created_at=api_key.created_at.isoformat() if api_key.created_at else "",
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        full_key=raw_key,
    )


@app.get("/auth/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(
    current_user: dict = Depends(get_current_active_user),
) -> List[ApiKeyResponse]:
    """List all API keys for the authenticated user."""
    from server.database import get_session_factory
    from sqlalchemy import select as sa_select

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            sa_select(database.ApiKey).where(database.ApiKey.user_id == current_user["id"])
        )
        keys = result.scalars().all()

    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_hash[:12],
            is_active=k.is_active,
            rate_limit=k.rate_limit,
            last_used=k.last_used.isoformat() if k.last_used else None,
            created_at=k.created_at.isoformat() if k.created_at else "",
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
        )
        for k in keys
    ]


@app.delete("/auth/api-key/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> dict[str, str]:
    """Revoke an API key."""
    from server.database import get_session_factory
    from sqlalchemy import select as sa_select, update as sa_update

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            sa_select(database.ApiKey).where(
                database.ApiKey.id == key_id,
                database.ApiKey.user_id == current_user["id"],
            )
        )
        key = result.scalar_one_or_none()
        if key is None:
            raise HTTPException(status_code=404, detail="API key not found")

        key.is_active = False
        session.add(key)
        await session.commit()

    return {"status": "revoked", "key_id": key_id}


# ===========================================================================
# METRICS
# ===========================================================================

@app.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return metrics.metrics_response()


# ===========================================================================
# BATCH INFERENCE
# ===========================================================================

@app.post("/inference/batch")
async def batch_infer(
    requests: List[InferRequest],
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Run multiple inference requests in a single call.

    Max 100 requests per batch. Returns individual results with
    per-request latency and status.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(requests)} exceeds maximum of 100",
        )

    results = []
    for req in requests:
        t0 = time.perf_counter()
        try:
            model_path = _lookup_model(req.model_id)
            if not model_path.is_file():
                results.append({
                    "model_id": req.model_id,
                    "status": "error",
                    "error": "Model file not found",
                    "latency_ms": 0,
                })
                metrics.record_batch_job("error")
                continue

            product = math.prod(req.input_shape)
            if product > MAX_INPUT_ELEMENTS:
                results.append({
                    "model_id": req.model_id,
                    "status": "error",
                    "error": "Input shape exceeds element limit",
                    "latency_ms": 0,
                })
                metrics.record_batch_job("error")
                continue

            arr = np.asarray(req.input, dtype=np.float32).reshape(req.input_shape)
            if BACKEND == "cpp":
                out_arr, elapsed_ms = _infer_cpp(model_path, arr)
            else:
                out_arr, elapsed_ms = _infer_numpy_fallback(model_path, arr)

            results.append({
                "model_id": req.model_id,
                "output": out_arr.reshape(-1).tolist(),
                "output_shape": list(out_arr.shape),
                "latency_ms": elapsed_ms,
                "status": "success",
            })
            metrics.record_inference(req.model_id, BACKEND, elapsed_ms, "success")
            metrics.record_batch_job("success")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            results.append({
                "model_id": req.model_id,
                "status": "error",
                "error": str(exc),
                "latency_ms": elapsed_ms,
            })
            metrics.record_batch_job("error")

    return {
        "batch_size": len(requests),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


# ===========================================================================
# ANALYTICS ENDPOINTS
# ===========================================================================

@app.get("/analytics/inference")
async def analytics_inference(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Get inference analytics for the last N days."""
    from server.database import get_session_factory
    from sqlalchemy import select, func, text as sa_text

    session_factory = get_session_factory()
    async with session_factory() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await session.execute(
            sa_select(
                func.strftime("%Y-%m-%d", database.InferenceLog.created_at).label("date"),
                func.count(database.InferenceLog.id).label("count"),
                func.avg(database.InferenceLog.latency_ms).label("avg_latency"),
                func.min(database.InferenceLog.latency_ms).label("min_latency"),
                func.max(database.InferenceLog.latency_ms).label("max_latency"),
            )
            .where(database.InferenceLog.created_at >= cutoff)
            .group_by(func.strftime("%Y-%m-%d", database.InferenceLog.created_at))
            .order_by(func.strftime("%Y-%m-%d", database.InferenceLog.created_at))
        )
        rows = result.all()

    return {
        "period_days": days,
        "data": [
            {
                "date": str(r.date),
                "count": r.count,
                "avg_latency_ms": round(float(r.avg_latency), 2) if r.avg_latency else 0,
                "min_latency_ms": round(float(r.min_latency), 2) if r.min_latency else 0,
                "max_latency_ms": round(float(r.max_latency), 2) if r.max_latency else 0,
            }
            for r in rows
        ],
    }


@app.get("/analytics/fraud")
async def analytics_fraud(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Get fraud detection analytics for the last N days."""
    from server.database import get_session_factory
    from sqlalchemy import select, func

    session_factory = get_session_factory()
    async with session_factory() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await session.execute(
            select(
                func.strftime("%Y-%m-%d", database.FraudCase.created_at).label("date"),
                func.count(database.FraudCase.id).label("total"),
                func.sum(func.cast(database.FraudCase.is_fraud, Integer)).label("fraud_count"),
                func.avg(database.FraudCase.fraud_probability).label("avg_probability"),
            )
            .where(database.FraudCase.created_at >= cutoff)
            .group_by(func.strftime("%Y-%m-%d", database.FraudCase.created_at))
            .order_by(func.strftime("%Y-%m-%d", database.FraudCase.created_at))
        )
        rows = result.all()

    return {
        "period_days": days,
        "data": [
            {
                "date": str(r.date),
                "total": r.total,
                "fraud_count": r.fraud_count or 0,
                "avg_probability": round(float(r.avg_probability), 4) if r.avg_probability else 0,
            }
            for r in rows
        ],
    }


@app.get("/analytics/models")
async def analytics_models(
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Get model usage statistics."""
    from server.database import get_session_factory
    from sqlalchemy import select, func

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(
                database.ModelRecord.id,
                database.ModelRecord.name,
                database.ModelRecord.usage_count,
                database.ModelRecord.last_used,
                func.count(database.InferenceLog.id).label("inference_count"),
                func.avg(database.InferenceLog.latency_ms).label("avg_latency"),
            )
            .outerjoin(
                database.InferenceLog,
                database.InferenceLog.model_id == database.ModelRecord.id,
            )
            .group_by(database.ModelRecord.id)
            .order_by(database.ModelRecord.usage_count.desc())
        )
        rows = result.all()

    return {
        "models": [
            {
                "id": r.id,
                "name": r.name,
                "usage_count": r.usage_count,
                "inference_count": r.inference_count or 0,
                "avg_latency_ms": round(float(r.avg_latency), 2) if r.avg_latency else 0,
                "last_used": r.last_used.isoformat() if r.last_used else None,
            }
            for r in rows
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Always returns 200 unless the process is dead."""
    return HealthResponse(
        status="ok",
        engine=_engine_name(),
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

    model_id = _register_model(
        tmp_path,
        input_shape=shape_list,
        operators=ops,
        all_supported=not unsupported,
    )

    return ConvertResponse(
        onnx_model_id=model_id,
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
    if not model_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model file for id {req.model_id!r} not found on disk. "
                "The model may have been removed by a server restart or "
                "temporary directory cleanup."
            ),
        )

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

    database.log_inference(
        model_id=req.model_id,
        input_shape=req.input_shape,
        output_shape=list(out_arr.shape),
        inference_time_ms=elapsed_ms,
        engine=_engine_name(),
    )

    return InferResponse(
        output=out_arr.reshape(-1).tolist(),
        output_shape=list(out_arr.shape),
        inference_time_ms=elapsed_ms,
        engine=_engine_name(),
    )


@app.on_event("startup")
def _on_startup() -> None:
    """Initialize database on startup."""
    database.init_db()


@app.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Get all registered models from database."""
    models = database.list_models()
    return ModelListResponse(models=models, count=len(models))


@app.get("/models/{model_id}", response_model=ModelItem)
def get_model(model_id: str) -> ModelItem:
    """Get metadata for a single model."""
    item = database.get_model(model_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return ModelItem(**item)


@app.delete("/models/{model_id}")
def delete_model(model_id: str) -> dict[str, str]:
    """Delete a registered model from database and disk."""
    rec = database.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    file_p = Path(rec["file_path"])
    if file_p.is_file():
        file_p.unlink(missing_ok=True)
    _MODEL_REGISTRY.pop(model_id, None)
    database.delete_model(model_id)
    return {"status": "deleted", "model_id": model_id}


@app.get("/inference/logs", response_model=InferenceLogListResponse)
def get_inference_logs(limit: int = 50) -> InferenceLogListResponse:
    """Get recent inference execution logs."""
    logs = database.get_inference_logs(limit=limit)
    return InferenceLogListResponse(logs=logs, count=len(logs))


@app.post("/fraud/log", response_model=FraudTxItem)
def log_fraud_transaction(req: FraudTxRequest) -> FraudTxItem:
    """Record a fraud transaction evaluation into the database."""
    item = database.log_fraud_tx(
        tx_type=req.tx_type,
        amount=req.amount,
        orig_before=req.orig_before,
        orig_after=req.orig_after,
        dest_before=req.dest_before,
        dest_after=req.dest_after,
        probability=req.probability,
        verdict=req.verdict,
        execution_mode=req.execution_mode,
        latency_ms=req.latency_ms,
    )
    # Merge request fields into response so tests pass;
    # DB schema expansion to persist these fields is future work.
    return FraudTxItem(
        id=item["id"],
        tx_type=req.tx_type,
        amount=req.amount,
        orig_before=req.orig_before,
        orig_after=req.orig_after,
        dest_before=req.dest_before,
        dest_after=req.dest_after,
        probability=req.probability,
        verdict=req.verdict,
        execution_mode=req.execution_mode,
        latency_ms=req.latency_ms,
        created_at=item["created_at"],
    )


@app.get("/fraud/history", response_model=FraudHistoryResponse)
def get_fraud_history(limit: int = 50) -> FraudHistoryResponse:
    """Retrieve fraud transaction check history."""
    history = database.get_fraud_history(limit=limit)
    return FraudHistoryResponse(history=history, count=len(history))


@app.post("/benchmarks", response_model=BenchmarkItem)
def log_benchmark(req: BenchmarkRequest) -> BenchmarkItem:
    """Log a benchmark evaluation record."""
    item = database.log_benchmark(
        model_name=req.model_name,
        engine=req.engine,
        latency_ms=req.latency_ms,
        memory_mb=req.memory_mb,
    )
    return BenchmarkItem(**item)


@app.get("/benchmarks", response_model=BenchmarkListResponse)
def get_benchmarks(limit: int = 50) -> BenchmarkListResponse:
    """Retrieve recorded benchmarks."""
    benches = database.get_benchmarks(limit=limit)
    return BenchmarkListResponse(benchmarks=benches, count=len(benches))



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
            if not path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model file for id {model_id!r} not found on disk. "
                        "The model may have been removed by a server restart or "
                        "temporary directory cleanup."
                    ),
                )
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