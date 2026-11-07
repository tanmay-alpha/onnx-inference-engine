"""Crucible FastAPI server.

Public:     GET /health, GET /ready, GET /operators, GET /metrics,
            POST /auth/register, POST /auth/login, /docs, /openapi.json
Protected:  everything else, via :func:`server.auth.require_principal`
            (``Authorization: Bearer <JWT>`` or ``X-API-Key``).

Inference
---------
``/infer`` and ``/inference/batch`` run only on the native C++ engine
(``crucible_py``). There is no fallback: without the module they return 503
and ``/health`` reports ``engine: "unavailable"``. ``inference_time_ms`` is
measured around ``crucible_py.run`` only. See :mod:`server.engine` for the
concurrency bound and the 504 timeout semantics (a timed-out native run
cannot be cancelled; it finishes in the background holding its slot).

Uploads
-------
``/convert`` streams the file to ``<CRUCIBLE_MODEL_DIR>/<uuid>.onnx`` (the
client filename is only a display name), then runs ``onnx.checker`` and the
native loader/checker (:mod:`server.validator`). Unsupported models are
stored and owned anyway, flagged with ``unsupported_reasons``.

Limits
------
Request bodies are capped by an ASGI middleware (``MAX_UPLOAD_BYTES`` + 1 MiB
multipart overhead on /convert and /validate, ``MAX_REQUEST_BODY_BYTES``
elsewhere) whether or not Content-Length is sent. ``/infer`` additionally
caps the tensor at ``MAX_INPUT_ELEMENTS``.

Ownership
---------
Models, inference logs and fraud checks belong to the authenticated user;
other users get 404. The optional operator key (``CRUCIBLE_API_KEY``) sees all.

Storage note: model files live on local disk. On hosts with ephemeral disks
(e.g. Render free tier) they disappear on restart while DB rows remain;
/infer then answers 410 for those models.
"""
from __future__ import annotations

import json
import logging
import math
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import status as http_status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select

from server import database, engine, logging_config, metrics, validator
from server.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    KEY_PREFIX_DISPLAY_LEN,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyResponse,
    Principal,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
    create_access_token,
    generate_api_key,
    hash_password,
    require_principal,
    require_user,
    verify_password,
)
from server.config import get_settings
from server.schemas import (
    BatchItemResult,
    BatchResponse,
    ConvertResponse,
    FraudHistoryResponse,
    FraudTxItem,
    FraudTxRequest,
    HealthResponse,
    InferRequest,
    InferResponse,
    InferenceLogListResponse,
    ModelItem,
    ModelListResponse,
    OperatorsResponse,
    ReadyResponse,
    ValidateResponse,
)

settings = get_settings()
logging_config.setup_logging()
_log = logging.getLogger("crucible.server")

SERVER_VERSION = settings.APP_VERSION
BACKEND = engine.BACKEND
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_BYTES
MAX_INPUT_ELEMENTS = settings.MAX_INPUT_ELEMENTS
INFERENCE_TIMEOUT_SEC = settings.INFERENCE_TIMEOUT_SEC
UPLOAD_PATHS = frozenset({"/convert", "/validate"})
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_CHUNK = 1024 * 1024


def _model_dir() -> Path:
    d = Path(os.environ.get("CRUCIBLE_MODEL_DIR", settings.CRUCIBLE_MODEL_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _contained_model_path(file_path: str) -> Optional[Path]:
    """Resolve a stored path and return it only if it lies inside the model dir."""
    try:
        p = Path(file_path).resolve()
    except (OSError, ValueError):
        return None
    return p if p.suffix == ".onnx" and p.is_relative_to(_model_dir()) else None


def _internal_error(exc: BaseException) -> HTTPException:
    trace_id = secrets.token_hex(8)
    _log.error("trace=%s: %s", trace_id, exc, exc_info=exc)
    return HTTPException(status_code=500, detail=f"Internal error (trace {trace_id}).")


# ---------------------------------------------------------------------------
# Body size limit (ASGI middleware; works with or without Content-Length)
# ---------------------------------------------------------------------------
class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    @staticmethod
    def limit_for(path: str) -> int:
        if path in UPLOAD_PATHS:
            return settings.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
        return settings.MAX_REQUEST_BODY_BYTES

    @staticmethod
    async def _send_413(send, limit: int) -> None:
        body = json.dumps({"detail": f"Request body exceeds the {limit}-byte limit"}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope["path"])
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > limit:
                    await self._send_413(send, limit)
                    return
        state = {"received": 0, "exceeded": False, "started": False, "replaced": False}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["exceeded"] = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            # Body parsers may catch our exception and answer 400; turn
            # any response produced after the limit tripped into a 413.
            if state["exceeded"]:
                if message["type"] == "http.response.start" and not state["replaced"]:
                    state["replaced"] = True
                    state["started"] = True
                    await self._send_413(send, limit)
                return
            if message["type"] == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not state["started"]:
                await self._send_413(send, limit)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app_instance: FastAPI):
    database.init_db()
    if not engine.native_available():
        _log.warning("Native engine unavailable: /infer will answer 503.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="REST API for the Crucible ONNX inference engine",
    version=SERVER_VERSION,
    lifespan=_lifespan,
)
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Public
# ===========================================================================
def _health_fields() -> Dict[str, Any]:
    native = engine.native_available()
    return {
        "engine": BACKEND if native else "unavailable",
        "native_available": native,
        "version": SERVER_VERSION,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: 200 while the process serves requests; reports engine availability."""
    return HealthResponse(status="ok", **_health_fields())


@app.get("/ready", response_model=ReadyResponse,
         responses={503: {"model": ReadyResponse}})
def readiness():
    """Readiness: 200 only if the database answers and the native engine is loaded."""
    db_ok = database.ping()
    fields = _health_fields()
    ready = db_ok and fields["native_available"]
    body = ReadyResponse(status="ready" if ready else "not_ready", database=db_ok, **fields)
    if ready:
        return body
    return JSONResponse(status_code=503, content=body.model_dump())


@app.get("/operators", response_model=OperatorsResponse)
def get_operators() -> OperatorsResponse:
    supported = engine.supported_ops()
    return OperatorsResponse(supported=supported, count=len(supported))


@app.get("/metrics")
def get_metrics() -> Response:
    return metrics.metrics_response()


# ===========================================================================
# Auth
# ===========================================================================
def _user_response(user: database.User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


@app.post("/auth/register", response_model=UserResponse, status_code=http_status.HTTP_201_CREATED)
async def register(user_in: UserCreate) -> UserResponse:
    email = user_in.email.lower()
    async with database.get_session_factory()() as session:
        existing = await session.execute(select(database.User).where(database.User.email == email))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered")
        user = database.User(
            id=uuid.uuid4().hex,
            email=email,
            hashed_password=await run_in_threadpool(hash_password, user_in.password),
            full_name=user_in.full_name,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return _user_response(user)


@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin) -> Token:
    async with database.get_session_factory()() as session:
        user = (await session.execute(
            select(database.User).where(database.User.email == credentials.email.lower())
        )).scalar_one_or_none()
        valid = user is not None and await run_in_threadpool(
            verify_password, credentials.password, user.hashed_password
        )
        if not valid:
            metrics.record_error("auth_failed")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated")
        user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()
        token = create_access_token(user.id, user.email, user.is_admin)
    return Token(access_token=token, token_type="bearer", expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)


@app.get("/auth/me", response_model=UserResponse)
async def get_me(principal: Principal = Depends(require_user)) -> UserResponse:
    async with database.get_session_factory()() as session:
        user = await session.get(database.User, principal.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return _user_response(user)


def _key_response(k: database.ApiKey) -> Dict[str, Any]:
    return dict(
        id=k.id,
        name=k.name,
        key_prefix=k.key_prefix or "(not recorded)",
        is_active=k.is_active,
        rate_limit=k.rate_limit,
        last_used=k.last_used.isoformat() if k.last_used else None,
        created_at=k.created_at.isoformat() if k.created_at else "",
        expires_at=k.expires_at.isoformat() if k.expires_at else None,
    )


@app.post("/auth/api-key", response_model=ApiKeyCreated)
async def create_api_key(key_in: ApiKeyCreate, principal: Principal = Depends(require_user)) -> ApiKeyCreated:
    raw_key, key_hash = generate_api_key()
    expires_at = None
    if key_in.expires_in_days:
        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=key_in.expires_in_days)
    async with database.get_session_factory()() as session:
        api_key = database.ApiKey(
            id=uuid.uuid4().hex,
            user_id=principal.user_id,
            key_hash=key_hash,
            key_prefix=raw_key[:KEY_PREFIX_DISPLAY_LEN],
            name=key_in.name,
            expires_at=expires_at,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)
    return ApiKeyCreated(**_key_response(api_key), full_key=raw_key)


@app.get("/auth/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(principal: Principal = Depends(require_user)) -> List[ApiKeyResponse]:
    async with database.get_session_factory()() as session:
        keys = (await session.execute(
            select(database.ApiKey).where(database.ApiKey.user_id == principal.user_id)
            .order_by(database.ApiKey.created_at.desc())
        )).scalars().all()
    return [ApiKeyResponse(**_key_response(k)) for k in keys]


@app.delete("/auth/api-key/{key_id}")
async def revoke_api_key(key_id: str, principal: Principal = Depends(require_user)) -> Dict[str, str]:
    async with database.get_session_factory()() as session:
        key = (await session.execute(
            select(database.ApiKey).where(database.ApiKey.id == key_id,
                                          database.ApiKey.user_id == principal.user_id)
        )).scalar_one_or_none()
        if key is None:
            raise HTTPException(status_code=404, detail="API key not found")
        key.is_active = False
        await session.commit()
    return {"status": "revoked", "key_id": key_id}


# ===========================================================================
# Model upload / validation
# ===========================================================================
async def _store_upload(upload: UploadFile, dest: Path) -> int:
    """Stream an upload to ``dest`` enforcing MAX_UPLOAD_BYTES. Returns size."""
    size = 0
    try:
        with dest.open("wb") as f:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")
    return size


async def _analyze(path: Path) -> validator.ModelAnalysis:
    try:
        return await run_in_threadpool(validator.analyze_model_file, path)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise _internal_error(exc) from exc


def _display_name(filename: Optional[str]) -> str:
    name = Path(filename or "model.onnx").name.strip() or "model.onnx"
    return name[:255]


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    upload: UploadFile = File(..., alias="model_file", description="ONNX model file"),
    input_shape: Optional[str] = Form(
        None, description="Ignored (legacy). Input signatures are read from the model."
    ),
    principal: Principal = Depends(require_principal),
) -> ConvertResponse:
    """Store an ONNX model owned by the caller and return its static analysis.

    Unsupported models are stored too (``all_supported=false`` plus reasons).
    """
    model_id = uuid.uuid4().hex
    path = _model_dir() / f"{model_id}.onnx"
    size = await _store_upload(upload, path)
    analysis = await _analyze(path)
    name = _display_name(upload.filename)
    try:
        await run_in_threadpool(
            lambda: database.save_model(
                model_id=model_id, name=name, file_path=str(path), size_bytes=size,
                owner_id=principal.user_id, opset=analysis.opset,
                operators_used=analysis.operators_used, unsupported_ops=analysis.unsupported_ops,
                unsupported_reasons=analysis.unsupported_reasons,
                all_supported=analysis.all_supported, inputs=analysis.inputs,
                outputs=analysis.outputs, native_checked=analysis.native_checked,
            )
        )
    except Exception as exc:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise _internal_error(exc) from exc
    return ConvertResponse(
        onnx_model_id=model_id,
        name=name,
        size_bytes=size,
        opset=analysis.opset,
        operators_used=analysis.operators_used,
        all_supported=analysis.all_supported,
        unsupported_ops=analysis.unsupported_ops,
        unsupported_reasons=analysis.unsupported_reasons,
        inputs=analysis.inputs,
        outputs=analysis.outputs,
    )


def _get_owned_model(model_id: str, principal: Principal) -> Dict[str, Any]:
    rec = database.get_model(model_id, owner_id=principal.user_id, all_owners=principal.all_owners)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return rec


def _model_file(rec: Dict[str, Any]) -> Path:
    path = _contained_model_path(rec["file_path"])
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=410,
            detail="Model file is no longer on this server's disk (storage is ephemeral); upload it again.",
        )
    return path


@app.post("/validate", response_model=ValidateResponse)
async def validate(
    upload: Optional[UploadFile] = File(None, alias="model_file", description="ONNX file (not stored)"),
    stored_id: Optional[str] = Form(None, alias="model_id", description="Id of a model you own"),
    principal: Principal = Depends(require_principal),
) -> ValidateResponse:
    """Analyse a model without storing it (file) or re-analyse a stored one (model_id)."""
    # Parameters are aliased: FastAPI builds a Pydantic model from form fields
    # and names starting with "model_" collide with Pydantic's namespace.
    if (upload is None) == (stored_id is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of `model_file` or `model_id`")
    if upload is not None:
        path = _model_dir() / f".validate-{uuid.uuid4().hex}.onnx"
        await _store_upload(upload, path)
        try:
            analysis = await _analyze(path)
        finally:
            path.unlink(missing_ok=True)
    else:
        rec = await run_in_threadpool(_get_owned_model, stored_id, principal)
        path = _model_file(rec)
        try:
            analysis = await run_in_threadpool(validator.analyze_model_file, path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ValidateResponse(
        valid=analysis.all_supported,
        operators=analysis.operators_used,
        unsupported=analysis.unsupported_ops,
        unsupported_reasons=analysis.unsupported_reasons,
        opset=analysis.opset,
        inputs=analysis.inputs,
        outputs=analysis.outputs,
    )


# ===========================================================================
# Models
# ===========================================================================
def _model_item(rec: Dict[str, Any]) -> ModelItem:
    return ModelItem(**{k: rec[k] for k in ModelItem.model_fields if k in rec})


@app.get("/models", response_model=ModelListResponse)
def list_models(principal: Principal = Depends(require_principal)) -> ModelListResponse:
    models = database.list_models(owner_id=principal.user_id, all_owners=principal.all_owners)
    return ModelListResponse(models=[_model_item(m) for m in models], count=len(models))


@app.get("/models/{model_id}", response_model=ModelItem)
def get_model(model_id: str, principal: Principal = Depends(require_principal)) -> ModelItem:
    return _model_item(_get_owned_model(model_id, principal))


@app.delete("/models/{model_id}")
def delete_model(model_id: str, principal: Principal = Depends(require_principal)) -> Dict[str, str]:
    rec = _get_owned_model(model_id, principal)
    path = _contained_model_path(rec["file_path"])
    if path is not None:
        path.unlink(missing_ok=True)
    else:
        _log.warning("model %s: stored path outside model dir, file left untouched", model_id)
    database.delete_model(model_id)
    return {"status": "deleted", "model_id": model_id}


# ===========================================================================
# Inference
# ===========================================================================
def _prepare(req: InferRequest, principal: Principal) -> tuple[Path, str, np.ndarray]:
    """Validate an inference request. Raises HTTPException (404/410/413/422)."""
    elements = math.prod(req.input_shape)
    if elements > MAX_INPUT_ELEMENTS:
        raise HTTPException(
            status_code=413,
            detail=f"input_shape has {elements} elements; the limit is {MAX_INPUT_ELEMENTS}",
        )
    if len(req.input) != elements:
        raise HTTPException(
            status_code=422,
            detail=f"input has {len(req.input)} values but input_shape {req.input_shape} needs {elements}",
        )
    rec = _get_owned_model(req.model_id, principal)
    if not rec["all_supported"]:
        reasons = "; ".join(rec["unsupported_reasons"] or rec["unsupported_ops"]) or "unsupported"
        raise HTTPException(status_code=422, detail=f"Model is not supported by Crucible: {reasons}")
    inputs = rec["inputs"]
    if len(inputs) != 1:
        raise HTTPException(status_code=422,
                            detail=f"/infer supports single-input models; this model has {len(inputs)} inputs")
    spec = inputs[0]
    if spec.get("dtype") != "float32":
        raise HTTPException(status_code=422, detail=f"Input {spec.get('name')!r} is {spec.get('dtype')}; only float32 is supported")
    declared = spec.get("shape")
    if declared is not None:
        mismatch = len(declared) != len(req.input_shape) or any(
            d > 0 and d != got for d, got in zip(declared, req.input_shape)
        )
        if mismatch:
            raise HTTPException(
                status_code=422,
                detail=f"input_shape {req.input_shape} does not match the model input {declared} (-1 = any)",
            )
    arr = np.asarray(req.input, dtype=np.float32).reshape(req.input_shape)
    if not np.isfinite(arr).all():
        raise HTTPException(status_code=422, detail="input contains NaN or Inf")
    return _model_file(rec), str(spec["name"]), arr


async def _run(req: InferRequest, principal: Principal) -> InferResponse:
    path, input_name, arr = await run_in_threadpool(_prepare, req, principal)

    async def _log(status: str, latency: float, out_shape: Optional[List[int]] = None,
                   error: Optional[str] = None) -> None:
        metrics.record_inference(BACKEND, latency, status)
        await run_in_threadpool(lambda: database.log_inference(
            model_id=req.model_id, user_id=principal.user_id, api_key_id=principal.api_key_id,
            input_shape=req.input_shape, output_shape=out_shape, latency_ms=latency,
            backend=BACKEND, status=status, error_message=error,
        ))

    try:
        out, elapsed_ms = await engine.run_native(path, input_name, arr, INFERENCE_TIMEOUT_SEC)
    except engine.EngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except engine.EngineBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except engine.EngineTimeout as exc:
        await _log("timeout", INFERENCE_TIMEOUT_SEC * 1000.0, error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except engine.EngineRejected as exc:
        await _log("error", 0.0, error=str(exc))
        raise HTTPException(status_code=422, detail=f"Engine rejected the request: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise _internal_error(exc) from exc

    if not np.isfinite(out).all():
        await _log("error", elapsed_ms, list(out.shape), "non-finite output")
        raise HTTPException(status_code=422,
                            detail="Engine produced NaN/Inf values, which cannot be encoded as JSON")
    await _log("success", elapsed_ms, list(out.shape))
    return InferResponse(
        model_id=req.model_id,
        output=out.reshape(-1).tolist(),
        output_shape=list(out.shape),
        inference_time_ms=elapsed_ms,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest, principal: Principal = Depends(require_principal)) -> InferResponse:
    """Run one inference on a model you own, on the native engine only."""
    return await _run(req, principal)


@app.post("/inference/batch", response_model=BatchResponse)
async def batch_infer(
    requests: List[InferRequest], principal: Principal = Depends(require_principal)
) -> BatchResponse:
    """Run up to MAX_BATCH_SIZE inference requests sequentially; per-item errors are reported inline."""
    if len(requests) > settings.MAX_BATCH_SIZE:
        raise HTTPException(status_code=422,
                            detail=f"Batch size {len(requests)} exceeds the limit of {settings.MAX_BATCH_SIZE}")
    total = sum(math.prod(r.input_shape) for r in requests)
    if total > MAX_INPUT_ELEMENTS:
        raise HTTPException(status_code=413,
                            detail=f"Batch has {total} input elements; the limit is {MAX_INPUT_ELEMENTS}")
    if not engine.native_available():
        raise HTTPException(status_code=503, detail="native engine (crucible_py) is not available on this server")
    results: List[BatchItemResult] = []
    for req in requests:
        try:
            r = await _run(req, principal)
            results.append(BatchItemResult(model_id=req.model_id, status="success", output=r.output,
                                           output_shape=r.output_shape,
                                           inference_time_ms=r.inference_time_ms))
            metrics.record_batch_job("success")
        except HTTPException as exc:
            results.append(BatchItemResult(model_id=req.model_id, status="error",
                                           error=str(exc.detail), status_code=exc.status_code))
            metrics.record_batch_job("error")
    ok = sum(1 for r in results if r.status == "success")
    return BatchResponse(batch_size=len(requests), success_count=ok,
                         error_count=len(results) - ok, results=results)


@app.get("/inference/logs", response_model=InferenceLogListResponse)
def get_inference_logs(
    limit: int = Query(50, ge=1, le=500), principal: Principal = Depends(require_principal)
) -> InferenceLogListResponse:
    logs = database.get_inference_logs(owner_id=principal.user_id, all_owners=principal.all_owners,
                                       limit=limit)
    return InferenceLogListResponse(logs=logs, count=len(logs))


# ===========================================================================
# Fraud check history (scores computed client-side; the server only records them)
# ===========================================================================
@app.post("/fraud/log", response_model=FraudTxItem)
def log_fraud_transaction(req: FraudTxRequest, principal: Principal = Depends(require_principal)) -> FraudTxItem:
    item = database.log_fraud_tx(user_id=principal.user_id, **req.model_dump())
    return FraudTxItem(**item)


@app.get("/fraud/history", response_model=FraudHistoryResponse)
def get_fraud_history(
    limit: int = Query(50, ge=1, le=500), principal: Principal = Depends(require_principal)
) -> FraudHistoryResponse:
    history = database.get_fraud_history(owner_id=principal.user_id, all_owners=principal.all_owners,
                                         limit=limit)
    return FraudHistoryResponse(history=[FraudTxItem(**h) for h in history], count=len(history))


# ===========================================================================
# Analytics (caller's own data)
# ===========================================================================
@app.get("/analytics/inference")
def analytics_inference(days: int = Query(7, ge=1, le=90),
                        principal: Principal = Depends(require_principal)) -> Dict[str, Any]:
    data = database.inference_analytics(owner_id=principal.user_id, all_owners=principal.all_owners, days=days)
    return {"period_days": days, "data": data}


@app.get("/analytics/fraud")
def analytics_fraud(days: int = Query(7, ge=1, le=90),
                    principal: Principal = Depends(require_principal)) -> Dict[str, Any]:
    data = database.fraud_analytics(owner_id=principal.user_id, all_owners=principal.all_owners, days=days)
    return {"period_days": days, "data": data}


@app.get("/analytics/models")
def analytics_models(principal: Principal = Depends(require_principal)) -> Dict[str, Any]:
    return {"models": database.model_analytics(owner_id=principal.user_id, all_owners=principal.all_owners)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
