"""SQLAlchemy database layer for the Crucible server.

URLs (``DATABASE_URL``):
  * ``sqlite+aiosqlite:///./crucible.db`` (development default)
  * ``postgresql://user:pass@host/db`` — async sessions use asyncpg, the sync
    CRUD helpers use psycopg2 (both drivers are in requirements.txt).

Auth endpoints use async sessions; the CRUD helpers below are synchronous and
are called from FastAPI threadpool (``def``) endpoints.

Ownership: models, inference logs and fraud cases carry the owning user id.
Every read helper takes ``owner_id`` plus ``all_owners`` (True only for the
operator key) and filters accordingly.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    case,
    create_engine as create_sync_engine,
    func,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, declarative_base, relationship

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(16), nullable=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    rate_limit = Column(Integer, default=100, nullable=False)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="api_keys")


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_type = Column(String(100), nullable=False, default="generic")
    framework = Column(String(100), nullable=False, default="onnx")
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    version = Column(String(50), nullable=False, default="1.0.0")
    input_schema = Column(Text, nullable=True)   # JSON list of {name, shape, dtype}
    output_schema = Column(Text, nullable=True)  # JSON list of {name, shape, dtype}
    metadata_json = Column(Text, nullable=True)  # JSON analysis (opset, reasons, ...)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    operators_supported = Column(Text, nullable=True)  # comma-separated ops used

    inference_logs = relationship("InferenceLog", back_populates="model", cascade="all, delete-orphan")


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    __table_args__ = (
        Index("idx_inference_logs_model_id_created_at", "model_id", "created_at"),
    )

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    model_id = Column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    api_key_id = Column(String(36), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    input_shape = Column(Text, nullable=True)
    output_shape = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=False)
    engine = Column(String(100), nullable=False, default="unknown", index=True)
    status = Column(String(20), nullable=False, default="success", index=True)
    error_message = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)

    model = relationship("ModelRecord", back_populates="inference_logs")


class FraudCase(Base):
    __tablename__ = "fraud_cases"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    inference_log_id = Column(String(36), nullable=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, default="low", index=True)
    features = Column(Text, nullable=True)  # JSON of the submitted transaction fields
    reviewed = Column(Boolean, default=False, nullable=False)
    reviewed_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def _get_database_url() -> str:
    """CRUCIBLE_DB_PATH (tests) > DATABASE_URL. Returns the async URL."""
    from server.config import get_settings

    db_path = os.environ.get("CRUCIBLE_DB_PATH")
    url = f"sqlite+aiosqlite:///{db_path}" if db_path else get_settings().DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    elif url.startswith("postgresql+psycopg2://"):
        url = "postgresql+asyncpg://" + url[len("postgresql+psycopg2://"):]
    elif url.startswith("sqlite:///"):
        url = "sqlite+aiosqlite:///" + url[len("sqlite:///"):]
    return url


def _sync_url(async_url: str) -> str:
    if async_url.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg2://" + async_url[len("postgresql+asyncpg://"):]
    return async_url.replace("+aiosqlite", "")


_engine = None
_async_session_factory = None
_sync_engine = None


def _prep_name_func() -> str:
    return f"__asyncpg_stmt_{uuid.uuid4().hex}__"


def get_engine():
    """Lazy-create the async engine."""
    global _engine
    if _engine is None:
        from server.config import get_settings

        url = _get_database_url()
        connect_args: Dict[str, Any] = {}
        pool_kwargs: Dict[str, Any] = {}
        if url.startswith("postgresql"):
            cache = get_settings().DB_STATEMENT_CACHE_SIZE
            pool_kwargs.update(pool_size=5, max_overflow=10, pool_timeout=30, pool_recycle=1800)
            connect_args = {
                "statement_cache_size": cache,
                "prepared_statement_cache_size": cache,
                "prepared_statement_name_func": _prep_name_func,
            }
        _engine = create_async_engine(url, future=True, connect_args=connect_args, **pool_kwargs)
    return _engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        url = _sync_url(_get_database_url())
        kwargs: Dict[str, Any] = {"pool_pre_ping": True} if url.startswith("postgresql") else {}
        _sync_engine = create_sync_engine(url, future=True, **kwargs)
    return _sync_engine


def reset_engine() -> None:
    """Drop cached engines (test fixtures switch databases)."""
    global _engine, _sync_engine, _async_session_factory
    _engine = None
    _sync_engine = None
    _async_session_factory = None


# Columns added after the initial schema; ALTERed into pre-existing
# development databases (Alembic handles production).
_LATE_COLUMNS = (
    ("inference_logs", "engine", "VARCHAR(100) DEFAULT 'unknown'"),
    ("api_keys", "key_prefix", "VARCHAR(16)"),
    ("fraud_cases", "user_id", "VARCHAR(36)"),
)


def init_db() -> None:
    """Create missing tables/columns (development convenience; use Alembic in production)."""
    engine = _get_sync_engine()
    Base.metadata.create_all(engine)
    for table, column, ddl in _LATE_COLUMNS:
        with engine.connect() as conn:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                conn.commit()
            except Exception:  # noqa: BLE001 - column already exists
                conn.rollback()


def ping() -> bool:
    """True if the database answers ``SELECT 1``."""
    try:
        with _get_sync_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_or(val: Optional[str], default: Any) -> Any:
    if not val:
        return default
    try:
        return json.loads(val)
    except (ValueError, TypeError):
        return default


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _scoped(query, column, owner_id: Optional[str], all_owners: bool):
    return query if all_owners else query.where(column == owner_id)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def save_model(
    *,
    model_id: str,
    name: str,
    file_path: str,
    size_bytes: int,
    owner_id: Optional[str],
    opset: Optional[int],
    operators_used: List[str],
    unsupported_ops: List[str],
    unsupported_reasons: List[str],
    all_supported: bool,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    native_checked: bool,
) -> str:
    meta = {
        "opset": opset,
        "operators_used": operators_used,
        "unsupported_ops": unsupported_ops,
        "unsupported_reasons": unsupported_reasons,
        "all_supported": all_supported,
        "native_checked": native_checked,
    }
    with Session(_get_sync_engine()) as session:
        session.add(ModelRecord(
            id=model_id,
            name=name,
            file_path=file_path,
            file_size=size_bytes,
            input_schema=json.dumps(inputs),
            output_schema=json.dumps(outputs),
            metadata_json=json.dumps(meta),
            operators_supported=",".join(operators_used),
            created_by=owner_id,
            usage_count=0,
        ))
        session.commit()
    return model_id


def _model_dict(rec: ModelRecord) -> Dict[str, Any]:
    meta = _json_or(rec.metadata_json, {})
    if not isinstance(meta, dict):
        meta = {}
    ops = meta.get("operators_used") or (rec.operators_supported.split(",") if rec.operators_supported else [])
    return {
        "id": rec.id,
        "name": rec.name,
        "size_bytes": rec.file_size,
        "created_at": _iso(rec.created_at) or "",
        "opset": meta.get("opset"),
        "operators_used": ops,
        "all_supported": bool(meta.get("all_supported", False)),
        "unsupported_ops": meta.get("unsupported_ops", []),
        "unsupported_reasons": meta.get("unsupported_reasons", []),
        "native_checked": bool(meta.get("native_checked", False)),
        "inputs": _json_or(rec.input_schema, []),
        "outputs": _json_or(rec.output_schema, []),
        "file_path": rec.file_path,
        "created_by": rec.created_by,
        "usage_count": rec.usage_count,
        "last_used": _iso(rec.last_used),
    }


def get_model(model_id: str, *, owner_id: Optional[str], all_owners: bool = False) -> Optional[Dict[str, Any]]:
    """Return the model if it exists and is visible to the caller, else None."""
    with Session(_get_sync_engine()) as session:
        q = _scoped(select(ModelRecord).where(ModelRecord.id == model_id),
                    ModelRecord.created_by, owner_id, all_owners)
        rec = session.execute(q).scalar_one_or_none()
        return _model_dict(rec) if rec else None


def list_models(*, owner_id: Optional[str], all_owners: bool = False) -> List[Dict[str, Any]]:
    with Session(_get_sync_engine()) as session:
        q = _scoped(select(ModelRecord), ModelRecord.created_by, owner_id, all_owners)
        rows = session.execute(q.order_by(ModelRecord.created_at.desc())).scalars().all()
        return [_model_dict(r) for r in rows]


def delete_model(model_id: str) -> bool:
    with Session(_get_sync_engine()) as session:
        rec = session.get(ModelRecord, model_id)
        if rec is None:
            return False
        session.delete(rec)
        session.commit()
    return True


# ---------------------------------------------------------------------------
# Inference logs
# ---------------------------------------------------------------------------
def log_inference(
    *,
    model_id: str,
    user_id: Optional[str],
    api_key_id: Optional[str],
    input_shape: List[int],
    output_shape: Optional[List[int]],
    latency_ms: float,
    backend: str,
    status: str = "success",
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    log_id = uuid.uuid4().hex
    with Session(_get_sync_engine()) as session:
        session.add(InferenceLog(
            id=log_id,
            model_id=model_id,
            user_id=user_id,
            api_key_id=api_key_id,
            input_shape=json.dumps(list(input_shape)),
            output_shape=json.dumps(list(output_shape)) if output_shape is not None else None,
            latency_ms=latency_ms,
            engine=backend,
            status=status,
            error_message=error_message,
        ))
        if status == "success":
            rec = session.get(ModelRecord, model_id)
            if rec is not None:
                rec.usage_count = (rec.usage_count or 0) + 1
                rec.last_used = _now()
        session.commit()
    return {"id": log_id}


def get_inference_logs(*, owner_id: Optional[str], all_owners: bool = False,
                       limit: int = 50) -> List[Dict[str, Any]]:
    with Session(_get_sync_engine()) as session:
        q = _scoped(select(InferenceLog), InferenceLog.user_id, owner_id, all_owners)
        rows = session.execute(q.order_by(InferenceLog.created_at.desc()).limit(limit)).scalars().all()
        return [
            {
                "id": r.id,
                "model_id": r.model_id,
                "latency_ms": r.latency_ms,
                "backend": r.engine,
                "input_shape": _json_or(r.input_shape, []),
                "output_shape": _json_or(r.output_shape, []),
                "status": r.status,
                "created_at": _iso(r.created_at) or "",
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Fraud checks (results of client-side/WASM scoring submitted by the user)
# ---------------------------------------------------------------------------
_FRAUD_FIELDS = ("tx_type", "orig_before", "orig_after", "dest_before", "dest_after",
                 "verdict", "execution_mode", "latency_ms")


def _risk_level(verdict: str) -> str:
    v = verdict.lower()
    if "high" in v:
        return "high"
    if "elevated" in v or "medium" in v:
        return "medium"
    return "low"


def log_fraud_tx(*, user_id: Optional[str], tx_type: str, amount: float, orig_before: float,
                 orig_after: float, dest_before: float, dest_after: float, probability: float,
                 verdict: str, execution_mode: str, latency_ms: float) -> Dict[str, Any]:
    case_id = uuid.uuid4().hex
    fields = {
        "tx_type": tx_type, "orig_before": orig_before, "orig_after": orig_after,
        "dest_before": dest_before, "dest_after": dest_after, "verdict": verdict,
        "execution_mode": execution_mode, "latency_ms": latency_ms,
    }
    risk = _risk_level(verdict)
    with Session(_get_sync_engine()) as session:
        rec = FraudCase(
            id=case_id,
            user_id=user_id,
            transaction_id=f"tx-{case_id}",
            amount=amount,
            fraud_probability=probability,
            is_fraud=risk == "high",
            risk_level=risk,
            features=json.dumps(fields),
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return _fraud_dict(rec)


def _fraud_dict(r: FraudCase) -> Dict[str, Any]:
    feats = _json_or(r.features, {})
    if not isinstance(feats, dict):
        feats = {}
    out: Dict[str, Any] = {k: feats[k] for k in _FRAUD_FIELDS if k in feats}
    out.update(id=r.id, amount=r.amount, probability=r.fraud_probability,
               created_at=_iso(r.created_at) or "")
    return out


def get_fraud_history(*, owner_id: Optional[str], all_owners: bool = False,
                      limit: int = 50) -> List[Dict[str, Any]]:
    with Session(_get_sync_engine()) as session:
        q = _scoped(select(FraudCase), FraudCase.user_id, owner_id, all_owners)
        rows = session.execute(q.order_by(FraudCase.created_at.desc()).limit(limit)).scalars().all()
        return [_fraud_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Analytics (aggregates over the caller's own rows)
# ---------------------------------------------------------------------------
def _cutoff(days: int) -> datetime:
    return _now() - timedelta(days=days)


def inference_analytics(*, owner_id: Optional[str], all_owners: bool, days: int) -> List[Dict[str, Any]]:
    day = func.date(InferenceLog.created_at)
    q = select(
        day.label("date"),
        func.count(InferenceLog.id).label("count"),
        func.avg(InferenceLog.latency_ms).label("avg_latency"),
        func.min(InferenceLog.latency_ms).label("min_latency"),
        func.max(InferenceLog.latency_ms).label("max_latency"),
    ).where(InferenceLog.created_at >= _cutoff(days), InferenceLog.status == "success")
    q = _scoped(q, InferenceLog.user_id, owner_id, all_owners).group_by(day).order_by(day)
    with Session(_get_sync_engine()) as session:
        rows = session.execute(q).all()
    return [
        {
            "date": str(r.date),
            "count": int(r.count),
            "avg_latency_ms": round(float(r.avg_latency), 3) if r.avg_latency is not None else 0,
            "min_latency_ms": round(float(r.min_latency), 3) if r.min_latency is not None else 0,
            "max_latency_ms": round(float(r.max_latency), 3) if r.max_latency is not None else 0,
        }
        for r in rows
    ]


def fraud_analytics(*, owner_id: Optional[str], all_owners: bool, days: int) -> List[Dict[str, Any]]:
    day = func.date(FraudCase.created_at)
    q = select(
        day.label("date"),
        func.count(FraudCase.id).label("total"),
        func.sum(case((FraudCase.is_fraud.is_(True), 1), else_=0)).label("fraud_count"),
        func.avg(FraudCase.fraud_probability).label("avg_probability"),
    ).where(FraudCase.created_at >= _cutoff(days))
    q = _scoped(q, FraudCase.user_id, owner_id, all_owners).group_by(day).order_by(day)
    with Session(_get_sync_engine()) as session:
        rows = session.execute(q).all()
    return [
        {
            "date": str(r.date),
            "total": int(r.total),
            "fraud_count": int(r.fraud_count or 0),
            "avg_probability": round(float(r.avg_probability), 4) if r.avg_probability is not None else 0,
        }
        for r in rows
    ]


def model_analytics(*, owner_id: Optional[str], all_owners: bool) -> List[Dict[str, Any]]:
    q = (
        select(
            ModelRecord.id,
            ModelRecord.name,
            ModelRecord.usage_count,
            ModelRecord.last_used,
            func.count(InferenceLog.id).label("inference_count"),
            func.avg(InferenceLog.latency_ms).label("avg_latency"),
        )
        .outerjoin(InferenceLog, (InferenceLog.model_id == ModelRecord.id) & (InferenceLog.status == "success"))
        .group_by(ModelRecord.id, ModelRecord.name, ModelRecord.usage_count, ModelRecord.last_used)
        .order_by(ModelRecord.usage_count.desc())
    )
    q = _scoped(q, ModelRecord.created_by, owner_id, all_owners)
    with Session(_get_sync_engine()) as session:
        rows = session.execute(q).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "usage_count": int(r.usage_count or 0),
            "inference_count": int(r.inference_count or 0),
            "avg_latency_ms": round(float(r.avg_latency), 3) if r.avg_latency is not None else 0,
            "last_used": _iso(r.last_used),
        }
        for r in rows
    ]
