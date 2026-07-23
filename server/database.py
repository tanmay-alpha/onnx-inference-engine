"""SQLAlchemy async database layer for Crucible Server.

Supports both SQLite (development) and PostgreSQL (production) via
the DATABASE_URL environment variable.

  Development:  DATABASE_URL=sqlite+aiosqlite:///./crucible.db
  Production:   DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname

ORM Models (imported by alembic/env.py for migrations):
  users, api_keys, models, inference_logs, fraud_cases, benchmarks

CRUD functions (used by server/main.py and tests):
  save_model, get_model, list_models, delete_model,
  log_inference, get_inference_logs,
  log_fraud_tx, get_fraud_history,
  log_benchmark, get_benchmarks
"""
from __future__ import annotations

import json
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

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
    func,
    create_engine as create_sync_engine,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
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
    fraud_cases = relationship("FraudCase", back_populates="reviewer")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
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
    input_schema = Column(Text, nullable=True)
    output_schema = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    operators_supported = Column(Text, nullable=True)

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
    inference_log_id = Column(String(36), nullable=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, default="low", index=True)
    features = Column(Text, nullable=True)
    reviewed = Column(Boolean, default=False, nullable=False)
    reviewed_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=True)

    reviewer = relationship("User", back_populates="fraud_cases")


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    model_name = Column(String(255), nullable=False)
    engine = Column(String(100), nullable=False, index=True)
    latency_ms = Column(Float, nullable=False)
    memory_mb = Column(Float, nullable=True)
    device = Column(String(50), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
def _get_database_url() -> str:
    """Resolve the database URL from environment settings.

    Priority: CRUCIBLE_DB_PATH (test fixture) > DATABASE_URL > default.
    """
    from server.config import get_settings
    settings = get_settings()

    db_path = os.environ.get("CRUCIBLE_DB_PATH")
    if db_path:
        url = f"sqlite+aiosqlite:///{db_path}"
    else:
        url = settings.DATABASE_URL

    # Allow plain "postgresql://" from tools that don't add the async driver
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# Engine + Session (async)
# ---------------------------------------------------------------------------
_engine = None
_async_session_factory = None


def _prep_name_func():
    return f"__asyncpg_stmt_{uuid.uuid4().hex}__"


def get_engine():
    """Lazy-create the async engine."""
    global _engine
    if _engine is None:
        from server.config import get_settings
        settings = get_settings()
        url = _get_database_url()
        connect_args = {}
        pool_kwargs = {}
        if "postgresql" in url:
            from sqlalchemy.pool import NullPool
            pool_kwargs["poolclass"] = NullPool
            connect_args = {
                "statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
                "prepared_statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
                "prepared_statement_name_func": _prep_name_func,
            }
        _engine = create_async_engine(
            url, echo=False, future=True, connect_args=connect_args, **pool_kwargs
        )
    return _engine


def get_session_factory():
    """Lazy-create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db() -> None:
    """Create all tables (development only — use Alembic in production)."""
    engine = _get_sync_engine()
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Sync helpers (used by tests and sync CRUD endpoints)
# ---------------------------------------------------------------------------
_sync_engine = None


def _get_sync_engine():
    """Lazy-create a synchronous engine for CRUD queries."""
    global _sync_engine
    if _sync_engine is None:
        url = _get_database_url()
        sync_url = url.replace("+aiosqlite", "").replace("+asyncpg", "").replace("+psycopg2", "")
        _sync_engine = create_sync_engine(sync_url, echo=False, future=True)
    return _sync_engine


def reset_engine() -> None:
    """Reset engine state (used by test fixtures)."""
    global _engine, _sync_engine, _async_session_factory
    _engine = None
    _sync_engine = None
    _async_session_factory = None


# ---------------------------------------------------------------------------
# CRUD — Models (sync, used by main.py sync endpoints)
# ---------------------------------------------------------------------------
def save_model(
    model_id: str,
    name: str,
    file_path: str,
    file_size_bytes: int = 0,
    input_shape: Optional[List[int]] = None,
    operators: Optional[List[str]] = None,
    all_supported: bool = True,
) -> str:
    """Persist a model record. Returns the model_id."""
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        rec = ModelRecord(
            id=model_id,
            name=name,
            file_path=file_path,
            file_size=file_size_bytes,
            input_schema=str(input_shape) if input_shape else None,
            metadata_json=str({"operators": operators or [], "all_supported": all_supported}),
            operators_supported=",".join(operators or []),
            usage_count=0,
        )
        session.add(rec)
        session.commit()
    return model_id


def _parse_shape(val: Optional[str]) -> List[int]:
    if not val:
        return []
    try:
        parsed = json.loads(val)
        if isinstance(parsed, dict):
            return parsed.get("shape", [])
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except Exception:
        pass
    return []


def get_model(model_id: str) -> Optional[dict]:
    """Return a model record as a dict, or None."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        result = session.execute(select(ModelRecord).where(ModelRecord.id == model_id))
        rec = result.scalar_one_or_none()
        if rec is None:
            return None
        return {
            "id": rec.id,
            "name": rec.name,
            "description": rec.description,
            "model_type": rec.model_type,
            "framework": rec.framework,
            "file_path": rec.file_path,
            "file_size_bytes": rec.file_size,
            "version": rec.version,
            "input_shape": _parse_shape(rec.input_schema),
            "output_shape": _parse_shape(rec.output_schema),
            "is_active": rec.is_active,
            "created_by": rec.created_by,
            "created_at": rec.created_at.isoformat() if rec.created_at else "",
            "updated_at": rec.updated_at.isoformat() if rec.updated_at else "",
            "last_used": rec.last_used.isoformat() if rec.last_used else None,
            "usage_count": rec.usage_count,
            "operators": rec.operators_supported.split(",") if rec.operators_supported else [],
            "all_supported": len(rec.operators_supported.split(",")) > 0 if rec.operators_supported else False,
        }


def list_models() -> List[dict]:
    """Return all model records as dicts."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        result = session.execute(
            select(ModelRecord).order_by(ModelRecord.created_at.desc())
        )
        return [
            {
                "id": r.id,
                "name": r.name,
                "file_path": r.file_path,
                "file_size_bytes": r.file_size,
                "input_shape": _parse_shape(r.input_schema),
                "output_shape": _parse_shape(r.output_schema),
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "last_used": r.last_used.isoformat() if r.last_used else None,
                "usage_count": r.usage_count,
                "operators": r.operators_supported.split(",") if r.operators_supported else [],
                "all_supported": len(r.operators_supported.split(",")) > 0 if r.operators_supported else False,
            }
            for r in result.scalars().all()
        ]


def delete_model(model_id: str) -> bool:
    """Delete a model record. Returns True if deleted, False if not found."""
    from sqlalchemy import select, delete as sa_delete
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        result = session.execute(
            select(ModelRecord).where(ModelRecord.id == model_id)
        )
        rec = result.scalar_one_or_none()
        if rec is None:
            return False
        session.execute(sa_delete(ModelRecord).where(ModelRecord.id == model_id))
        session.commit()
    return True


# ---------------------------------------------------------------------------
# CRUD — Inference Logs
# ---------------------------------------------------------------------------
def log_inference(model_id: str, input_shape, output_shape,
                  inference_time_ms: float, engine: str,
                  status: str = "success", error_message: str = "") -> dict:
    """Log an inference request. Returns the logged record."""
    from sqlalchemy import insert
    from sqlalchemy.orm import Session
    import uuid as _uuid

    log_id = _uuid.uuid4().hex
    session = Session(_get_sync_engine())
    try:
        session.execute(insert(InferenceLog).values(
            id=log_id,
            model_id=model_id,
            input_shape=str(input_shape) if input_shape else None,
            output_shape=str(output_shape) if output_shape else None,
            latency_ms=inference_time_ms,
            engine=engine,
            status=status,
            error_message=error_message or None,
        ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return {
        "id": log_id,
        "model_id": model_id,
        "input_shape": list(input_shape) if input_shape else [],
        "output_shape": list(output_shape) if output_shape else [],
        "inference_time_ms": inference_time_ms,
        "engine": engine,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def get_inference_logs(model_id: Optional[str] = None, limit: int = 50) -> List[dict]:
    """Get recent inference logs (sync), optionally filtered by model_id."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        query = select(InferenceLog).order_by(InferenceLog.created_at.desc()).limit(limit)
        if model_id:
            query = query.where(InferenceLog.model_id == model_id)
        result = session.execute(query)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "model_id": r.model_id,
                "input_shape": json.loads(r.input_shape) if r.input_shape else [],
                "output_shape": json.loads(r.output_shape) if r.output_shape else [],
                "inference_time_ms": r.latency_ms,
                "engine": r.engine,
                "status": r.status,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# CRUD — Fraud Transactions
# ---------------------------------------------------------------------------
def log_fraud_tx(
    tx_type: str = "",
    amount: float = 0.0,
    orig_before: float = 0.0,
    orig_after: float = 0.0,
    dest_before: float = 0.0,
    dest_after: float = 0.0,
    probability: float = 0.0,
    verdict: str = "",
    execution_mode: str = "wasm",
    latency_ms: float = 0.0,
    transaction_id: str = "",
    is_fraud: bool = False,
    risk_level: str = "low",
    inference_log_id: str = "",
    features: str = "",
) -> dict:
    """Log a fraud detection result."""
    from sqlalchemy import insert
    from sqlalchemy.orm import Session

    case_id = uuid.uuid4().hex
    tx_id = transaction_id or f"tx-{case_id}"
    session = Session(_get_sync_engine())
    try:
        session.execute(insert(FraudCase).values(
            id=case_id,
            transaction_id=tx_id,
            inference_log_id=inference_log_id or None,
            amount=amount,
            fraud_probability=probability,
            is_fraud=is_fraud,
            risk_level=risk_level,
            features=features or None,
        ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return {
        "id": case_id,
        "transaction_id": tx_id,
        "amount": amount,
        "fraud_probability": probability,
        "probability": probability,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def get_fraud_history(limit: int = 50) -> List[dict]:
    """Get fraud detection history (sync)."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        query = select(FraudCase).order_by(FraudCase.created_at.desc()).limit(limit)
        result = session.execute(query)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "transaction_id": r.transaction_id,
                "amount": r.amount,
                "fraud_probability": r.fraud_probability,
                "is_fraud": r.is_fraud,
                "risk_level": r.risk_level,
                "reviewed": r.reviewed,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# CRUD — Benchmarks
# ---------------------------------------------------------------------------
def log_benchmark(
    model_name: str,
    engine: str,
    latency_ms: float,
    memory_mb: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Record a benchmark result."""
    from sqlalchemy import insert
    from sqlalchemy.orm import Session

    bench_id = uuid.uuid4().hex
    session = Session(_get_sync_engine())
    try:
        session.execute(insert(Benchmark).values(
            id=bench_id,
            model_name=model_name,
            engine=engine,
            latency_ms=latency_ms,
            memory_mb=memory_mb if memory_mb > 0 else None,
            device=device,
        ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return {
        "id": bench_id,
        "model_name": model_name,
        "engine": engine,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def get_benchmarks(limit: int = 50) -> List[dict]:
    """Get benchmark history (sync)."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = _get_sync_engine()
    with Session(engine) as session:
        query = select(Benchmark).order_by(Benchmark.created_at.desc()).limit(limit)
        result = session.execute(query)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "model_name": r.model_name,
                "engine": r.engine,
                "latency_ms": r.latency_ms,
                "memory_mb": r.memory_mb,
                "device": r.device,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
