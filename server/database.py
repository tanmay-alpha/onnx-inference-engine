"""SQLite Database Module for Crucible Server.

Provides thread-safe storage for:
  - Registered ONNX models metadata
  - Inference execution logs
  - Fraud detection transaction history
  - Benchmark performance logs
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_db_path() -> Path:
    """Resolve the SQLite database filepath."""
    env_path = os.environ.get("CRUCIBLE_DB_PATH")
    if env_path:
        p = Path(env_path)
    else:
        p = Path(os.environ.get("CRUCIBLE_MODEL_DIR", "/tmp/models")).parent / "crucible.db"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        p = Path("./crucible.db")
    return p


def get_connection() -> sqlite3.Connection:
    """Open a thread-safe connection to the SQLite database with WAL mode."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Execute table and index creation SQL statements safely."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            input_shape TEXT NOT NULL,
            operators_json TEXT NOT NULL,
            all_supported INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS inference_logs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            input_shape TEXT NOT NULL,
            output_shape TEXT NOT NULL,
            inference_time_ms REAL NOT NULL,
            engine TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS fraud_transactions (
            id TEXT PRIMARY KEY,
            tx_type TEXT NOT NULL,
            amount REAL NOT NULL,
            orig_before REAL NOT NULL,
            orig_after REAL NOT NULL,
            dest_before REAL NOT NULL,
            dest_after REAL NOT NULL,
            probability REAL NOT NULL,
            verdict TEXT NOT NULL,
            execution_mode TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS benchmarks (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            engine TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            memory_mb REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);
        CREATE INDEX IF NOT EXISTS idx_inference_logs_model ON inference_logs(model_id);
        CREATE INDEX IF NOT EXISTS idx_fraud_created_at ON fraud_transactions(created_at);
        CREATE INDEX IF NOT EXISTS idx_benchmarks_created_at ON benchmarks(created_at);
    """)


def init_db() -> None:
    """Initialize database tables and indexes."""
    with get_connection() as _conn:
        pass


# ---------------------------------------------------------------------------
# Models CRUD
# ---------------------------------------------------------------------------
def save_model(
    model_id: str,
    name: str,
    file_path: Path | str,
    file_size_bytes: int,
    input_shape: List[int],
    operators: List[str],
    all_supported: bool,
) -> Dict[str, Any]:
    """Store a registered model record."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models
            (id, name, file_path, file_size_bytes, input_shape, operators_json, all_supported)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                name,
                str(file_path),
                file_size_bytes,
                json.dumps(input_shape),
                json.dumps(operators),
                1 if all_supported else 0,
            ),
        )
    return get_model(model_id)  # type: ignore[return-value]


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve model by UUID."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["input_shape"] = json.loads(d["input_shape"])
        d["operators"] = json.loads(d["operators_json"])
        d["all_supported"] = bool(d["all_supported"])
        del d["operators_json"]
        return d


def list_models() -> List[Dict[str, Any]]:
    """List all registered models ordered by newest first."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["input_shape"] = json.loads(d["input_shape"])
            d["operators"] = json.loads(d["operators_json"])
            d["all_supported"] = bool(d["all_supported"])
            del d["operators_json"]
            result.append(d)
        return result


def delete_model(model_id: str) -> bool:
    """Delete a model record from the database."""
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Inference Logging
# ---------------------------------------------------------------------------
def log_inference(
    model_id: str,
    input_shape: List[int],
    output_shape: List[int],
    inference_time_ms: float,
    engine: str,
) -> Dict[str, Any]:
    """Record an inference execution."""
    log_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO inference_logs
            (id, model_id, input_shape, output_shape, inference_time_ms, engine)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                model_id,
                json.dumps(input_shape),
                json.dumps(output_shape),
                inference_time_ms,
                engine,
            ),
        )
        row = conn.execute("SELECT * FROM inference_logs WHERE id = ?", (log_id,)).fetchone()
        d = dict(row)
        d["input_shape"] = json.loads(d["input_shape"])
        d["output_shape"] = json.loads(d["output_shape"])
        return d


def get_inference_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent inference logs."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM inference_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["input_shape"] = json.loads(d["input_shape"])
            d["output_shape"] = json.loads(d["output_shape"])
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# Fraud Transactions
# ---------------------------------------------------------------------------
def log_fraud_tx(
    tx_type: str,
    amount: float,
    orig_before: float,
    orig_after: float,
    dest_before: float,
    dest_after: float,
    probability: float,
    verdict: str,
    execution_mode: str = "wasm",
    latency_ms: float = 0.0,
) -> Dict[str, Any]:
    """Record a fraud check transaction."""
    tx_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO fraud_transactions
            (id, tx_type, amount, orig_before, orig_after, dest_before, dest_after,
             probability, verdict, execution_mode, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tx_id,
                tx_type,
                amount,
                orig_before,
                orig_after,
                dest_before,
                dest_after,
                probability,
                verdict,
                execution_mode,
                latency_ms,
            ),
        )
        row = conn.execute("SELECT * FROM fraud_transactions WHERE id = ?", (tx_id,)).fetchone()
        return dict(row)


def get_fraud_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent fraud check transaction history."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM fraud_transactions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def log_benchmark(
    model_name: str, engine: str, latency_ms: float, memory_mb: float = 0.0
) -> Dict[str, Any]:
    """Record a benchmark run."""
    b_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO benchmarks (id, model_name, engine, latency_ms, memory_mb)
            VALUES (?, ?, ?, ?, ?)
            """,
            (b_id, model_name, engine, latency_ms, memory_mb),
        )
        row = conn.execute("SELECT * FROM benchmarks WHERE id = ?", (b_id,)).fetchone()
        return dict(row)


def get_benchmarks(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent benchmark records."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
