"""Shared fixtures for the server test suite.

Native-engine tests use the ``native`` fixture: it SKIPS when crucible_py is
not importable, and FAILS instead when CRUCIBLE_REQUIRE_NATIVE=1 (CI with a
built engine must not silently skip them).
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from server import database, engine, ratelimit  # noqa: E402
from server.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> Path:
    """Fresh SQLite DB + model dir per test; no operator key unless a test sets one."""
    model_dir = tmp_path / "models"
    monkeypatch.setenv("CRUCIBLE_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("CRUCIBLE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CRUCIBLE_API_KEY", "")
    database.reset_engine()
    ratelimit.reset_all()
    database.init_db()
    yield model_dir
    database.reset_engine()


@pytest.fixture
def model_dir(_isolated_env) -> Path:
    return _isolated_env


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def native():
    if engine.crucible_py is None:
        if os.environ.get("CRUCIBLE_REQUIRE_NATIVE") == "1":
            pytest.fail("CRUCIBLE_REQUIRE_NATIVE=1 but crucible_py is not importable "
                        "(put the build's python/ dir on PYTHONPATH)")
        pytest.skip("crucible_py (native engine) not built/importable")
    return engine.crucible_py


@pytest.fixture
def make_user(client) -> Callable[..., Dict[str, str]]:
    """Register + log in a user; returns Bearer headers."""
    def _make(email: Optional[str] = None, password: str = "correct-horse-1") -> Dict[str, str]:
        email = email or f"user-{uuid.uuid4().hex[:8]}@example.com"
        r = client.post("/auth/register", json={"email": email, "password": password})
        assert r.status_code == 201, r.text
        r = client.post("/auth/login", json={"email": email, "password": password})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _make
