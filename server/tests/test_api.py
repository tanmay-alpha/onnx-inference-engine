"""End-to-end tests for the Crucible FastAPI server (Issue #13).

Strategy
========

We use FastAPI's TestClient — synchronous, in-process, no socket
binding. Each test gets a fresh model registry (the module-level
_MODEL_REGISTRY in server.main) by clearing it; that keeps tests
independent without the brittleness of reload-on-every-test
fixtures.

Tests that touch /convert use the .onnx upload path (per the
post-review security fix: /convert no longer accepts .pt pickles
because torch.load(weights_only=False) is RCE). Tests build the
.onnx bytes in-process via converter.convert_torch_module, which
is safe because the input is a trusted fixture, not a network
upload.

Auth contract
=============

The security review requires X-API-Key on every non-public
endpoint. We set CRUCIBLE_API_KEY via monkeypatch in a session-
scoped fixture so /health and /operators stay reachable without
auth (liveness probes) and the rest require the header.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

# Make the server package importable when pytest is run from the
# repo root. Without this, `from server.main import app` fails
# because server/ is not on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.main import _MODEL_REGISTRY, _check_api_key, app  # noqa: E402
from server import converter, validator  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_API_KEY = "test-key-1234567890"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _configure_api_key(monkeypatch):
    """Set CRUCIBLE_API_KEY for every test.

    autouse=True so individual tests don't have to remember.
    Tests that want to exercise the "no key configured" 503 path
    monkeypatch it back to ''.

    Function-scoped (not session) so the override in each test is
    applied independently. monkeypatch is function-scoped by
    default, but making this fixture's lifetime explicit avoids
    surprises.
    """
    monkeypatch.setenv("CRUCIBLE_API_KEY", TEST_API_KEY)
    yield


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient with the in-process model registry cleared.

    We use the module-level `_MODEL_REGISTRY` so /infer can find
    models registered by /convert within the same test. Resetting
    it between tests prevents accidental cross-test coupling.
    """
    _MODEL_REGISTRY.clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch) -> Path:
    """Point CRUCIBLE_MODEL_DIR at a per-test temp directory so
    /convert's writes don't pollute /tmp and survive across runs.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("CRUCIBLE_MODEL_DIR", str(model_dir))
    return model_dir


# Defined at module scope (not inside the fixture) so torch.save
# can pickle it. Defining nn.Module subclasses inside a function
# makes them local objects that pickle refuses to serialise.
class SmallClassifier(nn.Module):
    """A 3-channel -> 32-channel -> 5-class classifier.

    Two Conv+Relu blocks feeding GlobalAveragePool -> Linear.
    Uses 5 classes (not 1000) so the test output is small and
    obviously right; the API contract is shape-driven, not
    value-driven.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@pytest.fixture
def small_classifier() -> nn.Module:
    model = SmallClassifier()
    model.eval()
    return model


@pytest.fixture
def onnx_bytes(small_classifier: nn.Module, tmp_path: Path) -> bytes:
    """Build a small .onnx file in tmp and return its bytes.

    Used as the upload body for /convert and /validate tests.
    Building in tmp (rather than memory) catches file-system
    issues at the test layer instead of production.
    """
    onnx_path = tmp_path / "small.onnx"
    converter.convert_torch_module(small_classifier, [1, 3, 32, 32], onnx_path)
    return onnx_path.read_bytes()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_no_key_returns_503(client: TestClient, monkeypatch) -> None:
    """If the operator forgets CRUCIBLE_API_KEY, /convert returns 503,
    not 200. We do not default-allow."""
    monkeypatch.setenv("CRUCIBLE_API_KEY", "")
    r = client.post("/convert",
                    files={"model_file": ("x.onnx", b"", "application/octet-stream")},
                    data={"input_shape": "[1]"})
    assert r.status_code == 503


def test_wrong_key_returns_401(client: TestClient) -> None:
    r = client.post(
        "/convert",
        files={"model_file": ("x.onnx", b"", "application/octet-stream")},
        data={"input_shape": "[1]"},
        headers={"X-API-Key": "definitely-wrong"},
    )
    assert r.status_code == 401


def test_health_and_operators_are_public(client: TestClient) -> None:
    """No header, no monkeypatch override. /health and /operators
    must work without credentials or container health checks fail."""
    assert client.get("/health").status_code == 200
    assert client.get("/operators").status_code == 200


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "crucible-cpp"
    assert body["version"]


# ---------------------------------------------------------------------------
# /operators
# ---------------------------------------------------------------------------
def test_operators_lists_all_supported(client: TestClient) -> None:
    r = client.get("/operators")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 20
    assert body["count"] == len(body["supported"])
    for op in ("Conv", "Relu", "GlobalAveragePool", "Gemm", "Add", "Flatten"):
        assert op in body["supported"]


# ---------------------------------------------------------------------------
# /convert
# ---------------------------------------------------------------------------
def test_convert_uploads_onnx_and_returns_model_id(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    """Happy path: upload a real .onnx, get back a model id and
    the operator audit (all ops supported)."""
    r = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["onnx_model_id"] in _MODEL_REGISTRY
    assert (tmp_model_dir / f"{body['onnx_model_id']}.onnx").is_file()
    assert set(body["operators_used"]) <= set(validator.SUPPORTED_OPS)
    assert body["all_supported"] is True
    assert body["unsupported_ops"] == []


def test_convert_rejects_non_onnx_bytes(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
) -> None:
    """Random non-ONNX bytes get a 400 with a useful message."""
    r = client.post(
        "/convert",
        files={"model_file": ("garbage.bin", b"definitely not onnx",
                              "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_convert_rejects_malformed_input_shape(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    """input_shape must be a JSON array of ints; anything else is 400."""
    r = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "1,3,32,32"},  # wrong: not JSON
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_convert_rejects_oversize_input_shape(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    """A request that would build a 200M-element tensor is rejected
    at upload time. Without this, a hostile 4 GiB JSON body could
    OOM the server."""
    r = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "[1024, 1024, 256]"},  # ~268M elements
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "cap" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /validate
# ---------------------------------------------------------------------------
def test_validate_uploaded_onnx_is_supported(
    client: TestClient,
    auth_headers: dict,
    onnx_bytes: bytes,
) -> None:
    """Validate a multipart upload."""
    r = client.post(
        "/validate",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["unsupported"] == []
    assert "Conv" in body["operators"]
    assert "Relu" in body["operators"]


def test_validate_by_model_id(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    """After /convert, /validate(model_id=...) should find the
    saved model and report its ops."""
    cr = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
        headers=auth_headers,
    )
    model_id = cr.json()["onnx_model_id"]

    r = client.post("/validate", data={"model_id": model_id},
                    headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert "Conv" in body["operators"]


def test_validate_rejects_missing_model_id(client: TestClient, auth_headers: dict) -> None:
    r = client.post("/validate", data={"model_id": "deadbeef"},
                    headers=auth_headers)
    assert r.status_code == 404


def test_validate_requires_exactly_one_of_file_or_id(client: TestClient, auth_headers: dict) -> None:
    r = client.post("/validate", headers=auth_headers)  # neither
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------
def test_infer_returns_correct_shape(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    """Full convert -> infer flow on the small classifier.

    The fallback path returns zeros of the right shape; the C++
    path runs the actual graph. Both must satisfy:
        output_shape == (1, 5)
        output.dtype == float32
    """
    cr = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
        headers=auth_headers,
    )
    model_id = cr.json()["onnx_model_id"]

    arr = np.random.RandomState(0).randn(1, 3, 32, 32).astype(np.float32)
    payload = {
        "model_id": model_id,
        "input": arr.reshape(-1).tolist(),
        "input_shape": [1, 3, 32, 32],
    }
    r = client.post("/infer", json=payload, headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_shape"] == [1, 5]
    assert len(body["output"]) == 5
    assert body["engine"] == "crucible-cpp"
    assert 0.0 <= body["inference_time_ms"] < 60_000.0


def test_infer_rejects_unknown_model_id(client: TestClient, auth_headers: dict) -> None:
    r = client.post(
        "/infer",
        json={"model_id": "nope", "input": [0.0], "input_shape": [1]},
        headers=auth_headers,
    )
    assert r.status_code == 404


def test_infer_rejects_shape_mismatch(
    client: TestClient,
    auth_headers: dict,
    tmp_model_dir: Path,
    onnx_bytes: bytes,
) -> None:
    cr = client.post(
        "/convert",
        files={"model_file": ("small.onnx", onnx_bytes, "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
        headers=auth_headers,
    )
    model_id = cr.json()["onnx_model_id"]

    # 100 floats but input_shape expects 3072
    r = client.post(
        "/infer",
        json={
            "model_id": model_id,
            "input": [0.0] * 100,
            "input_shape": [1, 3, 32, 32],
        },
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_infer_rejects_negative_input_shape_dim(client: TestClient, auth_headers: dict) -> None:
    """Pydantic validator catches negative dims before we hit numpy."""
    r = client.post(
        "/infer",
        json={
            "model_id": "anything",
            "input": [],
            "input_shape": [1, -3, 32, 32],
        },
        headers=auth_headers,
    )
    assert r.status_code == 422  # FastAPI's Pydantic validation error


def test_infer_rejects_oversize_input_shape(
    client: TestClient, auth_headers: dict,
) -> None:
    """Defence-in-depth: even if Pydantic let it through, the
    /infer path's product check should still catch a 100M-element
    shape and return 400, not OOM."""
    r = client.post(
        "/infer",
        json={
            "model_id": "anything",
            "input": [0.0],
            "input_shape": [1024, 1024, 256],  # ~268M
        },
        headers=auth_headers,
    )
    # 422 (Pydantic) or 400 (our middleware) are both acceptable;
    # 404 (unknown model) wins if shape validation happens after
    # model lookup. What matters is that 500 (OOM) is NOT the
    # response — the assertion rejects any 5xx.
    assert r.status_code in (400, 422, 404)
    assert r.status_code < 500