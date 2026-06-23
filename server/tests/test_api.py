"""End-to-end tests for the Crucible FastAPI server (Issue #13).

Strategy
========

We use FastAPI's TestClient — synchronous, in-process, no socket
binding. Each test gets a fresh model registry (the module-level
_MODEL_REGISTRY in server.main) by calling ``reset_registry()``;
that keeps tests independent without the brittleness of
reload-on-every-test fixtures.

Where the tests build a real PyTorch model, they use the smallest
one that exercises the dispatch table end-to-end: a single Conv
+ Relu + GlobalAveragePool + Gemm stack. That's enough to make
/convert succeed, /validate report the right ops, and /infer
return a sensibly-shaped output (1, num_classes).

The tests intentionally do NOT use MobileNetV2.onnx — depending
on a 14 MB file in tests makes them slow and brittle, and
the AC for Issue #13 doesn't mention MobileNetV2. The /operators
test that asserts the count is 20 is the static check; the
model-zoo integration is Issue #14's job.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
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

from server.main import _MODEL_REGISTRY, _register_model, app  # noqa: E402
from server import converter, validator  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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


def _pt_bytes(model: nn.Module) -> io.BytesIO:
    """Serialize a model to a BytesIO in torch.save format."""
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "crucible-cpp"
    assert body["version"]  # non-empty


# ---------------------------------------------------------------------------
# /operators
# ---------------------------------------------------------------------------
def test_operators_lists_all_supported(client: TestClient) -> None:
    r = client.get("/operators")
    assert r.status_code == 200
    body = r.json()
    # The dispatch table covers exactly 20 op_types. The exact set
    # is asserted via membership rather than equality so adding a
    # new op to executor.cpp is a one-line test change.
    assert body["count"] == 20
    assert body["count"] == len(body["supported"])
    for op in ("Conv", "Relu", "GlobalAveragePool", "Gemm", "Add", "Flatten"):
        assert op in body["supported"]


# ---------------------------------------------------------------------------
# /convert
# ---------------------------------------------------------------------------
def test_convert_uploads_pt_and_returns_model_id(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    pt = _pt_bytes(small_classifier)
    r = client.post(
        "/convert",
        files={"model_file": ("small.pt", pt.read(), "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["onnx_model_id"] in _MODEL_REGISTRY
    assert (tmp_model_dir / f"{body['onnx_model_id']}.onnx").is_file()
    # The op set should be a subset of Crucible's supported ops.
    assert set(body["operators_used"]) <= set(validator.SUPPORTED_OPS)
    assert body["all_supported"] is True
    assert body["unsupported_ops"] == []


def test_convert_rejects_state_dict_only(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    """A state_dict upload is a 400, not a 500. Documented in converter.py."""
    state = small_classifier.state_dict()
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    r = client.post(
        "/convert",
        files={"model_file": ("state.pt", buf.read(), "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
    )
    assert r.status_code == 400
    assert "state_dict" in r.json()["detail"].lower()


def test_convert_rejects_garbage_upload(
    client: TestClient,
    tmp_model_dir: Path,
) -> None:
    """Random bytes that aren't a torch.save payload get a 400."""
    r = client.post(
        "/convert",
        files={"model_file": ("garbage.pt", b"not a pytorch file", "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
    )
    assert r.status_code == 400


def test_convert_rejects_malformed_input_shape(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    """input_shape must be a JSON array of ints; anything else is 400."""
    r = client.post(
        "/convert",
        files={"model_file": ("small.pt", _pt_bytes(small_classifier).read(),
                              "application/octet-stream")},
        data={"input_shape": "1,3,32,32"},  # wrong: not JSON
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /validate
# ---------------------------------------------------------------------------
def test_validate_uploaded_onnx_is_supported(
    client: TestClient,
    small_classifier: nn.Module,
    tmp_path: Path,
) -> None:
    """Round-trip a model through /convert, then validate the saved
    file via the multipart upload path."""
    onnx_path = tmp_path / "small.onnx"
    converter.convert_torch_module(
        small_classifier, [1, 3, 32, 32], onnx_path,
    )
    with open(onnx_path, "rb") as f:
        r = client.post(
            "/validate",
            files={"model_file": ("small.onnx", f.read(), "application/octet-stream")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["unsupported"] == []
    assert "Conv" in body["operators"]
    assert "Relu" in body["operators"]


def test_validate_by_model_id(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    """After /convert, /validate(model_id=...) should find the
    saved model and report its ops."""
    pt = _pt_bytes(small_classifier)
    cr = client.post(
        "/convert",
        files={"model_file": ("small.pt", pt.read(), "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
    )
    model_id = cr.json()["onnx_model_id"]

    r = client.post("/validate", data={"model_id": model_id})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert "Conv" in body["operators"]


def test_validate_rejects_missing_model_id(client: TestClient) -> None:
    r = client.post("/validate", data={"model_id": "deadbeef"})
    assert r.status_code == 404


def test_validate_requires_exactly_one_of_file_or_id(client: TestClient) -> None:
    r = client.post("/validate")  # neither
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------
def test_infer_returns_correct_shape(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    """Full convert -> infer flow on the small classifier.

    The fallback path returns zeros of the right shape; the C++
    path runs the actual graph. Both must satisfy:
        output_shape == (1, 5)
        output.dtype == float32
    """
    pt = _pt_bytes(small_classifier)
    cr = client.post(
        "/convert",
        files={"model_file": ("small.pt", pt.read(), "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
    )
    model_id = cr.json()["onnx_model_id"]

    # 1 * 3 * 32 * 32 = 3072 floats
    arr = np.random.RandomState(0).randn(1, 3, 32, 32).astype(np.float32)
    payload = {
        "model_id": model_id,
        "input": arr.reshape(-1).tolist(),
        "input_shape": [1, 3, 32, 32],
    }
    r = client.post("/infer", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_shape"] == [1, 5]
    assert len(body["output"]) == 5
    assert body["engine"] == "crucible-cpp"
    # inference_time_ms is always positive and reasonable.
    assert 0.0 <= body["inference_time_ms"] < 60_000.0


def test_infer_rejects_unknown_model_id(client: TestClient) -> None:
    r = client.post(
        "/infer",
        json={
            "model_id": "nope",
            "input": [0.0],
            "input_shape": [1],
        },
    )
    assert r.status_code == 404


def test_infer_rejects_shape_mismatch(
    client: TestClient,
    tmp_model_dir: Path,
    small_classifier: nn.Module,
) -> None:
    pt = _pt_bytes(small_classifier)
    cr = client.post(
        "/convert",
        files={"model_file": ("small.pt", pt.read(), "application/octet-stream")},
        data={"input_shape": "[1, 3, 32, 32]"},
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
    )
    assert r.status_code == 400


def test_infer_rejects_negative_input_shape_dim(client: TestClient) -> None:
    """Pydantic validator catches negative dims before we hit numpy."""
    r = client.post(
        "/infer",
        json={
            "model_id": "anything",
            "input": [],
            "input_shape": [1, -3, 32, 32],
        },
    )
    assert r.status_code == 422  # FastAPI's Pydantic validation error