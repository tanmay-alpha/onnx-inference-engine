"""API tests for the Crucible server.

Native-engine tests (``native`` fixture) compare /infer against onnxruntime;
they skip without crucible_py unless CRUCIBLE_REQUIRE_NATIVE=1.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from server import database, engine, main
from server.tests.helpers import (
    conv_bytes,
    files_in,
    mlp_bytes,
    ort_run,
    unsupported_bytes,
    upload,
)

RTOL, ATOL = 1e-4, 1e-5


def _infer(client, headers, model_id, x: np.ndarray):
    return client.post("/infer", headers=headers, json={
        "model_id": model_id, "input": x.reshape(-1).tolist(), "input_shape": list(x.shape),
    })


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
def test_health_reports_engine_availability(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["native_available"] is engine.native_available()
    assert body["engine"] == ("crucible-native" if engine.native_available() else "unavailable")


def test_ready_requires_native_engine(client) -> None:
    r = client.get("/ready")
    assert r.json()["database"] is True
    assert r.status_code == (200 if engine.native_available() else 503)


def test_operators_public_and_complete(client) -> None:
    body = client.get("/operators").json()
    assert body["count"] == len(body["supported"]) == 20
    assert {"Conv", "Gemm", "Relu", "Softmax", "GlobalAveragePool"} <= set(body["supported"])


def test_static_operator_mirror_matches_native(native) -> None:
    from server.validator import SUPPORTED_OPS
    assert sorted(native.supported_ops()) == sorted(SUPPORTED_OPS)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
PROTECTED = [
    ("post", "/convert"), ("post", "/validate"), ("post", "/infer"), ("post", "/inference/batch"),
    ("get", "/models"), ("get", "/models/abc"), ("delete", "/models/abc"), ("get", "/inference/logs"),
    ("post", "/fraud/log"), ("get", "/fraud/history"), ("get", "/analytics/inference"),
    ("get", "/analytics/fraud"), ("get", "/analytics/models"), ("post", "/auth/api-key"),
    ("get", "/auth/api-keys"), ("delete", "/auth/api-key/abc"), ("get", "/auth/me"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_routes_reject_missing_credentials(client, method, path) -> None:
    assert getattr(client, method)(path).status_code == 401


def test_invalid_credentials_rejected(client) -> None:
    assert client.get("/models", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    assert client.get("/models", headers={"X-API-Key": "cr_nope"}).status_code == 401


def test_unset_operator_key_does_not_block_users(client, make_user, monkeypatch) -> None:
    monkeypatch.delenv("CRUCIBLE_API_KEY", raising=False)
    monkeypatch.setattr(main.settings, "CRUCIBLE_API_KEY", "")
    headers = make_user()
    assert client.get("/models", headers=headers).status_code == 200
    assert client.get("/models", headers={"X-API-Key": ""}).status_code == 401


def test_register_rejects_duplicate_email_case_insensitively(client, make_user) -> None:
    make_user("Dup@Example.com")
    r = client.post("/auth/register", json={"email": "dup@example.com", "password": "another-pass-1"})
    assert r.status_code == 409


def test_user_api_key_lifecycle(client, make_user) -> None:
    jwt_headers = make_user()
    created = client.post("/auth/api-key", headers=jwt_headers, json={"name": "ci"}).json()
    key = created["full_key"]
    assert created["key_prefix"] == key[:12]
    key_headers = {"X-API-Key": key}
    upload(client, key_headers, mlp_bytes(), "via-key.onnx")
    models = client.get("/models", headers=jwt_headers).json()
    assert [m["name"] for m in models["models"]] == ["via-key.onnx"]  # key acts as its owner
    listed = client.get("/auth/api-keys", headers=jwt_headers).json()
    assert [k["key_prefix"] for k in listed] == [key[:12]]
    assert client.delete(f"/auth/api-key/{created['id']}", headers=jwt_headers).status_code == 200
    assert client.get("/models", headers=key_headers).status_code == 401


def test_operator_key_sees_all_users(client, make_user, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_API_KEY", "operator-key-0123456789")
    a = make_user()
    model_id = upload(client, a, mlp_bytes())["onnx_model_id"]
    op = {"X-API-Key": "operator-key-0123456789"}
    assert model_id in [m["id"] for m in client.get("/models", headers=op).json()["models"]]
    assert client.get(f"/models/{model_id}", headers=op).status_code == 200
    assert client.get("/auth/api-keys", headers=op).status_code == 403  # not a user account


# ---------------------------------------------------------------------------
# /convert + /models
# ---------------------------------------------------------------------------
def test_convert_reports_metadata_and_stores_uuid_file(client, make_user, model_dir) -> None:
    headers = make_user()
    body = upload(client, headers, mlp_bytes(), "../../evil name.onnx")
    assert body["name"] == "evil name.onnx"
    assert body["all_supported"] is True
    assert body["unsupported_ops"] == [] and body["unsupported_reasons"] == []
    assert body["operators_used"] == ["Gemm", "Relu", "Softmax"]
    assert body["opset"] == 13
    assert body["inputs"] == [{"name": "X", "shape": [-1, 4], "dtype": "float32"}]
    assert body["outputs"][0]["name"] == "Y"
    assert files_in(model_dir) == [model_dir / f"{body['onnx_model_id']}.onnx"]

    item = client.get(f"/models/{body['onnx_model_id']}", headers=headers).json()
    assert item["inputs"] == body["inputs"] and item["size_bytes"] == len(mlp_bytes())
    assert set(item) >= {"id", "name", "size_bytes", "created_at", "opset", "operators_used",
                         "all_supported", "unsupported_reasons", "inputs", "outputs"}


def test_convert_accepts_legacy_input_shape_field(client, make_user) -> None:
    body = upload(client, make_user(), mlp_bytes(), input_shape="[1, 4]")
    assert body["inputs"][0]["shape"] == [-1, 4]


def test_convert_rejects_malformed_onnx_and_leaves_no_file(client, make_user, model_dir) -> None:
    r = client.post("/convert", headers=make_user(),
                    files={"model_file": ("x.onnx", b"definitely not onnx", "application/octet-stream")})
    assert r.status_code == 400
    assert files_in(model_dir) == []


def test_convert_stores_unsupported_model_with_reasons(client, make_user) -> None:
    headers = make_user()
    body = upload(client, headers, unsupported_bytes())
    assert body["all_supported"] is False
    assert body["unsupported_ops"] == ["Cos"]
    assert body["unsupported_reasons"]
    listed = client.get("/models", headers=headers).json()
    assert listed["count"] == 1 and listed["models"][0]["all_supported"] is False


def test_validate_file_and_model_id(client, make_user, model_dir) -> None:
    headers = make_user()
    r = client.post("/validate", headers=headers,
                    files={"model_file": ("u.onnx", unsupported_bytes(), "application/octet-stream")})
    assert r.status_code == 200
    assert r.json()["valid"] is False and r.json()["unsupported"] == ["Cos"]
    assert files_in(model_dir) == []  # validation does not store
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    r = client.post("/validate", headers=headers, data={"model_id": model_id})
    assert r.status_code == 200 and r.json()["valid"] is True
    assert client.post("/validate", headers=headers).status_code == 400


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------
def test_ownership_isolation(client, make_user, model_dir) -> None:
    a, b = make_user(), make_user()
    model_id = upload(client, a, mlp_bytes())["onnx_model_id"]

    assert client.get("/models", headers=b).json()["count"] == 0
    assert client.get(f"/models/{model_id}", headers=b).status_code == 404
    assert client.delete(f"/models/{model_id}", headers=b).status_code == 404
    assert client.post("/validate", headers=b, data={"model_id": model_id}).status_code == 404
    assert _infer(client, b, model_id, np.zeros((1, 4), np.float32)).status_code == 404
    assert (model_dir / f"{model_id}.onnx").is_file()
    assert client.get(f"/models/{model_id}", headers=a).status_code == 200


def test_unauthenticated_delete_is_401_and_keeps_model(client, make_user, model_dir) -> None:
    a = make_user()
    model_id = upload(client, a, mlp_bytes())["onnx_model_id"]
    assert client.delete(f"/models/{model_id}").status_code == 401
    assert (model_dir / f"{model_id}.onnx").is_file()


def test_owner_delete_removes_file_and_row(client, make_user, model_dir) -> None:
    a = make_user()
    model_id = upload(client, a, mlp_bytes())["onnx_model_id"]
    r = client.delete(f"/models/{model_id}", headers=a)
    assert r.status_code == 200 and r.json() == {"status": "deleted", "model_id": model_id}
    assert files_in(model_dir) == []
    assert client.get(f"/models/{model_id}", headers=a).status_code == 404


def test_delete_never_unlinks_outside_model_dir(client, make_user, tmp_path) -> None:
    a = make_user()
    owner_id = client.get("/auth/me", headers=a).json()["id"]
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"precious")
    database.save_model(
        model_id="m-outside", name="x", file_path=str(outside), size_bytes=8, owner_id=owner_id,
        opset=13, operators_used=[], unsupported_ops=[], unsupported_reasons=[], all_supported=True,
        inputs=[], outputs=[], native_checked=False,
    )
    assert client.delete("/models/m-outside", headers=a).status_code == 200
    assert outside.read_bytes() == b"precious"


# ---------------------------------------------------------------------------
# Body limits
# ---------------------------------------------------------------------------
def test_upload_over_middleware_limit_is_413(client, make_user, monkeypatch, model_dir) -> None:
    headers = make_user()
    monkeypatch.setattr(main.settings, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main, "MULTIPART_OVERHEAD_BYTES", 0)
    r = client.post("/convert", headers=headers,
                    files={"model_file": ("big.onnx", b"\0" * 5000, "application/octet-stream")})
    assert r.status_code == 413
    assert files_in(model_dir) == []


def test_upload_over_file_limit_is_413(client, make_user, monkeypatch, model_dir) -> None:
    headers = make_user()
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/convert", headers=headers,
                    files={"model_file": ("m.onnx", mlp_bytes(), "application/octet-stream")})
    assert r.status_code == 413
    assert files_in(model_dir) == []


def test_json_body_limit_with_and_without_content_length(client, make_user, monkeypatch) -> None:
    headers = make_user()
    monkeypatch.setattr(main.settings, "MAX_REQUEST_BODY_BYTES", 1000)
    payload = b'{"model_id": "x", "input": [' + b"0.0," * 1000 + b'0.0], "input_shape": [1001]}'
    r = client.post("/infer", headers={**headers, "Content-Type": "application/json"}, content=payload)
    assert r.status_code == 413
    chunked = client.post("/infer", headers={**headers, "Content-Type": "application/json"},
                          content=iter([payload[:600], payload[600:]]))
    assert chunked.status_code == 413


def test_infer_element_cap_is_413(client, make_user, monkeypatch) -> None:
    monkeypatch.setattr(main, "MAX_INPUT_ELEMENTS", 10)
    r = _infer(client, make_user(), "anything", np.zeros((3, 4), np.float32))
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# /infer validation (no native engine needed)
# ---------------------------------------------------------------------------
def test_infer_value_count_mismatch_is_422(client, make_user) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    r = client.post("/infer", headers=headers,
                    json={"model_id": model_id, "input": [0.0] * 5, "input_shape": [2, 4]})
    assert r.status_code == 422


def test_infer_shape_mismatch_vs_model_is_422(client, make_user) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    assert _infer(client, headers, model_id, np.zeros((2, 5), np.float32)).status_code == 422
    assert _infer(client, headers, model_id, np.zeros((1, 2, 4), np.float32)).status_code == 422
    conv_id = upload(client, headers, conv_bytes())["onnx_model_id"]
    assert _infer(client, headers, conv_id, np.zeros((2, 3, 8, 8), np.float32)).status_code == 422


def test_infer_bad_shape_values_are_422(client, make_user) -> None:
    r = client.post("/infer", headers=make_user(),
                    json={"model_id": "x", "input": [], "input_shape": [1, -3]})
    assert r.status_code == 422


def test_infer_unknown_model_is_404(client, make_user) -> None:
    assert _infer(client, make_user(), "nope", np.zeros((1, 4), np.float32)).status_code == 404


def test_infer_unsupported_model_is_422(client, make_user) -> None:
    headers = make_user()
    model_id = upload(client, headers, unsupported_bytes())["onnx_model_id"]
    r = _infer(client, headers, model_id, np.zeros((1, 4), np.float32))
    assert r.status_code == 422 and "Cos" in r.json()["detail"]


def test_infer_missing_model_file_is_410(client, make_user, model_dir) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    (model_dir / f"{model_id}.onnx").unlink()
    assert _infer(client, headers, model_id, np.zeros((1, 4), np.float32)).status_code == 410


def test_infer_and_batch_503_without_native_engine(client, make_user, monkeypatch) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    monkeypatch.setattr(engine, "crucible_py", None)
    assert client.get("/health").json()["engine"] == "unavailable"
    r = _infer(client, headers, model_id, np.ones((1, 4), np.float32))
    assert r.status_code == 503
    assert "output" not in r.json()
    batch = [{"model_id": model_id, "input": [1.0] * 4, "input_shape": [1, 4]}]
    assert client.post("/inference/batch", headers=headers, json=batch).status_code == 503
    assert client.get("/inference/logs", headers=headers).json()["count"] == 0


def test_infer_timeout_is_504(client, make_user, monkeypatch) -> None:
    """A run that outlives INFERENCE_TIMEOUT_SEC yields 504 (engine stubbed to block)."""
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]

    class _SlowModule:
        @staticmethod
        def load_model(path):
            return object()

        @staticmethod
        def run(model, inputs):
            time.sleep(0.5)
            return np.zeros((1, 3), np.float32)

    monkeypatch.setattr(engine, "crucible_py", _SlowModule)
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SEC", 0.05)
    r = _infer(client, headers, model_id, np.ones((1, 4), np.float32))
    assert r.status_code == 504
    logs = client.get("/inference/logs", headers=headers).json()["logs"]
    assert [l["status"] for l in logs] == ["timeout"]
    time.sleep(0.6)  # let the uncancellable background run release its slot


# ---------------------------------------------------------------------------
# /infer on the native engine vs onnxruntime
# ---------------------------------------------------------------------------
def test_infer_mlp_matches_onnxruntime_jwt(client, make_user, native) -> None:
    headers = make_user()
    data = mlp_bytes()
    model_id = upload(client, headers, data)["onnx_model_id"]
    x = np.random.RandomState(42).randn(3, 4).astype(np.float32)
    r = _infer(client, headers, model_id, x)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == body["engine"] == "crucible-native"
    assert body["model_id"] == model_id
    assert body["output_shape"] == [3, 3]
    assert body["inference_time_ms"] > 0
    got = np.asarray(body["output"], np.float32).reshape(body["output_shape"])
    np.testing.assert_allclose(got, ort_run(data, x), rtol=RTOL, atol=ATOL)

    logs = client.get("/inference/logs", headers=headers).json()
    assert logs["count"] == 1
    log = logs["logs"][0]
    assert log["backend"] == "crucible-native" and log["status"] == "success"
    assert log["input_shape"] == [3, 4] and log["output_shape"] == [3, 3]
    assert log["model_id"] == model_id


def test_infer_conv_matches_onnxruntime_user_api_key(client, make_user, native) -> None:
    jwt_headers = make_user()
    key = client.post("/auth/api-key", headers=jwt_headers, json={"name": "k"}).json()["full_key"]
    headers = {"X-API-Key": key}
    data = conv_bytes()
    model_id = upload(client, headers, data)["onnx_model_id"]
    x = np.random.RandomState(7).randn(1, 3, 8, 8).astype(np.float32)
    r = _infer(client, headers, model_id, x)
    assert r.status_code == 200, r.text
    got = np.asarray(r.json()["output"], np.float32).reshape(r.json()["output_shape"])
    np.testing.assert_allclose(got, ort_run(data, x), rtol=RTOL, atol=ATOL)


def test_fraud_detector_end_to_end(client, make_user, native) -> None:
    path = Path(__file__).resolve().parents[2] / "models" / "fraud" / "fraud_detector.onnx"
    if not path.is_file():
        pytest.skip("models/fraud/fraud_detector.onnx not present")
    headers = make_user()
    data = path.read_bytes()
    body = upload(client, headers, data, "fraud_detector.onnx")
    assert body["all_supported"] is True
    shape = body["inputs"][0]["shape"]
    x = np.random.RandomState(3).randn(5, shape[1]).astype(np.float32)
    r = _infer(client, headers, body["onnx_model_id"], x)
    assert r.status_code == 200, r.text
    got = np.asarray(r.json()["output"], np.float32).reshape(r.json()["output_shape"])
    np.testing.assert_allclose(got, ort_run(data, x), rtol=RTOL, atol=ATOL)


def test_batch_matches_onnxruntime_and_reports_item_errors(client, make_user, native) -> None:
    headers = make_user()
    data = mlp_bytes()
    model_id = upload(client, headers, data)["onnx_model_id"]
    x = np.random.RandomState(5).randn(2, 4).astype(np.float32)
    batch = [
        {"model_id": model_id, "input": x.reshape(-1).tolist(), "input_shape": [2, 4]},
        {"model_id": "missing", "input": [0.0] * 4, "input_shape": [1, 4]},
    ]
    r = client.post("/inference/batch", headers=headers, json=batch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["success_count"], body["error_count"]) == (1, 1)
    ok, bad = body["results"]
    np.testing.assert_allclose(np.asarray(ok["output"], np.float32).reshape(ok["output_shape"]),
                               ort_run(data, x), rtol=RTOL, atol=ATOL)
    assert ok["backend"] == "crucible-native"
    assert bad["status_code"] == 404


def test_analytics_reflect_own_inferences(client, make_user, native) -> None:
    a, b = make_user(), make_user()
    model_id = upload(client, a, mlp_bytes())["onnx_model_id"]
    assert _infer(client, a, model_id, np.ones((1, 4), np.float32)).status_code == 200
    inf = client.get("/analytics/inference", headers=a).json()
    assert sum(d["count"] for d in inf["data"]) == 1
    assert client.get("/analytics/inference", headers=b).json()["data"] == []
    models = client.get("/analytics/models", headers=a).json()["models"]
    assert [(m["id"], m["usage_count"], m["inference_count"]) for m in models] == [(model_id, 1, 1)]
    assert client.get("/inference/logs", headers=b).json()["count"] == 0


# ---------------------------------------------------------------------------
# Fraud check history
# ---------------------------------------------------------------------------
def test_fraud_history_persists_fields_and_is_scoped(client, make_user) -> None:
    a, b = make_user(), make_user()
    tx = {"tx_type": "TRANSFER", "amount": 500.0, "orig_before": 1000.0, "orig_after": 500.0,
          "dest_before": 0.0, "dest_after": 500.0, "probability": 0.91, "verdict": "High risk",
          "execution_mode": "wasm", "latency_ms": 0.8}
    r = client.post("/fraud/log", headers=a, json=tx)
    assert r.status_code == 200
    hist = client.get("/fraud/history", headers=a).json()
    assert hist["count"] == 1
    item = hist["history"][0]
    assert {k: item[k] for k in tx} == tx
    assert client.get("/fraud/history", headers=b).json()["count"] == 0
    fraud = client.get("/analytics/fraud", headers=a).json()["data"]
    assert [(d["total"], d["fraud_count"]) for d in fraud] == [(1, 1)]
