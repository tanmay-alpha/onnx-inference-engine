"""API tests for the Crucible server.

Native-engine tests (``native`` fixture) compare /infer against onnxruntime;
they skip without crucible_py unless CRUCIBLE_REQUIRE_NATIVE=1.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from server import database, engine, main, validator
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


def test_login_unknown_email_still_verifies_a_hash(client, monkeypatch) -> None:
    calls = []
    real_verify = main.verify_password

    def _counting(plain, hashed):
        calls.append(hashed)
        return real_verify(plain, hashed)

    monkeypatch.setattr(main, "verify_password", _counting)
    r = client.post("/auth/login", json={"email": "nobody@example.com", "password": "whatever-123"})
    assert r.status_code == 401
    assert calls == [main.DUMMY_PASSWORD_HASH]


def test_login_rate_limited_per_ip_and_email(client, make_user) -> None:
    make_user("victim@example.com")  # its login is the first attempt for this IP+email
    bad = {"email": "victim@example.com", "password": "wrong-password"}
    assert [client.post("/auth/login", json=bad).status_code for _ in range(9)] == [401] * 9
    r = client.post("/auth/login", json={"email": "victim@example.com", "password": "correct-horse-1"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # Other emails from the same IP keep their own (per IP+email) budget.
    other = client.post("/auth/login", json={"email": "other@example.com", "password": "wrong-password"})
    assert other.status_code == 401


def test_auth_rate_limited_per_ip_and_proxy_header_opt_in(client, monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "AUTH_RATE_LIMIT_PER_IP", 3)
    creds = [{"email": f"u{i}@example.com", "password": "wrong-password"} for i in range(3)]
    assert [client.post("/auth/login", json=c).status_code for c in creds] == [401] * 3
    assert client.post("/auth/login", json={"email": "x@example.com", "password": "p"}).status_code == 429
    new_user = {"email": "new@example.com", "password": "correct-horse-1"}
    assert client.post("/auth/register", json=new_user).status_code == 429
    spoofed = {"X-Forwarded-For": "203.0.113.9"}
    # Untrusted by default: a client-chosen X-Forwarded-For does not open a new bucket.
    assert client.post("/auth/register", json=new_user, headers=spoofed).status_code == 429
    monkeypatch.setattr(main.settings, "TRUST_PROXY_HEADERS", True)
    assert client.post("/auth/register", json=new_user,
                       headers={"X-Forwarded-For": "203.0.113.9, 10.0.0.1"}).status_code == 201


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


def test_analysis_busy_is_503_and_stores_nothing(client, make_user, monkeypatch, model_dir) -> None:
    headers = make_user()
    busy = threading.BoundedSemaphore(1)
    busy.acquire()
    monkeypatch.setattr(main, "_ANALYSIS_SLOTS", busy)
    files = {"model_file": ("m.onnx", mlp_bytes(), "application/octet-stream")}
    r = client.post("/convert", headers=headers, files=files)
    assert r.status_code == 503 and r.headers["Retry-After"]
    assert client.post("/validate", headers=headers, files=files).status_code == 503
    assert files_in(model_dir) == []
    assert client.get("/models", headers=headers).json()["count"] == 0
    busy.release()
    upload(client, headers, mlp_bytes())


def test_graph_node_limit(client, make_user, monkeypatch, model_dir) -> None:
    headers = make_user()
    monkeypatch.setattr(validator, "MAX_GRAPH_NODES", 3)  # the MLP has 4 nodes
    files = {"model_file": ("m.onnx", mlp_bytes(), "application/octet-stream")}
    r = client.post("/convert", headers=headers, files=files)
    assert r.status_code == 400 and "4 nodes; the limit is 3" in r.json()["detail"]
    assert files_in(model_dir) == []
    assert client.get("/models", headers=headers).json()["count"] == 0
    r = client.post("/validate", headers=headers, files=files)
    assert r.status_code == 200
    assert r.json()["valid"] is False
    assert r.json()["unsupported_reasons"] == ["graph has 4 nodes; the limit is 3"]


def test_model_count_quota(client, make_user, monkeypatch, model_dir) -> None:
    a, b = make_user(), make_user()
    monkeypatch.setattr(main, "MAX_MODELS_PER_USER", 2)
    upload(client, a, mlp_bytes())
    upload(client, a, mlp_bytes())
    r = client.post("/convert", headers=a,
                    files={"model_file": ("m.onnx", mlp_bytes(), "application/octet-stream")})
    assert r.status_code == 409
    assert len(files_in(model_dir)) == 2
    upload(client, b, mlp_bytes())  # quotas are per user


def test_storage_bytes_quota(client, make_user, monkeypatch, model_dir) -> None:
    headers = make_user()
    size = len(mlp_bytes())
    monkeypatch.setattr(main, "MAX_STORAGE_BYTES_PER_USER", 2 * size - 1)
    first = upload(client, headers, mlp_bytes())["onnx_model_id"]
    r = client.post("/convert", headers=headers,
                    files={"model_file": ("m.onnx", mlp_bytes(), "application/octet-stream")})
    assert r.status_code == 413 and "quota" in r.json()["detail"]
    assert files_in(model_dir) == [model_dir / f"{first}.onnx"]
    assert client.delete(f"/models/{first}", headers=headers).status_code == 200
    upload(client, headers, mlp_bytes())


def test_operator_key_is_exempt_from_quotas(client, monkeypatch) -> None:
    monkeypatch.setenv("CRUCIBLE_API_KEY", "operator-key-0123456789")
    monkeypatch.setattr(main, "MAX_MODELS_PER_USER", 1)
    monkeypatch.setattr(main, "MAX_STORAGE_BYTES_PER_USER", 1)
    op = {"X-API-Key": "operator-key-0123456789"}
    upload(client, op, mlp_bytes())
    upload(client, op, mlp_bytes())


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


def test_infer_rejects_model_requesting_huge_tensor(client, make_user, native) -> None:
    # A ~100-byte model whose Conv pads request a ~640 GB output: the engine's
    # per-tensor budget must turn this into a 422, not an OOM-killed worker.
    graph = helper.make_graph(
        [helper.make_node("Conv", ["X", "W"], ["Y"], pads=[200000] * 4)], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 400004, 400004])],
        [numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    headers = make_user()
    model_id = upload(client, headers, model.SerializeToString())["onnx_model_id"]
    r = _infer(client, headers, model_id, np.ones((1, 1, 4, 4), np.float32))
    assert r.status_code == 422
    assert "limit" in r.json()["detail"]


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


class _ResourceLimitError(RuntimeError):
    """Stands in for crucible_py.ResourceLimitError in stub modules."""


def _stub_engine(run_fn):
    class _Stub:
        ResourceLimitError = _ResourceLimitError
        load_model = staticmethod(lambda path: object())
        run = staticmethod(run_fn)
    return _Stub


def _wait_all_slots_free(timeout: float) -> bool:
    """True once every inference slot can be taken (i.e. no native run in flight)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        taken = 0
        while taken < engine.MAX_CONCURRENT and engine._SLOTS.acquire(blocking=False):
            taken += 1
        for _ in range(taken):
            engine._SLOTS.release()
        if taken == engine.MAX_CONCURRENT:
            return True
        time.sleep(0.02)
    return False


def test_infer_passes_limits_and_backstop_timeout_is_504(client, make_user, monkeypatch) -> None:
    """The deadline and memory limits reach crucible_py.run; a run overrunning them still gets 504."""
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    seen = {}

    def _slow_run(model, inputs, **limits):
        seen.update(limits)
        time.sleep(0.5)  # ignores its deadline: the asyncio backstop must answer
        return np.zeros((1, 3), np.float32)

    monkeypatch.setattr(engine, "crucible_py", _stub_engine(_slow_run))
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SEC", 0.05)
    r = _infer(client, headers, model_id, np.ones((1, 4), np.float32))
    assert r.status_code == 504
    assert seen == {"timeout_ms": 50, "max_tensor_elements": engine.MAX_TENSOR_ELEMENTS,
                    "max_total_elements": engine.MAX_TOTAL_ELEMENTS}
    logs = client.get("/inference/logs", headers=headers).json()["logs"]
    assert [l["status"] for l in logs] == ["timeout"]
    assert _wait_all_slots_free(2.0)


@pytest.mark.parametrize("message,status", [
    ("run deadline exceeded (in node 3, MatMul)", 504),
    ("intermediate tensors need 70000000 elements in total", 422),
    ("tensor of 20000000 elements is above the configured limit of 16000000 elements per tensor", 422),
])
def test_engine_resource_limit_errors_map_to_status(client, make_user, monkeypatch, message, status) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]

    def _raising_run(model, inputs, **limits):
        raise _ResourceLimitError(message)

    monkeypatch.setattr(engine, "crucible_py", _stub_engine(_raising_run))
    r = _infer(client, headers, model_id, np.ones((1, 4), np.float32))
    assert r.status_code == status, r.text
    logs = client.get("/inference/logs", headers=headers).json()["logs"]
    assert [l["status"] for l in logs] == ["timeout" if status == 504 else "error"]


def _matmul_chain_bytes(n_nodes: int, dim: int) -> bytes:
    """X[dim,dim] -> MatMul(W) repeated n_nodes times; W is the identity."""
    w = numpy_helper.from_array(np.eye(dim, dtype=np.float32), "W")
    nodes, prev = [], "X"
    for i in range(n_nodes):
        out = "Y" if i == n_nodes - 1 else f"T{i}"
        nodes.append(helper.make_node("MatMul", [prev, "W"], [out]))
        prev = out
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [dim, dim])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [dim, dim])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def test_native_deadline_stops_the_run(client, make_user, monkeypatch, native) -> None:
    """A ~400-node MatMul chain takes seconds; the engine must stop at the deadline and free its slot."""
    headers = make_user()
    model_id = upload(client, headers, _matmul_chain_bytes(400, 512))["onnx_model_id"]
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SEC", 0.3)
    t0 = time.monotonic()
    r = _infer(client, headers, model_id, np.ones((512, 512), np.float32))
    assert r.status_code == 504, r.text
    assert time.monotonic() - t0 < 5.0
    # Without the native deadline the run would keep its slot for many seconds.
    assert _wait_all_slots_free(2.0)


def test_native_total_elements_limit_is_422(client, make_user, monkeypatch, native) -> None:
    """Eight 1M-element copies of an initializer, all live until summed, exceed a 4M budget."""
    big = numpy_helper.from_array(np.ones((1000, 1000), np.float32), "B")
    copies = [f"I{i}" for i in range(8)]
    nodes = [helper.make_node("Identity", ["B"], [c]) for c in copies]  # all run first
    acc = copies[0]
    for i, c in enumerate(copies[1:], 1):
        nodes.append(helper.make_node("Add", [acc, c], [f"S{i}"]))
        acc = f"S{i}"
    nodes.append(helper.make_node("Add", [acc, "X"], ["Y"]))
    graph = helper.make_graph(
        nodes, "copies",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1000])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1000, 1000])],
        [big],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    headers = make_user()
    model_id = upload(client, headers, model.SerializeToString())["onnx_model_id"]
    x = np.ones((1, 1000), np.float32)
    monkeypatch.setattr(engine, "MAX_TOTAL_ELEMENTS", 4_000_000)
    r = _infer(client, headers, model_id, x)
    assert r.status_code == 422, r.text[:500]
    assert "in total" in r.json()["detail"]
    monkeypatch.setattr(engine, "MAX_TOTAL_ELEMENTS", 64_000_000)
    r = _infer(client, headers, model_id, x)
    assert r.status_code == 200
    assert r.json()["output"][:3] == [9.0, 9.0, 9.0]


def test_output_cap_is_413_for_infer_and_batch(client, make_user, monkeypatch) -> None:
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    monkeypatch.setattr(engine, "crucible_py",
                        _stub_engine(lambda model, inputs, **limits: np.zeros((1, 3), np.float32)))
    item = {"model_id": model_id, "input": [1.0] * 4, "input_shape": [1, 4]}
    monkeypatch.setattr(main, "MAX_OUTPUT_ELEMENTS", 2)
    r = _infer(client, headers, model_id, np.ones((1, 4), np.float32))
    assert r.status_code == 413 and "Output has 3 elements" in r.json()["detail"]
    monkeypatch.setattr(main, "MAX_OUTPUT_ELEMENTS", 5)
    assert _infer(client, headers, model_id, np.ones((1, 4), np.float32)).status_code == 200
    r = client.post("/inference/batch", headers=headers, json=[item, item, item])
    assert r.status_code == 413 and "in total" in r.json()["detail"]
    assert client.post("/inference/batch", headers=headers, json=[item]).status_code == 200


def test_api_key_rate_limit_is_enforced(client, make_user, monkeypatch) -> None:
    from sqlalchemy.orm import Session

    jwt_headers = make_user()
    created = client.post("/auth/api-key", headers=jwt_headers, json={"name": "ci"}).json()
    assert created["rate_limit"] == 100
    with Session(database._get_sync_engine()) as s:
        s.get(database.ApiKey, created["id"]).rate_limit = 3
        s.commit()
    key_headers = {"X-API-Key": created["full_key"]}
    x = np.zeros((1, 4), np.float32)
    assert [_infer(client, key_headers, "missing", x).status_code for _ in range(3)] == [404] * 3
    r = _infer(client, key_headers, "missing", x)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # JWT callers have no per-minute limit.
    assert _infer(client, jwt_headers, "missing", x).status_code == 404
    # A batch spends one unit per item and is refused as a whole when they don't fit.
    monkeypatch.setattr(engine, "crucible_py",
                        _stub_engine(lambda model, inputs, **limits: np.zeros((1, 3), np.float32)))
    item = {"model_id": "missing", "input": [0.0] * 4, "input_shape": [1, 4]}
    assert client.post("/inference/batch", headers=key_headers, json=[item]).status_code == 429


def test_loaded_model_is_cached_and_delete_evicts(client, make_user, monkeypatch, model_dir) -> None:
    """The second /infer reuses the loaded model; DELETE drops it from the cache."""
    headers = make_user()
    model_id = upload(client, headers, mlp_bytes())["onnx_model_id"]
    path = str((model_dir / f"{model_id}.onnx").resolve())
    loads = []

    class _CountingModule:
        @staticmethod
        def load_model(p):
            loads.append(p)
            return object()

        @staticmethod
        def run(model, inputs, **limits):
            return np.zeros((1, 3), np.float32)

    monkeypatch.setattr(engine, "crucible_py", _CountingModule)
    x = np.ones((1, 4), np.float32)
    assert _infer(client, headers, model_id, x).status_code == 200
    assert _infer(client, headers, model_id, x).status_code == 200
    assert loads == [path]
    assert path in engine.cached_paths()
    assert client.delete(f"/models/{model_id}", headers=headers).status_code == 200
    assert path not in engine.cached_paths()


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
