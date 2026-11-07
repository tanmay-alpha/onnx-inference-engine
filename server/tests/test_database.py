"""Tests for the synchronous CRUD helpers in server.database."""
from __future__ import annotations

import uuid

from server import database


def _save(model_id: str, owner: str | None) -> None:
    database.save_model(
        model_id=model_id, name=f"{model_id}.onnx", file_path=f"/tmp/models/{model_id}.onnx",
        size_bytes=1024, owner_id=owner, opset=13, operators_used=["Gemm", "Relu"],
        unsupported_ops=[], unsupported_reasons=[], all_supported=True,
        inputs=[{"name": "X", "shape": [-1, 4], "dtype": "float32"}],
        outputs=[{"name": "Y", "shape": [-1, 3], "dtype": "float32"}], native_checked=True,
    )


def test_model_roundtrip_and_scoping() -> None:
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    _save("m1", a)
    _save("m2", b)
    rec = database.get_model("m1", owner_id=a)
    assert rec is not None
    assert rec["size_bytes"] == 1024 and rec["opset"] == 13
    assert rec["operators_used"] == ["Gemm", "Relu"] and rec["all_supported"] is True
    assert rec["inputs"] == [{"name": "X", "shape": [-1, 4], "dtype": "float32"}]
    assert database.get_model("m1", owner_id=b) is None
    assert database.get_model("m1", owner_id=None, all_owners=True) is not None
    assert [m["id"] for m in database.list_models(owner_id=a)] == ["m1"]
    assert {m["id"] for m in database.list_models(owner_id=None, all_owners=True)} == {"m1", "m2"}
    assert database.delete_model("m1") is True
    assert database.delete_model("m1") is False


def test_inference_logs_scoped_and_count_usage() -> None:
    a = uuid.uuid4().hex
    _save("m", a)
    database.log_inference(model_id="m", user_id=a, api_key_id=None, input_shape=[1, 4],
                           output_shape=[1, 3], latency_ms=1.5, backend="crucible-native")
    database.log_inference(model_id="m", user_id=a, api_key_id=None, input_shape=[1, 4],
                           output_shape=None, latency_ms=0.0, backend="crucible-native",
                           status="error", error_message="bad")
    logs = database.get_inference_logs(owner_id=a)
    assert [l["status"] for l in logs] and {l["status"] for l in logs} == {"success", "error"}
    ok = next(l for l in logs if l["status"] == "success")
    assert ok["input_shape"] == [1, 4] and ok["output_shape"] == [1, 3] and ok["backend"] == "crucible-native"
    assert database.get_inference_logs(owner_id=uuid.uuid4().hex) == []
    assert database.get_model("m", owner_id=a)["usage_count"] == 1


def test_fraud_history_roundtrip() -> None:
    a = uuid.uuid4().hex
    item = database.log_fraud_tx(user_id=a, tx_type="CASH_OUT", amount=10.0, orig_before=20.0,
                                 orig_after=10.0, dest_before=0.0, dest_after=10.0, probability=0.2,
                                 verdict="Low risk", execution_mode="server", latency_ms=1.0)
    assert item["tx_type"] == "CASH_OUT" and item["probability"] == 0.2
    hist = database.get_fraud_history(owner_id=a)
    assert [h["id"] for h in hist] == [item["id"]]
    assert database.get_fraud_history(owner_id=None) == []


def test_postgres_urls_use_installed_drivers(monkeypatch) -> None:
    monkeypatch.delenv("CRUCIBLE_DB_PATH")
    from server.config import get_settings
    monkeypatch.setattr(get_settings(), "DATABASE_URL", "postgres://u:p@h/db")
    url = database._get_database_url()
    assert url == "postgresql+asyncpg://u:p@h/db"
    assert database._sync_url(url) == "postgresql+psycopg2://u:p@h/db"
