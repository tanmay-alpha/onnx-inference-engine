"""Tests for Crucible server database layer."""
from __future__ import annotations

import uuid

import pytest

from server.database import (
    init_db,
    reset_engine,
    save_model,
    get_model,
    list_models,
    delete_model,
    log_inference,
    get_inference_logs,
    log_fraud_tx,
    get_fraud_history,
    log_benchmark,
    get_benchmarks,
)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Use a fresh SQLite database for each test."""
    db_path = tmp_path / "test_crucible.db"
    monkeypatch.setenv("CRUCIBLE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("CRUCIBLE_DB_PATH", str(db_path))
    import server.database as db_mod

    db_mod.reset_engine()
    from server.database import init_db
    init_db()
    yield


class TestModelCRUD:
    def test_save_and_get_model(self):
        model_id = uuid.uuid4().hex
        save_model(
            model_id=model_id,
            name="test-model.onnx",
            file_path="/tmp/models/test-model.onnx",
            file_size_bytes=1024,
            input_shape=[1, 3, 224, 224],
            operators=["Conv", "Relu", "Gemm"],
            all_supported=True,
        )
        rec = get_model(model_id)
        assert rec is not None
        assert rec["id"] == model_id
        assert rec["name"] == "test-model.onnx"
        assert rec["file_size_bytes"] == 1024
        assert rec["input_shape"] == [1, 3, 224, 224]
        assert rec["all_supported"] is True

    def test_list_models(self):
        initial_count = len(list_models())
        for i in range(3):
            save_model(
                model_id=uuid.uuid4().hex,
                name=f"model-{i}.onnx",
                file_path=f"/tmp/models/model-{i}.onnx",
                file_size_bytes=512 * (i + 1),
                input_shape=[1, 3],
                operators=["Relu"],
                all_supported=True,
            )
        models = list_models()
        assert len(models) == initial_count + 3

    def test_delete_model(self):
        model_id = uuid.uuid4().hex
        save_model(
            model_id=model_id,
            name="to-delete.onnx",
            file_path="/tmp/models/to-delete.onnx",
            file_size_bytes=256,
            input_shape=[1],
            operators=[],
            all_supported=True,
        )
        result = delete_model(model_id)
        assert result is True
        assert get_model(model_id) is None


class TestInferenceLogs:
    def test_log_and_retrieve(self):
        model_id = uuid.uuid4().hex
        save_model(
            model_id=model_id,
            name="inf-model.onnx",
            file_path="/tmp/models/inf-model.onnx",
            file_size_bytes=512,
            input_shape=[1, 3],
            operators=["Relu"],
            all_supported=True,
        )
        log = log_inference(
            model_id=model_id,
            input_shape=[1, 3],
            output_shape=[1, 10],
            inference_time_ms=12.5,
            engine="crucible-cpp",
        )
        assert log["model_id"] == model_id
        assert log["inference_time_ms"] == 12.5

        logs = get_inference_logs(limit=10)
        assert len(logs) >= 1
        assert any(l["id"] == log["id"] for l in logs)


class TestFraudTransactions:
    def test_log_and_retrieve(self):
        tx = log_fraud_tx(
            tx_type="TRANSFER",
            amount=50000.0,
            orig_before=60000.0,
            orig_after=10000.0,
            dest_before=20000.0,
            dest_after=70000.0,
            probability=0.82,
            verdict="High risk",
            execution_mode="wasm",
            latency_ms=1.2,
        )
        assert tx["amount"] == 50000.0
        assert tx["probability"] == 0.82

        history = get_fraud_history(limit=10)
        assert any(h["id"] == tx["id"] for h in history)


class TestBenchmarks:
    def test_log_and_retrieve(self):
        bench = log_benchmark(
            model_name="test-model.onnx",
            engine="crucible-wasm",
            latency_ms=14.3,
            memory_mb=2.5,
        )
        assert bench["latency_ms"] == 14.3
        assert bench["memory_mb"] == 2.5

        benches = get_benchmarks(limit=10)
        assert any(b["id"] == bench["id"] for b in benches)
