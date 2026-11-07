"""Train the 7-feature fraud model and export it as a MatMul -> Add -> Sigmoid ONNX graph.

Run from the repository root:

    # Reproduce the committed models/fraud/fraud_detector.onnx + model_config.json
    python models/fraud/train_fraud_model.py

    # Train on the real PaySim CSV (download it from Kaggle into models/data/ first)
    python models/fraud/train_fraud_model.py --paysim-csv models/data/PS_20174392719_1491204439457_log.csv

The model is scikit-learn LogisticRegression on z-score normalised features.
The graph is built by hand with onnx.helper so it only uses MatMul, Add and
Sigmoid, which every Crucible runtime (native engine and WASM) executes.

Feature order (the ONNX input is [batch, 7] in this order):
    amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
    type_CASH_OUT (1.0/0.0), type_TRANSFER (1.0/0.0)
Each feature is normalised as (x - mean) / std with the training-split
statistics written to model_config.json. The graph does NOT normalise; callers
must (wasm/src/lib.rs bakes in the committed constants).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]
SEED = 42
TRAIN_FRACTION = 0.8

_FRAUD_DIR = Path(__file__).resolve().parent
_REPO = _FRAUD_DIR.parent.parent


def synthetic_data(rng: np.random.Generator, n: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Two-population synthetic data that produced the committed model.

    Fraud rows are large transfers that empty the origin account; legitimate
    rows are small. The classes are separable by construction, so a held-out
    AUC on this data says nothing about real-world fraud detection.
    """
    n_fraud = int(n * 0.013)
    n_legit = n - n_fraud

    amount_l = rng.exponential(5_000, n_legit).astype(np.float32)
    oldorg_l = rng.exponential(80_000, n_legit).astype(np.float32)
    neworg_l = np.clip(oldorg_l - amount_l, 100, None).astype(np.float32)
    olddst_l = rng.exponential(60_000, n_legit).astype(np.float32)
    newdst_l = (olddst_l + amount_l).astype(np.float32)
    co_l = rng.choice([0, 1], n_legit, p=[0.65, 0.35]).astype(np.float32)
    tr_l = rng.choice([0, 1], n_legit, p=[0.85, 0.15]).astype(np.float32)

    amount_f = rng.uniform(100_000, 1_000_000, n_fraud).astype(np.float32)
    oldorg_f = (amount_f * rng.uniform(0.98, 1.05, n_fraud)).astype(np.float32)
    neworg_f = np.zeros(n_fraud, dtype=np.float32)
    olddst_f = rng.exponential(10_000, n_fraud).astype(np.float32)
    newdst_f = (olddst_f + amount_f).astype(np.float32)
    co_f = rng.choice([0, 1], n_fraud, p=[0.5, 0.5]).astype(np.float32)
    tr_f = rng.choice([0, 1], n_fraud, p=[0.5, 0.5]).astype(np.float32)

    x = np.vstack(
        [
            np.c_[amount_l, oldorg_l, neworg_l, olddst_l, newdst_l, co_l, tr_l],
            np.c_[amount_f, oldorg_f, neworg_f, olddst_f, newdst_f, co_f, tr_f],
        ]
    ).astype(np.float32)
    y = np.concatenate([np.zeros(n_legit, dtype=np.int32), np.ones(n_fraud, dtype=np.int32)])
    idx = rng.permutation(n)
    return x[idx], y[idx]


def paysim_data(csv_path: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Load the Kaggle PaySim CSV (columns: type, amount, oldbalanceOrg, ..., isFraud)."""
    import pandas as pd

    df = pd.read_csv(
        csv_path,
        usecols=["type", "amount", "oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest", "isFraud"],
    )
    x = np.c_[
        df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].to_numpy(),
        (df["type"] == "CASH_OUT").to_numpy(),
        (df["type"] == "TRANSFER").to_numpy(),
    ].astype(np.float32)
    y = df["isFraud"].to_numpy(dtype=np.int32)
    idx = rng.permutation(len(y))
    return x[idx], y[idx]


def export_onnx(model: LogisticRegression) -> onnx.ModelProto:
    """prob = sigmoid(input @ W + b) with input [batch, 7], prob [batch, 1]."""
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "W"], ["z_raw"]),
            helper.make_node("Add", ["z_raw", "b"], ["z"]),
            helper.make_node("Sigmoid", ["z"], ["prob"]),
        ],
        "fraud_detector",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, len(FEATURES)])],
        [helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None, 1])],
        [
            numpy_helper.from_array(model.coef_.T.astype(np.float32), "W"),
            numpy_helper.from_array(model.intercept_.astype(np.float32), "b"),
        ],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paysim-csv", type=Path,
                   help="Kaggle PaySim CSV; omit to use the synthetic data behind the committed model")
    p.add_argument("--out-dir", type=Path, default=_FRAUD_DIR,
                   help="where to write fraud_detector.onnx and model_config.json (default: models/fraud)")
    p.add_argument("--publish", action="store_true",
                   help="also copy the ONNX file to web/public/models/ for the browser demo")
    args = p.parse_args()

    rng = np.random.default_rng(SEED)
    if args.paysim_csv:
        x, y = paysim_data(args.paysim_csv, rng)
        source = f"PaySim CSV {args.paysim_csv.name}"
    else:
        x, y = synthetic_data(rng)
        source = "synthetic two-population data (synthetic_data(), seed 42, 50,000 rows)"
    print(f"{len(y):,} rows from {source}; fraud rate {y.mean() * 100:.2f}%")

    n_train = int(len(y) * TRAIN_FRACTION)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    mean = x_train.mean(0).astype(np.float64)
    std = np.where(x_train.std(0) == 0, 1.0, x_train.std(0)).astype(np.float64)

    model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    model.fit(((x_train - mean) / std).astype(np.float32), y_train)
    auc = roc_auc_score(y_test, model.predict_proba(((x_test - mean) / std).astype(np.float32))[:, 1])
    print(f"held-out AUC ({len(y_test):,} rows): {auc:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "fraud_detector.onnx"
    onnx_path.write_bytes(export_onnx(model).SerializeToString())
    config = {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": 0.5,
        "model_type": "LogisticRegression",
        "training_data": source,
        "test_auc": round(float(auc), 4),
        "test_auc_note": (
            "ROC AUC on the last 20% of the shuffled rows. On the synthetic data the "
            "classes are separable by construction, so this is not a real-world accuracy figure."
        ),
    }
    (args.out_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes) and model_config.json")

    if args.publish:
        web = _REPO / "web" / "public" / "models" / "fraud_detector.onnx"
        web.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, web)
        print(f"copied to {web}")
    if args.paysim_csv:
        print("Normalisation constants changed: update FRAUD_MEAN/FRAUD_STD in wasm/src/lib.rs "
              "from model_config.json before publishing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
