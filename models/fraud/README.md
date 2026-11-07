# Fraud model

`fraud_detector.onnx` (195 bytes) is a logistic regression over 7 transaction
features: three ONNX nodes, `MatMul → Add → Sigmoid`, opset 13. It exists to
give Crucible a tiny real model to run in the browser demo (`/fraud`), the API
and the parity tests. It is **not** a validated fraud detector.

## How it was trained

`train_fraud_model.py` with no arguments reproduces the committed
`fraud_detector.onnx` byte for byte, and `model_config.json`:

```bash
pip install numpy scikit-learn onnx
python models/fraud/train_fraud_model.py --out-dir /tmp/fraud   # compare with models/fraud/
```

* **Data:** 50,000 synthetic rows from `synthetic_data()` (seed 42, 1.3%
  fraud). Fraud rows are large transfers that empty the origin account;
  legitimate rows are small. These are not real transactions.
* **Split:** first 80% of the shuffled rows train, last 20% test.
* **Model:** scikit-learn `LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)`.
* **Export:** the weights are written into a hand-built `MatMul → Add → Sigmoid`
  graph with `onnx.helper`, so Crucible runs it without the `ai.onnx.ml` operators.

The script reports a held-out AUC of 1.0. The synthetic classes are separable by
construction, so that number only shows the script works. It is not an
accuracy claim.

### Training on real PaySim data

Download the PaySim CSV from Kaggle
([`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1)) into
`models/data/`, then:

```bash
pip install pandas
python models/fraud/train_fraud_model.py --paysim-csv models/data/PS_20174392719_1491204439457_log.csv --out-dir /tmp/fraud-paysim
```

The features, split, model and export are the same. The normalisation
constants will be different. The WASM runtime hard-codes them, so before
shipping such a model, copy the new `mean` and `std` into `FRAUD_MEAN` and
`FRAUD_STD` in `wasm/src/lib.rs`. `--publish` also copies the ONNX file to
`web/public/models/`. No PaySim-trained model or metric is committed.

## Input contract

Input `input`: float32 `[batch, 7]`. Output `prob`: float32 `[batch, 1]`,
the fraud probability. The UI treats ≥ 0.5 as fraud (`threshold` in `model_config.json`).

| # | Feature | Browser demo field (`web/src/routes/fraud.tsx`) |
|---|---------|------------------------------------------------|
| 0 | `amount` | Amount |
| 1 | `oldbalanceOrg` | Origin balance before |
| 2 | `newbalanceOrig` | Origin balance after |
| 3 | `oldbalanceDest` | Destination balance before |
| 4 | `newbalanceDest` | Destination balance after |
| 5 | `type_CASH_OUT` | 1.0 if type is `CASH_OUT`, else 0.0 |
| 6 | `type_TRANSFER` | 1.0 if type is `TRANSFER`, else 0.0 (`OTHER` sets both to 0) |

The graph takes **normalised** features: `(x - mean[i]) / std[i]`, using the
training-split `mean`/`std` in `model_config.json`. In the browser,
`runFraudDetection` (`web/src/lib/crucible-wasm.ts`) passes the raw values
and one-hot flags to `runFraudModel` (`wasm/src/lib.rs`). That function
normalises with the same constants (`FRAUD_MEAN`, `FRAUD_STD`) and then runs the
graph on `[1, 7]`. Callers of the API or CLI pass the raw graph input, so they
must normalise the features themselves.

## Files

| File | Description |
|------|-------------|
| `fraud_detector.onnx` | The model. `web/public/models/fraud_detector.onnx` is an identical copy |
| `model_config.json` | Feature order, normalisation `mean`/`std`, threshold, data source, held-out AUC |
| `train_fraud_model.py` | Reproduces both files (see above) |
