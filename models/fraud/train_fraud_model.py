"""
Fraud detection model — LogisticRegression -> manual ONNX export.
Run with: python -X utf8 train_fraud_model.py
"""

import numpy as np
import json, os, shutil
import onnx
from onnx import numpy_helper, TensorProto, helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import onnxruntime as rt

# ── 1. Synthetic PaySim dataset (two-population) ──────────────────────────────
print("Step 1: Generating data...")
rng = np.random.default_rng(42)
N = 50_000
n_fraud = int(N * 0.013)
n_legit = N - n_fraud

amount_l      = rng.exponential(5_000,   n_legit).astype(np.float32)
oldorg_l      = rng.exponential(80_000,  n_legit).astype(np.float32)
neworg_l      = np.clip(oldorg_l - amount_l, 100, None).astype(np.float32)
olddst_l      = rng.exponential(60_000,  n_legit).astype(np.float32)
newdst_l      = (olddst_l + amount_l).astype(np.float32)
co_l          = rng.choice([0,1], n_legit, p=[0.65,0.35]).astype(np.float32)
tr_l          = rng.choice([0,1], n_legit, p=[0.85,0.15]).astype(np.float32)

amount_f      = rng.uniform(100_000, 1_000_000, n_fraud).astype(np.float32)
oldorg_f      = (amount_f * rng.uniform(0.98, 1.05, n_fraud)).astype(np.float32)
neworg_f      = np.zeros(n_fraud, dtype=np.float32)
olddst_f      = rng.exponential(10_000, n_fraud).astype(np.float32)
newdst_f      = (olddst_f + amount_f).astype(np.float32)
co_f          = rng.choice([0,1], n_fraud, p=[0.5,0.5]).astype(np.float32)
tr_f          = rng.choice([0,1], n_fraud, p=[0.5,0.5]).astype(np.float32)

X = np.vstack([
    np.c_[amount_l, oldorg_l, neworg_l, olddst_l, newdst_l, co_l, tr_l],
    np.c_[amount_f, oldorg_f, neworg_f, olddst_f, newdst_f, co_f, tr_f],
]).astype(np.float32)
y = np.concatenate([np.zeros(n_legit,dtype=np.int32), np.ones(n_fraud,dtype=np.int32)])
idx = rng.permutation(N)
X, y = X[idx], y[idx]
print(f"  {N:,} rows, fraud={y.mean()*100:.2f}%")

# ── 2. Normalize ──────────────────────────────────────────────────────────────
print("Step 2: Normalizing...")
n_tr = int(N * 0.8)
X_tr, X_te = X[:n_tr], X[n_tr:]
y_tr, y_te = y[:n_tr], y[n_tr:]
mean = X_tr.mean(0).astype(np.float64)
std  = np.where(X_tr.std(0)==0, 1.0, X_tr.std(0)).astype(np.float64)
X_tr_s = ((X_tr - mean)/std).astype(np.float32)
X_te_s  = ((X_te  - mean)/std).astype(np.float32)

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("Step 3: Training LogisticRegression...")
lr = LogisticRegression(C=1.0, max_iter=300, solver='lbfgs', n_jobs=1)
lr.fit(X_tr_s, y_tr)
auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:,1])
print(f"  AUC: {auc:.4f}")
assert auc > 0.95, f"AUC {auc:.4f} < 0.95"

# ── 4. ONNX export ─────────────────────────────────────────────────────────────
print("Step 4: Building ONNX graph...")
W = lr.coef_.T.astype(np.float32)
b = lr.intercept_.astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("MatMul",["input","W"],["z_raw"]),
     helper.make_node("Add",   ["z_raw","b"],["z"]),
     helper.make_node("Sigmoid",["z"],       ["prob"])],
    "fraud_detector",
    [helper.make_tensor_value_info("input",TensorProto.FLOAT,[None,7])],
    [helper.make_tensor_value_info("prob", TensorProto.FLOAT,[None,1])],
    [numpy_helper.from_array(W,"W"), numpy_helper.from_array(b,"b")],
)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])
m.ir_version = 8
onnx.checker.check_model(m)

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("Step 5: Saving...")
os.makedirs("models/fraud", exist_ok=True)
onnx_path = "models/fraud/fraud_detector.onnx"
with open(onnx_path,"wb") as f: f.write(m.SerializeToString())
print(f"  {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

with open("models/fraud/model_config.json","w") as f:
    json.dump({"features":["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest",
               "newbalanceDest","type_CASH_OUT","type_TRANSFER"],
               "mean":mean.tolist(),"std":std.tolist(),"threshold":0.5,
               "auc":round(float(auc),4),"model_type":"LogisticRegression"}, f, indent=2)

os.makedirs("web/public/models", exist_ok=True)
shutil.copyfile(onnx_path,"web/public/models/fraud_detector.onnx")

# ── 6. Verify ─────────────────────────────────────────────────────────────────
print("Step 6: Verifying with onnxruntime...")
sess = rt.InferenceSession(onnx_path)
preds = sess.run(None,{"input":X_te_s[:5]})[0]
print(f"  probs: {[round(float(p),4) for p in preds.flatten()]}")
print("Done!")
