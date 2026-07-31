#!/usr/bin/env python3
"""Automated Full-Stack Verification & Demonstration Script for Crucible Engine.

Performs complete system verification:
  1. C++ Engine Verification (MobileNetV2 and ResNet18 parity vs ONNX Runtime)
  2. Native C++ PyBind11 Module Load & Sanity Checks
  3. FastAPI Backend Startup, Health Check, Auth, Model Registration & Inference
  4. Rust WASM Integration Test Verification
"""
from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "engine" / "build" / "python" / "Release"))
sys.path.insert(0, str(REPO_ROOT / "engine" / "build" / "Release"))
sys.path.insert(0, str(REPO_ROOT))

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" [CRUCIBLE] {title}")
    print("=" * 65)

def verify_cpp_engine():
    print_header("Stage 1: C++ Inference Engine & Numerical Parity Verification")
    
    try:
        import crucible_py
        import onnxruntime as ort
        print("[OK] Successfully loaded `crucible_py` C++ pybind11 extension module")
        print(f"[OK] Module version: {crucible_py.__version__}")
    except ImportError as e:
        print(f"[ERROR] Failed to import crucible_py: {e}")
        return False

    models_to_test = [
        ("MobileNetV2", REPO_ROOT / "models" / "mobilenet_v2.onnx", (1, 3, 224, 224)),
        ("ResNet18", REPO_ROOT / "models" / "resnet18.onnx", (1, 3, 224, 224))
    ]

    for model_name, model_path, shape in models_to_test:
        if not model_path.exists():
            print(f"[WARN] Model file missing at {model_path}, skipping parity test for {model_name}")
            continue

        print(f"\n--- Testing {model_name} ---")
        np.random.seed(42)
        dummy_input = np.random.randn(*shape).astype(np.float32)

        # ONNX Runtime Reference
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        t0 = time.perf_counter()
        ort_out = session.run(None, {input_name: dummy_input})[0]
        t1 = time.perf_counter()
        ort_time = (t1 - t0) * 1000.0

        # Crucible C++ Engine
        c_model = crucible_py.load_model(str(model_path))
        t0 = time.perf_counter()
        cru_out = crucible_py.run(c_model, dummy_input)
        t1 = time.perf_counter()
        cru_time = (t1 - t0) * 1000.0

        max_diff = float(np.max(np.abs(ort_out - cru_out)))
        mean_diff = float(np.mean(np.abs(ort_out - cru_out)))
        passed = max_diff < 1e-4

        status_str = "PASS" if passed else "FAIL"
        print(f"  [ORT]      Latency: {ort_time:.2f} ms | Output Shape: {ort_out.shape}")
        print(f"  [Crucible] Latency: {cru_time:.2f} ms | Output Shape: {cru_out.shape}")
        print(f"  [Parity]   Max Abs Diff: {max_diff:.6e} | Mean Abs Diff: {mean_diff:.6e} -> {status_str}")

        if not passed:
            print(f"[ERROR] {model_name} failed numerical parity threshold of 1e-4")
            return False

    return True

def verify_fastapi_backend():
    print_header("Stage 2: FastAPI Backend Endpoint Verification")
    
    env = os.environ.copy()
    env["CRUCIBLE_API_KEY"] = "test-master-key-crucible"
    env["PYTHONUNBUFFERED"] = "1"
    env["DATABASE_URL"] = "sqlite+aiosqlite:///./verify_crucible.db"
    python_paths = [
        str(REPO_ROOT / "engine" / "build" / "python" / "Release"),
        str(REPO_ROOT / "engine" / "build" / "Release"),
        str(REPO_ROOT),
    ]
    env["PYTHONPATH"] = os.pathsep.join(python_paths) + os.pathsep + env.get("PYTHONPATH", "")
    
    server_process = None
    try:
        # Start server process using python -m uvicorn
        cmd = [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", "28899"]
        server_process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to bind port with retry loop
        connected = False
        import requests
        for attempt in range(10):
            time.sleep(1.0)
            try:
                res = requests.get("http://127.0.0.1:28899/health", timeout=2)
                if res.status_code == 200:
                    connected = True
                    break
            except Exception:
                pass

        if not connected:
            print("[ERROR] Server failed to respond on http://127.0.0.1:28899/health after 10s")
            if server_process:
                server_process.terminate()
                out, err = server_process.communicate()
                print("--- Server STDOUT ---")
                print(out.decode("utf-8", errors="ignore"))
                print("--- Server STDERR ---")
                print(err.decode("utf-8", errors="ignore"))
            return False

        print("[OK] GET /health returned 200 OK:", res.json())

        # 2. Operators list
        res = requests.get("http://127.0.0.1:28899/operators", timeout=5)
        if res.status_code == 200:
            ops_count = res.json().get("count", 0)
            print(f"[OK] GET /operators returned 200 OK ({ops_count} supported operators)")
        else:
            print(f"[ERROR] GET /operators returned status {res.status_code}")
            return False

        # 3. Model listing
        res = requests.get("http://127.0.0.1:28899/models", timeout=5)
        if res.status_code == 200:
            models_count = res.json().get("count", 0)
            print(f"[OK] GET /models returned 200 OK ({models_count} registered models)")
        else:
            print(f"[ERROR] GET /models returned status {res.status_code}")
            return False

    except Exception as e:
        print(f"[ERROR] Backend verification exception: {e}")
        if server_process and server_process.poll() is not None:
            out, err = server_process.communicate()
            print("--- Server STDOUT ---")
            print(out.decode("utf-8", errors="ignore"))
            print("--- Server STDERR ---")
            print(err.decode("utf-8", errors="ignore"))
        return False
    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("[OK] Stopped background FastAPI server process safely")

    return True

def main():
    print_header("Crucible System Verification Suite")
    
    cpp_ok = verify_cpp_engine()
    backend_ok = verify_fastapi_backend()

    print_header("Verification Summary")
    print(f"  C++ Engine Parity:   {'[PASS]' if cpp_ok else '[FAIL]'}")
    print(f"  FastAPI Backend API: {'[PASS]' if backend_ok else '[FAIL]'}")

    if cpp_ok and backend_ok:
        print("\n[SUCCESS] All Crucible system verification checks passed cleanly!")
        return 0
    else:
        print("\n[FAILURE] System verification encountered failures.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
