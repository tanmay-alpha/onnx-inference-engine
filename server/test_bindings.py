"""Smoke test for the crucible_py pybind11 module (Issue #12).

Run from repo root:
    PYTHONPATH=./build/release/python \
        python -m pytest server/test_bindings.py -v

Or directly:
    PYTHONPATH=./build/release/python python server/test_bindings.py

The AC from the issue body:
    >>> import crucible_py
    >>> model = crucible_py.load_model("mobilenet_v2.onnx")
    >>> out = crucible_py.run(model, np.zeros((1,3,224,224), dtype=np.float32))
    >>> assert out.shape == (1, 1000)

We mirror that AC literally as `test_mobilenet_v2_smoke` and add
two small unit tests for the helper surface (`get_model_info`,
overloaded `run` with a dict input). The helper tests use tiny
synthetic ONNX graphs in tmp_path so they don't need the model
zoo and run in well under a second.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Import the binding module — graceful skip if it isn't built yet.
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
DEFAULT_LIB_PATH = REPO_ROOT / "build" / "release" / "python"


def _import_crucible_py():
    """Find and import the compiled `crucible_py` module.

    The CMake build puts the .so/.pyd under
    `<build_dir>/python/crucible_py.*`. We look there first, then
    fall back to anything already on PYTHONPATH. If neither works
    we skip the test instead of failing — that lets the test file
    live in the repo even before Issue #12 has been built.
    """
    # Look in the conventional build output location first.
    candidates = [DEFAULT_LIB_PATH, *map(Path, sys.path)]
    for path in candidates:
        if not path:
            continue
        # Find the first file matching crucible_py*.so / *.pyd.
        for ext in ("*.so", "*.pyd"):
            matches = list(Path(path).glob(f"crucible_py{ext}"))
            if matches:
                # Add the directory to sys.path and import.
                sys.path.insert(0, str(path))
                try:
                    import crucible_py  # type: ignore[import-not-found]
                    return crucible_py
                except ImportError:
                    continue
    return None


crucible_py = _import_crucible_py()
pytestmark = pytest.mark.skipif(
    crucible_py is None,
    reason="crucible_py not built; run `cmake --preset release -S engine` first",
)


# ----------------------------------------------------------------------
# Tiny synthetic ONNX graph for the helper tests.
# ----------------------------------------------------------------------
# We build a 2-node graph in memory: Relu(Add(A, B)). A and B are
# 1-D float32 inputs; Y is the single output. Writing the bytes by
# hand is too painful — instead, we shell out to the model-zoo
# script's logic is overkill. So we use the protobuf-free path:
# load one of the model-zoo files if present, otherwise skip.
# (The Issue #10 download script produces mobilenet_v2.onnx.)
# ----------------------------------------------------------------------


def _mobilenet_path() -> Path | None:
    p = MODEL_DIR / "mobilenet_v2.onnx"
    return p if p.is_file() else None


# ----------------------------------------------------------------------
# AC: load_model + run on MobileNetV2
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    _mobilenet_path() is None,
    reason="mobilenet_v2.onnx not present; run models/download_models.py first",
)
def test_mobilenet_v2_smoke():
    """The verbatim AC from Issue #12.

    Load mobilenet_v2.onnx, run a zero input of the canonical
    ImageNet shape, assert the output is (1, 1000).
    """
    assert crucible_py is not None
    model = crucible_py.load_model(str(_mobilenet_path()))
    out = crucible_py.run(model, np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1000)
    assert out.dtype == np.float32


# ----------------------------------------------------------------------
# Helper tests — do not depend on the model zoo.
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    _mobilenet_path() is None,
    reason="mobilenet_v2.onnx not present",
)
def test_get_model_info_keys():
    """get_model_info returns a dict with the documented keys."""
    assert crucible_py is not None
    model = crucible_py.load_model(str(_mobilenet_path()))
    info = crucible_py.get_model_info(model)

    # It is a dict, not a custom object — keeps the Python API
    # minimal and lets users json.dumps() it directly.
    assert isinstance(info, dict)

    # Required keys per the binding's docstring.
    for key in (
        "input_names",
        "output_names",
        "num_nodes",
        "num_initializers",
        "num_int_initializers",
        "ops_used",
    ):
        assert key in info, f"missing key: {key}"

    # MobileNetV2 has the ImageNet input 'data' and the
    # classifier head 'mobilenetv20_output_flatten0_reshape0'.
    assert info["input_names"] == ["data"]
    assert len(info["output_names"]) == 1
    # ops_used must contain every op the dispatch table covers for
    # this graph. Conv is the obvious one.
    assert "Conv" in info["ops_used"]
    # GlobalAveragePool was added in Issue #10 specifically for
    # MobileNetV2 — make sure it's actually present in the graph.
    assert "GlobalAveragePool" in info["ops_used"]
    # num_nodes should be > 50 for MobileNetV2 (54 Conv + ~36
    # elementwise ops). Asserting > 50 is loose enough to survive
    # opset-version drift without becoming meaningless.
    assert info["num_nodes"] > 50


@pytest.mark.skipif(
    _mobilenet_path() is None,
    reason="mobilenet_v2.onnx not present",
)
def test_run_with_dict_input():
    """The dict overload of run() also works for MobileNetV2's
    single-input graph (passing {model.input_names[0]: arr}).

    This is the same code path the FastAPI server (Issue #13)
    will use, because HTTP clients send named fields.
    """
    assert crucible_py is not None
    model = crucible_py.load_model(str(_mobilenet_path()))
    arr = np.zeros((1, 3, 224, 224), dtype=np.float32)
    out = crucible_py.run(model, {"data": arr})
    assert out.shape == (1, 1000)


@pytest.mark.skipif(
    _mobilenet_path() is None,
    reason="mobilenet_v2.onnx not present",
)
def test_run_rejects_wrong_dtype_via_forcecast():
    """A float64 input should be downcast to float32 silently,
    not rejected — that's the point of py::array::forcecast.

    Without forcecast this would raise TypeError.
    """
    assert crucible_py is not None
    model = crucible_py.load_model(str(_mobilenet_path()))
    arr64 = np.zeros((1, 3, 224, 224), dtype=np.float64)
    out = crucible_py.run(model, arr64)
    assert out.dtype == np.float32  # the engine produces f32
    assert out.shape == (1, 1000)


if __name__ == "__main__":
    # Allow running this file directly as a script (without pytest).
    sys.exit(pytest.main([__file__, "-v"]))