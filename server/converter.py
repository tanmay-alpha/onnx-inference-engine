"""PyTorch -> ONNX converter used by the /convert endpoint.

The plan originally said POST /convert takes a multipart upload of
a .pt / .pth file plus an input_shape field. After the Issue #13
security review, the public /convert path accepts a pre-exported
.onnx file (and an input_shape that the server uses to record the
model's expected input dims). PyTorch -> ONNX conversion still
exists for in-process use, but it is NOT exposed over HTTP because
the only safe loader (torch.load(weights_only=True)) cannot
reconstruct a full nn.Module — and the unsafe loader
(weights_only=False) is RCE-equivalent.

Three flows remain:

  1. Public /convert: caller uploads a .onnx file. We onnx-load,
     onnx.check it, and save it under the model registry. This is
     the only flow that crosses the network.

  2. convert_torch_module(model, input_shape, output_path): in-process
     use from tests and from any future operator-side tooling.
     The caller supplies a real nn.Module so we can call
     torch.onnx.export directly. No deserialization involved.

  3. _safe_torch_load(pt_bytes): if a future endpoint needs to
     accept a state_dict, this is the only sanctioned loader. It
     uses weights_only=True exclusively. A full nn.Module pickle
     raises ValueError (HTTP 400) — the right answer, because
     "send a pre-exported .onnx" is the secure equivalent.

Why opset_version=13?
    Crucible's executor targets ONNX opset 7 (the version
    MobileNetV2 + ResNet18 in models/ were exported with). opset 13
    is the lowest version that torch.onnx.export can emit today
    while still producing graphs the C++ parser handles cleanly.
    Higher opsets are fine — the parser ignores unknown attributes
    rather than failing — but 13 is the sweet spot.

Why dynamic_axes=False?
    The plan requires input_shape on the request. With static axes,
    torch.onnx.export embeds the shape in the graph and Crucible's
    executor gets exactly the dimensions it expects at runtime.
    Dynamic axes would force us to plumb a reshape path through
    the executor, which isn't in scope for Issue #13.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List, Union

import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
CRUCIBLE_OPSET = 13  # See file header for why 13 specifically.

# Models are persisted under this directory. The default matches the
# config key in .env.example. This is the ONLY directory the converter
# will write to; we validate that explicitly to prevent path traversal.
_DEFAULT_MODEL_DIR = os.environ.get("CRUCIBLE_MODEL_DIR", "/tmp/models")


def _resolve_safe_path(output_path: Union[str, Path]) -> Path:
    """Resolve output_path and ensure it lives under the model directory or temp dir.

    Prevents path traversal attacks where a caller passes
    "../../etc/passwd" or similar. Both the caller-supplied path and
    the model directory are canonicalized with realpath before
    comparison, so symlinks and relative segments are resolved.
    """
    model_dir = Path(os.environ.get("CRUCIBLE_MODEL_DIR", "/tmp/models")).resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    target = Path(output_path).resolve()

    try:
        target.relative_to(model_dir)
        return target
    except ValueError:
        pass

    try:
        target.relative_to(temp_dir)
        return target
    except ValueError:
        pass

    raise ValueError(
        f"output_path must be inside {model_dir} or {temp_dir}, got {target}"
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def accept_onnx_upload(
    onnx_bytes: bytes,
    input_shape: List[int],
    output_path: Union[str, Path],
) -> onnx.ModelProto:
    """Validate and persist an uploaded .onnx file.

    Replaces convert_pt_bytes for the public HTTP path. The caller
    (FastAPI /convert) supplies a multipart .onnx upload; we
    onnx-load + onnx.check it, then write the parsed model to
    `output_path`. The model proto is returned so /convert can run
    the supported-ops check before responding.

    Why onnx.check at the upload boundary?
        onnx.checker surfaces malformed protobuf / dangling
        initializers / shape inference errors that we would
        otherwise discover at /infer time (and translate to a 500).
        Catching them at upload gives a 400 with a useful message.

    Why a path-containment check?
        output_path is a public argument. If a future caller
        accepts user-controlled paths, ``../../etc/passwd`` style
        inputs would otherwise escape the model directory. We
        resolve both sides and assert containment.
    """
    if not onnx_bytes:
        raise ValueError("Empty upload — no bytes received")
    if not input_shape:
        raise ValueError("input_shape must be non-empty")
    if any(d <= 0 for d in input_shape):
        raise ValueError("input_shape dims must be positive")
    # input_shape is recorded by /convert into ConvertResponse so
    # the caller doesn't have to repeat it on /infer. We don't
    # embed it into the .onnx (ONNX has no metadata for that),
    # so /infer still requires an input_shape field. The
    # duplication is the price of not parsing the ONNX again on
    # /infer.

    safe_path = _resolve_safe_path(output_path)

    try:
        model_proto = onnx.load_from_string(onnx_bytes)
    except Exception as exc:
        raise ValueError(f"Invalid ONNX (load failed): {exc}") from exc

    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"Invalid ONNX (check failed): {exc}") from exc

    safe_path.write_bytes(onnx_bytes)
    return model_proto


def _safe_torch_load(pt_bytes: bytes) -> object:
    """Load a torch.save'd object with NO deserialization RCE risk.

    Hard rule: we ONLY call torch.load with weights_only=True. The
    previous version had a weights_only=False fallback for "legitimate
    nn.Module pickles"; the security review correctly flagged that as
    equivalent to RCE against any attacker who can hit /convert. We
    accept the regression that full nn.Module uploads no longer work
    through this path: clients who want to convert an nn.Module must
    upload a pre-exported .onnx (handled by accept_onnx_upload above)
    or call convert_torch_module directly in-process.

    This helper is kept for two reasons:
      1. A future /admin/state_dict endpoint might want to accept a
         state_dict. weights_only=True can load those.
      2. Tests can use it to round-trip a state_dict without going
         through the network.
    """
    try:
        return torch.load(io.BytesIO(pt_bytes),
                          map_location="cpu", weights_only=True)
    except Exception as exc:
        # Catch all (not just UnpicklingError) so a hostile pickle
        # always becomes a clean 400, never a 500.
        raise ValueError(
            "Refusing to deserialize pickle that weights_only=True "
            f"cannot load: {type(exc).__name__}: {exc}"
        ) from exc


def convert_torch_module(
    model: nn.Module,
    input_shape: List[int],
    output_path: Union[str, Path],
) -> onnx.ModelProto:
    """Convert an in-memory nn.Module to ONNX, write to disk, return
    the parsed ModelProto.

    Used by tests and by any future operator-side tooling that has
    a real nn.Module in memory. NOT exposed over HTTP (see file
    header for the rationale).
    """
    if not input_shape:
        raise ValueError("input_shape must be non-empty")
    if any(d <= 0 for d in input_shape):
        raise ValueError("input_shape dims must be positive")

    safe_path = _resolve_safe_path(output_path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Build a dummy input on CPU. We pin map_location="cpu" above so
    # even models trained on GPU export successfully on a CPU-only
    # container.
    dummy = torch.randn(*input_shape)

    # dynamic_axes is omitted on purpose — see file header.
    torch.onnx.export(
        model,
        (dummy,),
        str(safe_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=CRUCIBLE_OPSET,
        do_constant_folding=True,
    )

    # Reload to get the parsed ModelProto (and to surface any
    # export-time corruption via onnx.checker).
    model_proto = onnx.load(str(safe_path))
    onnx.checker.check_model(model_proto)
    return model_proto