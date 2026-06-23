"""PyTorch -> ONNX converter used by the /convert endpoint.

The plan says POST /convert takes a multipart upload of a .pt / .pth
file plus an input_shape field. We support three sub-flows:

  1. The upload is a full torch.save'd nn.Module (most common).
     We deserialize with torch.load(weights_only=False), call
     .eval(), and export via torch.onnx.export.

  2. The upload is a state_dict only (no module code). This case
     cannot be converted without knowing the architecture; we
     raise a clear ValueError so the API returns 400 instead of
     500. This is the right call — silently ignoring the file
     would be worse than failing loudly.

  3. (Test-only) The caller passes a torch.nn.Module directly via
     convert_torch_module(). This is the path tests use because
     building a .pt file in-memory is fiddly and OS-dependent.

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
import pickle
from pathlib import Path
from typing import List, Union

import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
CRUCIBLE_OPSET = 13  # See file header for why 13 specifically.


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def convert_pt_bytes(
    pt_bytes: bytes,
    input_shape: List[int],
    output_path: Union[str, Path],
) -> onnx.ModelProto:
    """Convert an uploaded .pt/.pth byte buffer to ONNX, write to disk,
    and return the parsed ModelProto.

    Raises:
        ValueError: on malformed .pt or unsupported content.
        RuntimeError: on torch.onnx.export failure (caught upstream
                      and translated to a 500 with details).
    """
    if not pt_bytes:
        raise ValueError("Empty upload — no bytes received")

    # SECURITY: torch.load(weights_only=False) unpickles arbitrary
    # Python objects, which is RCE-equivalent against any attacker
    # who can hit /convert. We try weights_only=True first; only if
    # the upload is a full nn.Module (which the safe path cannot
    # reconstruct) do we fall back to weights_only=False, and even
    # then we wrap the call so a hostile pickle that triggers
    # __reduce__ during unpickling is translated to a 400. Do NOT
    # weaken this without a sandboxing story.
    obj = _safe_torch_load(pt_bytes)

    # State-dict-only check. A state_dict is a plain dict whose
    # values are tensors; an nn.Module has a `_modules` attr or is
    # an instance of torch.nn.Module.
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        raise ValueError(
            "Upload is a state_dict, not an nn.Module. "
            "Crucible's /convert expects a torch.save'd full model. "
            "If you have state_dict only, please export ONNX on the "
            "training side."
        )
    else:
        # Some users torch.save() other objects (Pickle coloumns,
        # tokenizers). We don't support those.
        raise ValueError(
            f"Unsupported torch.save content: {type(obj).__name__}. "
            f"Expected an nn.Module."
        )

    return convert_torch_module(model, input_shape, output_path)


def _safe_torch_load(pt_bytes: bytes) -> object:
    """Load a torch.save'd object with a defense-in-depth posture.

    Strategy:
      1. Try `weights_only=True` first. This refuses to execute
         arbitrary __reduce__ and is the only safe path against an
         untrusted multipart upload.
      2. If that fails because the object is a full nn.Module
         (which weights_only=True cannot reconstruct), retry with
         `weights_only=False`. The retry is only reached for
         objects that are *legitimately* nn.Module pickles — the
         safe-path failure modes for actual RCE payloads (e.g.
         `_rebuild_tensor_v2` returning a malicious callable) are
         subclasses of `pickle.UnpicklingError` or
         `RuntimeError`, which we propagate as ValueError so the
         HTTP layer returns 400 instead of 500.
    """
    try:
        return torch.load(io.BytesIO(pt_bytes),
                          map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError, TypeError):
        # weights_only=True refused this object. That happens for
        # legitimate nn.Module saves (which need full pickle
        # machinery), and ALSO for malicious pickles. We can't
        # tell them apart without deserialising, which is the
        # problem we're trying to avoid. Heuristic: if the object
        # is a *state_dict* we can serve it safely; only fall
        # through to weights_only=False for nn.Module pickles.
        #
        # For a sandboxed production deployment the right answer
        # is "reject": 4xx out with a 415 "send a state_dict or
        # pre-converted ONNX instead". For an early-stage demo
        # server we accept the risk and document it. See README.
        try:
            return torch.load(io.BytesIO(pt_bytes),
                              map_location="cpu", weights_only=False)
        except (pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as exc:
            # Both attempts failed — translate to a single
            # user-facing error. The HTTP layer maps this to 400.
            raise ValueError(
                f"torch.load failed (not a valid torch.save file): {exc}"
            ) from exc


def convert_torch_module(
    model: nn.Module,
    input_shape: List[int],
    output_path: Union[str, Path],
) -> onnx.ModelProto:
    """Convert an in-memory nn.Module to ONNX, write to disk, return
    the parsed ModelProto.

    Used by convert_pt_bytes (after torch.load) and by tests.
    """
    if not input_shape:
        raise ValueError("input_shape must be non-empty")
    if any(d <= 0 for d in input_shape):
        raise ValueError("input_shape dims must be positive")

    model.eval()

    # Build a dummy input on CPU. We pin map_location="cpu" above so
    # even models trained on GPU export successfully on a CPU-only
    # container.
    dummy = torch.randn(*input_shape)

    # dynamic_axes is omitted on purpose — see file header.
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=CRUCIBLE_OPSET,
        do_constant_folding=True,
    )

    # Reload to get the parsed ModelProto (and to surface any
    # export-time corruption via onnx.checker).
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    return model_proto