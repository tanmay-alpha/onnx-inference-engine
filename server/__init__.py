"""Crucible FastAPI inference server.

Public surface:
  * server.main          FastAPI app instance (`app`)
  * server.schemas       Pydantic request/response models
  * server.validator     ONNX structural validation + supported-op catalogue
  * server.converter     PyTorch -> ONNX conversion

The bindings test in `server.test_bindings` lives at module level
(in test_bindings.py) rather than under server/ because it is
about the C++ extension, not this server.
"""
