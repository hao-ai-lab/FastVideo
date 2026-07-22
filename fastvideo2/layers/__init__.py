"""Shared model layers — extracted from the vendored official Wan2.1 model.

Every layer here preserves two things exactly, and the anchor gate proves it
(the DiT rows must stay bitwise 0.0 against the official goldens):

* **state-dict keys** — sequential blocks keep their integer indices
  (``ffn.0.weight``), norms keep ``weight`` — so official checkpoints load
  with no mapping;
* **cast semantics** — where fp32/fp64 islands sit and where dtype promotion
  happens is part of the trained model's numerics, not implementation detail.
  Each layer's docstring names its load-bearing precision behavior.

Layer modules import torch lazily or at module level (they define nn.Modules)
but never core/runtime modules. This ``__init__`` re-exports lazily so that
torch-free consumers (backend selection logic, T0 tests) can import
``fastvideo2.layers.attention`` without torch installed.
"""
from typing import Any

__all__ = ["MLP", "RMSNorm", "FP32LayerNorm", "sinusoidal_embedding_1d",
           "rope_params", "rope_apply", "flash_attention", "attention"]

_EXPORTS = {
    "MLP": "fastvideo2.layers.mlp",
    "RMSNorm": "fastvideo2.layers.norms",
    "FP32LayerNorm": "fastvideo2.layers.norms",
    "sinusoidal_embedding_1d": "fastvideo2.layers.embeddings",
    "rope_params": "fastvideo2.layers.rotary",
    "rope_apply": "fastvideo2.layers.rotary",
    "flash_attention": "fastvideo2.layers.attention",
    "attention": "fastvideo2.layers.attention",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
