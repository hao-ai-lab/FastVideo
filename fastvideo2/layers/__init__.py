"""Shared model layers — extracted from the vendored official Wan2.1 model.

Every layer here preserves two things exactly, and the anchor gate proves it
(the DiT rows must stay bitwise 0.0 against the official goldens):

* **state-dict keys** — sequential blocks keep their integer indices
  (``ffn.0.weight``), norms keep ``weight`` — so official checkpoints load
  with no mapping;
* **cast semantics** — where fp32/fp64 islands sit and where dtype promotion
  happens is part of the trained model's numerics, not implementation detail.
  Each layer's docstring names its load-bearing precision behavior.

Layer modules import torch (they define nn.Modules) but never core/runtime
modules — they sit at the bottom of the import graph, next to nothing.
"""
from fastvideo2.layers.attention import attention, flash_attention
from fastvideo2.layers.embeddings import sinusoidal_embedding_1d
from fastvideo2.layers.mlp import MLP
from fastvideo2.layers.norms import FP32LayerNorm, RMSNorm
from fastvideo2.layers.rotary import rope_apply, rope_params

__all__ = ["MLP", "RMSNorm", "FP32LayerNorm", "sinusoidal_embedding_1d",
           "rope_params", "rope_apply", "flash_attention", "attention"]
