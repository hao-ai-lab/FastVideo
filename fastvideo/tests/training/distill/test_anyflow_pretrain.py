# SPDX-License-Identifier: Apache-2.0
"""AnyFlow pretrain method tests.

CPU-only unit tests covering:
- Config flag defaults (bit-identity preserved on legacy paths).
- ``WanTimeTextImageEmbedding`` dual-timestep forward (additive default = bit-identical
  to legacy; gated mode reproduces AnyFlow's ``(1 - g) * temb + g * delta_emb`` fusion).
- ``WanTransformer3DModel.forward`` accepts ``r_timestep``.
- ``FlowMapEulerDiscreteScheduler`` numerics: ``apply_shift``, ``get_train_weight``,
  ``step``.
- ``(t, r)`` per-batch sampling distribution.
- Central-difference target math.
- AnyFlow HF checkpoint key remap (``remap_anyflow_keys``).
"""

from __future__ import annotations

import copy

import pytest
import torch

from fastvideo.configs.models.dits import WanVideoConfig


# ---------------------------------------------------------------------------
# Task 1: r_embedder config flags default to bit-identity preservation.
# ---------------------------------------------------------------------------


def test_wan_arch_defaults_preserve_bit_identity() -> None:
    cfg = WanVideoConfig()
    arch = cfg.arch_config
    assert arch.r_embedder is False
    assert arch.r_embedder_fusion == "additive"
    assert arch.r_embedder_gate_value == 0.25
    assert arch.r_embedder_deltatime_type == "r"
