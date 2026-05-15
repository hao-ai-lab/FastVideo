# SPDX-License-Identifier: Apache-2.0
"""GPU loading smoke test for ``WanCausalModel``.

Verifies that ``WanCausalModel.__init__`` resolves the
``CausalWanTransformer3DModel`` class override and successfully loads
weights from the regular Wan2.1 1.3B checkpoint.

A real forward pass is intentionally omitted here: the causal
transformer requires per-frame timesteps, a block-causal attention
mask, and KV cache state that ``WanCausalModel.predict_noise_streaming``
manages for production callers.  PR 5 (per-method tests) exercises that
streaming forward path end-to-end.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29516")

from pathlib import Path

import pytest
import torch

from fastvideo.train.models.wan import WanCausalModel
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" /
    "wan_causal_t2v_min.yaml")


@pytest.mark.usefixtures("distributed_setup")
def test_wan_causal_model_loads():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = WanCausalModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )

    transformer = model.transformer
    assert isinstance(transformer, torch.nn.Module)
    assert sum(p.numel() for p in transformer.parameters()) > 0

    # Spot-check that the override pulled the causal class, not the
    # plain Wan one.  Transformer is a torch.nn.Module wrapping the
    # CausalWanTransformer3DModel architecture.
    assert "Causal" in type(transformer).__name__, (
        f"expected CausalWanTransformer3DModel-derived class, got "
        f"{type(transformer).__name__}")
