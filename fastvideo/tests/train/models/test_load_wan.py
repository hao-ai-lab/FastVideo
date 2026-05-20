# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``WanModel``.

Loads the real Wan2.1 1.3B checkpoint via ``WanModel.__init__`` and
runs one transformer forward pass on synthetic inputs. Catches loader
or forward-signature regressions in
``fastvideo.train.models.wan.WanModel`` and the underlying
``WanTransformer3DModel``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.wan import WanModel
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "wan_t2v_min.yaml")


@pytest.mark.usefixtures("distributed_setup")
def test_wan_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = WanModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )

    transformer = model.transformer
    assert isinstance(transformer, torch.nn.Module)
    assert sum(p.numel() for p in transformer.parameters()) > 0

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    transformer = transformer.to(device=device, dtype=dtype).eval()

    # Tiny synthetic inputs at the smallest patch-aligned resolution
    # (Wan patch_size=(1,2,2)).  Keeps activations small so the test
    # fits comfortably alongside the 1.3B model on a single L40S.
    hidden_states = torch.randn(1, 16, 4, 32, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1,
                                        16,
                                        4096,
                                        device=device,
                                        dtype=dtype)
    encoder_attention_mask = torch.ones(1, 16, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=False,
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape, (
        f"output shape {tuple(out.shape)} != input shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
