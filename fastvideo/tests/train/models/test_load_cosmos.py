# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``CosmosModel``.

Loads the real Cosmos-Predict2.5-2B checkpoint (~2B at bf16) via
``CosmosModel.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.cosmos.CosmosModel`` and the underlying
``Cosmos25Transformer3DModel``.

Cosmos's transformer takes a different forward signature than Wan: a
single Reason1 text embedding (``crossattn_proj_in_channels=100352``),
plus ``condition_mask`` / ``padding_mask`` / ``fps``. This mirrors the
kwargs in ``CosmosModel._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29518")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.cosmos import CosmosModel
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "cosmos_t2w_min.yaml")

# Cosmos 2.5 Reason1 (Qwen2.5-VL) text embedding width.
_COSMOS_TEXT_DIM = 100352


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = CosmosModel(
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

    # Cosmos transformer takes [B, C, T, H, W] (out_channels=16) plus a
    # condition_mask / padding_mask and a single Reason1 text embedding.
    # Small spatial + few frames so this fits next to the 2B model.
    b, c, t, h, w = 1, 16, 4, 32, 32
    hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(b,
                                        16,
                                        _COSMOS_TEXT_DIM,
                                        device=device,
                                        dtype=dtype)
    timestep = torch.tensor([[500]], device=device, dtype=dtype)
    condition_mask = torch.zeros(b, 1, t, h, w, device=device, dtype=dtype)
    padding_mask = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
            fps=16,
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape, (
        f"output shape {tuple(out.shape)} != input shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
