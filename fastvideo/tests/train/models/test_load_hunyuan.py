# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``HunyuanModel``.

Loads the real HunyuanVideo checkpoint (~13B at bf16) via
``HunyuanModel.__init__`` and runs one transformer forward pass on
synthetic inputs. Hunyuan's transformer takes a slightly different
forward signature than Wan (no ``encoder_attention_mask``, no
``return_dict``); this test mirrors the kwargs in
``HunyuanModel._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.hunyuan import HunyuanModel
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" /
    "hunyuan_t2v_min.yaml")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = HunyuanModel(
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

    # Hunyuan transformer takes [B, C, T, H, W] (in_channels=16,
    # patch_size=2 spatial, patch_size_t=1).  Small spatial + few
    # frames so this fits next to the 13B model on a single L40S.
    hidden_states = torch.randn(1, 16, 8, 16, 16, device=device, dtype=dtype)
    # Hunyuan splits encoder_hidden_states into a leading global token
    # and per-token text embeddings, so length must be >= 2.
    encoder_hidden_states = torch.randn(1, 4, 4096, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape, (
        f"output shape {tuple(out.shape)} != input shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
