# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``LongCatModel``.

Loads the real LongCat-Video-T2V checkpoint (~13.6B at bf16) via
``LongCatModel.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.longcat.LongCatModel`` and the underlying
``LongCatTransformer3DModel``.

Forward-only (``trainable=False``, ``torch.no_grad``): a single-GPU
backward of the 13.6B transformer exceeds the L40S CI runner, so
LongCat training is not covered by a per-method grad-norm test. The
forward fits (the ~13B HunyuanModel loading test already runs on the
same runner).
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.longcat import LongCatModel
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "longcat_t2v_min.yaml")

# LongCat caption_channels (text embedding width).
_LONGCAT_TEXT_DIM = 4096


@pytest.mark.usefixtures("distributed_setup")
def test_longcat_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = LongCatModel(
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

    # LongCat transformer takes [B, C, T, H, W] (in_channels=16,
    # patch_size=(1,2,2)) and a [B, N_text, 4096] text embedding.
    hidden_states = torch.randn(1, 16, 4, 32, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1,
                                        16,
                                        _LONGCAT_TEXT_DIM,
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
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape, (
        f"output shape {tuple(out.shape)} != input shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
