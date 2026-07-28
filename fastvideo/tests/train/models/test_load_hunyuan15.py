# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``Hunyuan15Model``.

Loads the real HunyuanVideo-1.5 480p T2V checkpoint (~8B at bf16) via
``Hunyuan15Model.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.hunyuan15.Hunyuan15Model`` and the underlying
``HunyuanVideo15Transformer3DModel``.

HY1.5's forward differs from Wan's: ``encoder_hidden_states`` is a list
of two tensors (Qwen 3584 + ByT5 1472), ``encoder_hidden_states_image``
is a one-element list whose all-zero content selects the T2V branch, and
there is no ``encoder_attention_mask`` / ``return_dict``. This mirrors
the kwargs in ``Hunyuan15Model._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29522")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.hunyuan15 import Hunyuan15Model
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "hunyuan15_t2v_min.yaml")

_QWEN_DIM = 3584
_BYT5_DIM = 1472
_IMAGE_TOKENS = 729
_IMAGE_DIM = 1152


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan15_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = Hunyuan15Model(
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

    # HY1.5 takes [B, C, T, H, W] with 32 latent channels. Small spatial
    # + few frames so this fits next to the 8B model.
    b, c, t, h, w = 1, 32, 4, 16, 16
    hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    qwen_embeds = torch.randn(b, 20, _QWEN_DIM, device=device, dtype=dtype)
    byt5_embeds = torch.randn(b, 8, _BYT5_DIM, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    # All-zero image embeddings select the T2V branch.
    zero_image_embeds = torch.zeros(b,
                                    _IMAGE_TOKENS,
                                    _IMAGE_DIM,
                                    device=device,
                                    dtype=dtype)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=[qwen_embeds, byt5_embeds],
            timestep=timestep,
            encoder_hidden_states_image=[zero_image_embeds],
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape, (
        f"output shape {tuple(out.shape)} != input shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan15_forwards_with_empty_byt5():
    """Captions without glyph text carry a zero-length ByT5 stream."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = Hunyuan15Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    transformer = model.transformer.to(device=device, dtype=dtype).eval()

    b, c, t, h, w = 1, 32, 4, 16, 16
    hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    qwen_embeds = torch.randn(b, 20, _QWEN_DIM, device=device, dtype=dtype)
    # Zero tokens, not zero values.
    byt5_embeds = torch.zeros(b, 0, _BYT5_DIM, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    zero_image_embeds = torch.zeros(b,
                                    _IMAGE_TOKENS,
                                    _IMAGE_DIM,
                                    device=device,
                                    dtype=dtype)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=[qwen_embeds, byt5_embeds],
            timestep=timestep,
            encoder_hidden_states_image=[zero_image_embeds],
        )

    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all().item(), (
        "empty ByT5 produced NaN/Inf in the output")
