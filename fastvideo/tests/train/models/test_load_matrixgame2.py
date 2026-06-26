# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``MatrixGame2Model``.

Loads the real Matrix-Game-2.0-Base checkpoint (~14B at bf16) via
``MatrixGame2Model.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.matrixgame2.MatrixGame2Model`` and the
underlying ``MatrixGame2WanModel``.

Matrix-Game 2.0 is an action-conditioned I2V world model: no text
encoder, image-action cross-attention (``image_dim=1280``). The
forward concatenates conditioning latents onto the noisy latents, so
``hidden_states`` is built at the transformer's loaded ``in_channels``.
Action inputs (mouse/keyboard) are left ``None`` (optional). This
mirrors ``MatrixGame2Model._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.matrixgame2.matrixgame2 import MatrixGame2Model
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "matrixgame2_min.yaml")

# Matrix-Game 2.0 CLIP image-embedding width.
_MG2_IMAGE_DIM = 1280


@pytest.mark.usefixtures("distributed_setup")
def test_matrixgame2_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = MatrixGame2Model(
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

    # Read the loaded in_channels off the patch embedding so the
    # synthetic latents match the conditioning-concatenated width the
    # transformer expects (16 noise + image/mask cond channels).
    in_channels = transformer.patch_embedding.proj.in_channels

    hidden_states = torch.randn(1, in_channels, 4, 32, 32, device=device,
                                dtype=dtype)
    # CLIP image embeds drive the image-action cross-attention; no text.
    encoder_hidden_states_image = torch.randn(1, 257, _MG2_IMAGE_DIM,
                                              device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            timestep=timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            mouse_cond=None,
            keyboard_cond=None,
            return_dict=False,
        )

    if isinstance(out, tuple):
        out = out[0]
    # out has out_channels (16); compare the non-channel dims.
    assert out.shape[0] == hidden_states.shape[0]
    assert out.shape[2:] == hidden_states.shape[2:], (
        f"output spatial/temporal shape {tuple(out.shape)} != input "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
