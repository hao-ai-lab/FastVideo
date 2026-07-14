# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``Kandinsky5Model``.

Loads the real Kandinsky-5.0 Lite checkpoint via ``Kandinsky5Model.__init__``
and runs one transformer forward pass on synthetic inputs. Kandinsky5's
transformer takes a structurally different forward signature than Wan/Hunyuan
-- dual text conditioning (``encoder_hidden_states`` + ``pooled_projections``),
explicit RoPE position tensors, a ``scale_factor``, and channel-last
``[B, T, H, W, C]`` hidden_states with a dict return (``.sample``) -- this
test mirrors the kwargs in ``Kandinsky5Model._build_distill_input_kwargs``
and ``Kandinsky5DenoisingStage``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29518")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.kandinsky5 import Kandinsky5Model
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" /
    "kandinsky5_t2v_min.yaml")


@pytest.mark.usefixtures("distributed_setup")
def test_kandinsky5_model_loads_and_forwards():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)
    model = Kandinsky5Model(
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

    arch = transformer.config.arch_config
    patch_size = arch.patch_size
    in_visual_dim = arch.in_visual_dim
    out_visual_dim = arch.out_visual_dim
    in_text_dim = arch.in_text_dim
    in_text_dim2 = arch.in_text_dim2

    # Small patch grid (T, H, W) so this fits next to the real checkpoint on
    # a single GPU. scale_factor matches the 480p band Kandinsky5Model
    # asserts on -- see fastvideo/train/models/kandinsky5/kandinsky5.py.
    grid_t, grid_h, grid_w = 2, 4, 4
    latent_t = grid_t * patch_size[0]
    latent_h = grid_h * patch_size[1]
    latent_w = grid_w * patch_size[2]

    latents = torch.randn(
        1, latent_t, latent_h, latent_w, in_visual_dim, device=device, dtype=dtype)
    if bool(getattr(transformer, "visual_cond", False)):
        # The shipped checkpoint has visual_cond=True (unified T2V/I2V):
        # Kandinsky5VisualEmbeddings expects [real | zero_cond | zero_mask]
        # concatenated on the channel dim, even for pure T2V. Mirrors
        # Kandinsky5Model._build_distill_input_kwargs / Kandinsky5LatentPreparationStage.
        cond = torch.zeros_like(latents)
        mask = torch.zeros(*latents.shape[:-1], 1, device=device, dtype=dtype)
        hidden_states = torch.cat([latents, cond, mask], dim=-1)
    else:
        hidden_states = latents
    encoder_hidden_states = torch.randn(1, 8, in_text_dim, device=device, dtype=dtype)
    pooled_projections = torch.randn(1, in_text_dim2, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)
    visual_rope_pos = [
        torch.arange(grid_t, device=device),
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
    ]
    text_rope_pos = torch.arange(encoder_hidden_states.shape[1], device=device)

    with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
    ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1.0, 2.0, 2.0),
            sparse_params=None,
            return_dict=True,
        ).sample

    expected_shape = (1, latent_t, latent_h, latent_w, out_visual_dim)
    assert tuple(out.shape) == expected_shape, (
        f"output shape {tuple(out.shape)} != expected {expected_shape}")
    assert torch.isfinite(out).all().item(), "output contains NaN/Inf"
