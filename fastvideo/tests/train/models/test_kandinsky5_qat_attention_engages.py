# SPDX-License-Identifier: Apache-2.0
"""Forward+backward smoke test confirming ATTN_QAT_TRAIN actually engages
for Kandinsky5's dense/local attention path, rather than silently falling
back to plain SDPA.

Two silent-fallback traps this guards against:
  - ``Kandinsky5Attention.forward`` catches ``AssertionError`` from
    ``LocalAttention`` when no forward context is set and falls back to
    ``F.scaled_dot_product_attention`` (a real fallback for standalone
    parity tests, not relevant here since we always set a forward context,
    but worth asserting past explicitly).
  - ``_cached_get_attn_backend`` is process-wide ``@cache``d, so a stale
    backend selection from an earlier test in the same process can silently
    linger. The role-local model option scopes and clears selection while the
    transformer is built.

This test itself must not become that stale-selection source for whichever
model test runs next in the same pytest process, so the cache is cleared
again in a ``finally`` block.

Unlike ``fastvideo.platforms.cuda``'s other backends, ATTN_QAT_TRAIN has no
silent-fallback path at all if the kernel isn't built -- backend selection
raises ``ImportError`` loudly. This test skips (does not fail) when the
kernel isn't available, since it's an optional build artifact from
fastvideo-kernel/.

Runs the *full* transformer forward (mirroring test_load_kandinsky5.py)
rather than invoking an attention submodule directly. Two earlier versions
of this test tried isolating a single Kandinsky5Attention module -- first a
freshly-random-initialized one (the fake-quant kernel produced NaN/Inf on
untrained weights), then a real submodule pulled out of the loaded model
(FSDP2's fully_shard() registers its unshard/reshard hooks on the top-level
transformer's __call__, not on submodules individually, so invoking a
submodule directly left its Linear layers' parameters in sharded DTensor
form -- incompatible with a plain-Tensor input regardless of the
`trainable` flag). Going through the properly-hooked top-level forward call
sidesteps both problems.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from pathlib import Path

import pytest
import torch

from fastvideo.attention import LocalAttention
from fastvideo.attention.selector import _cached_get_attn_backend
from fastvideo.forward_context import set_forward_context
from fastvideo.platforms import AttentionBackendEnum

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" /
    "kandinsky5_t2v_min.yaml")


@pytest.mark.usefixtures("distributed_setup")
def test_kandinsky5_attn_qat_train_engages_and_backprops():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from fastvideo.attention.backends.attn_qat_train import (
        is_attn_qat_train_available, )
    if not is_attn_qat_train_available():
        pytest.skip("fastvideo_kernel ATTN_QAT_TRAIN kernel is not built")

    _cached_get_attn_backend.cache_clear()
    try:
        from fastvideo.train.models.kandinsky5 import Kandinsky5Model
        from fastvideo.train.utils.config import load_run_config

        cfg = load_run_config(_FIXTURE)
        model = Kandinsky5Model(
            init_from=cfg.models["student"]["init_from"],
            training_config=cfg.training,
            trainable=True,
            attention_backend="ATTN_QAT_TRAIN",
        )

        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        transformer = model.transformer.to(device=device, dtype=dtype)

        attn = transformer.visual_transformer_blocks[0].self_attention
        assert isinstance(attn.local_attention, LocalAttention)
        assert attn.local_attention.backend == AttentionBackendEnum.ATTN_QAT_TRAIN, (
            f"expected ATTN_QAT_TRAIN, got {attn.local_attention.backend} -- "
            "backend selection silently fell back")

        arch = transformer.config.arch_config
        patch_size = arch.patch_size
        in_visual_dim = arch.in_visual_dim
        in_text_dim = arch.in_text_dim
        in_text_dim2 = arch.in_text_dim2

        grid_t, grid_h, grid_w = 2, 4, 4
        latent_t = grid_t * patch_size[0]
        latent_h = grid_h * patch_size[1]
        latent_w = grid_w * patch_size[2]

        latents = torch.randn(
            1, latent_t, latent_h, latent_w, in_visual_dim, device=device, dtype=dtype,
            requires_grad=True)
        if bool(getattr(transformer, "visual_cond", False)):
            # See Kandinsky5Model._build_distill_input_kwargs /
            # Kandinsky5LatentPreparationStage: visual_cond=True checkpoints
            # always expect [real | zero_cond | zero_mask] concatenated on the
            # channel dim.
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

        with set_forward_context(current_timestep=0, attn_metadata=None):
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

        assert torch.isfinite(out).all().item(), "output contains NaN/Inf"

        out.sum().backward()
        assert latents.grad is not None
        assert torch.isfinite(latents.grad).all().item(), "input grad contains NaN/Inf"
        assert attn.to_query.weight.grad is not None
        assert torch.isfinite(attn.to_query.weight.grad.to_local()
                              if hasattr(attn.to_query.weight.grad, "to_local") else
                              attn.to_query.weight.grad).all().item(), "weight grad contains NaN/Inf"
    finally:
        # Clear the process-wide selector cache so a later model test cannot
        # reuse this test's ATTN_QAT_TRAIN selection.
        _cached_get_attn_backend.cache_clear()
