# SPDX-License-Identifier: Apache-2.0
"""Stage-boundary regression: ``Kandinsky5DmdDenoisingStage.forward`` must
invoke the selected attention backend on every denoising step.

``Kandinsky5Attention.forward`` catches the ``AssertionError``
``LocalAttention`` raises when no pipeline forward context is set and
silently falls back to plain ``F.scaled_dot_product_attention`` -- so if
the ``set_forward_context`` wrapper inside the stage's denoising loop were
ever removed, generation would still "work" while skipping whichever
kernel ``FASTVIDEO_ATTENTION_BACKEND`` selected. The GPU test
``fastvideo/tests/train/models/test_kandinsky5_qat_attention_engages.py``
installs ``set_forward_context`` itself around a raw transformer call, so
it cannot catch that removal; this test drives the *production stage* and
spies on the resolved backend impl (``LocalAttention.attn_impl.forward``,
which is only reachable once ``get_forward_context()`` succeeds inside
``LocalAttention.forward``).

CPU-only: a tiny randomly-initialized Kandinsky5 transformer
(~127K params) driven through the real stage code. TORCH_SDPA is pinned
via env var so backend resolution is identical on CPU-only and CUDA
machines. Output *numerics* are deliberately not asserted -- with random
weights the 4-step predict-x0/re-noise feedback loop amplifies
magnitudes without bound; the contract under test is backend routing,
not sample quality (the nightly e2e owns that).
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo.attention import LocalAttention
from fastvideo.attention.selector import _cached_get_attn_backend
from fastvideo.configs.models.dits.kandinsky5 import (
    Kandinsky5ArchConfig,
    Kandinsky5VideoConfig,
)
from fastvideo.forward_context import set_forward_context
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )

DMD_STEPS = [1000, 750, 500, 250]
TEXT_SEQ_LEN = 6


def _tiny_arch() -> Kandinsky5ArchConfig:
    # model_dim must be divisible by head_dim = sum(axes_dims) = 32.
    return Kandinsky5ArchConfig(
        in_visual_dim=4,
        in_text_dim=32,
        in_text_dim2=16,
        time_dim=32,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=64,
        ff_dim=128,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(8, 12, 12),
        visual_cond=False,
        attention_type="regular",
    )


def _build_stage_and_spy(monkeypatch):
    """Tiny transformer + real DMD stage on CPU, with every resolved
    backend impl wrapped in a call counter."""
    # Pin the backend so resolution is identical on CPU-only and CUDA
    # machines, and clear the process-wide selector cache keyed on
    # (head_size, dtype, supported_backends) -- NOT on the env var (same
    # defensive pattern as test_kandinsky5_qat_attention_engages.py).
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    _cached_get_attn_backend.cache_clear()

    from fastvideo.models.dits.kandinsky5 import Kandinsky5Transformer3DModel
    import fastvideo.pipelines.stages.kandinsky5 as k5_stage_mod

    torch.manual_seed(0)
    arch = _tiny_arch()
    transformer = Kandinsky5Transformer3DModel(Kandinsky5VideoConfig(arch_config=arch), hf_config={})
    transformer.eval()

    # get_local_torch_device() never returns "cpu" (cuda -> mps fallback);
    # pin the stage to CPU so this test runs identically everywhere.
    monkeypatch.setattr(k5_stage_mod, "get_local_torch_device", lambda: torch.device("cpu"))

    stage = k5_stage_mod.Kandinsky5DmdDenoisingStage(transformer, FlowMatchEulerDiscreteScheduler(shift=5.0))

    backend_calls: list[int] = []
    for module in transformer.modules():
        if isinstance(module, LocalAttention):

            def _spy(*args, __orig=module.attn_impl.forward, **kwargs):
                backend_calls.append(1)
                return __orig(*args, **kwargs)

            module.attn_impl.forward = _spy

    return stage, transformer, arch, backend_calls


def _make_batch() -> types.SimpleNamespace:
    # 5 frames / 64x64 with the real VAE ratios (4 temporal, 8 spatial)
    # -> latents [1, 2, 8, 8, 4], patchified to a 2x4x4 grid.
    return types.SimpleNamespace(
        latents=torch.randn(1, 2, 8, 8, 4),
        prompt_embeds=[torch.randn(1, TEXT_SEQ_LEN, 32), torch.randn(1, 16)],
        prompt_attention_mask=[torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)],
        height=64,
        width=64,
        num_frames=5,
        image_latent=None,
        generator=None,
    )


def _make_fastvideo_args(arch: Kandinsky5ArchConfig) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_loaded={"transformer": True},
        disable_autocast=True,
        pipeline_config=types.SimpleNamespace(
            dit_precision="fp32",
            dmd_denoising_steps=list(DMD_STEPS),
            dit_config=types.SimpleNamespace(arch_config=arch),
            vae_config=types.SimpleNamespace(arch_config=types.SimpleNamespace(
                temporal_compression_ratio=4,
                spatial_compression_ratio=8,
            )),
        ),
    )


def test_dmd_stage_invokes_selected_backend_every_step(monkeypatch):
    stage, _, arch, backend_calls = _build_stage_and_spy(monkeypatch)
    try:
        batch = _make_batch()

        result = stage.forward(batch, _make_fastvideo_args(arch))

        # 3 LocalAttention modules (text self-attn, visual self-attn, visual
        # cross-attn) x one forward per DMD step. Any silent
        # missing-forward-context fallback to F.scaled_dot_product_attention
        # bypasses attn_impl.forward entirely and shows up here as a shortfall.
        expected = 3 * len(DMD_STEPS)
        assert len(backend_calls) == expected, (
            f"selected attention backend invoked {len(backend_calls)} times, expected {expected} -- "
            "the stage's set_forward_context wrapper is no longer covering the transformer call, "
            "so Kandinsky5Attention silently fell back to raw SDPA")
        assert result.latents.shape == (1, 2, 8, 8, 4)
    finally:
        _cached_get_attn_backend.cache_clear()


def test_raw_transformer_without_context_bypasses_backend(monkeypatch):
    """Documents the trap the stage wrapper exists to prevent: without a
    forward context the transformer still returns output, but the selected
    backend is never invoked."""
    stage, transformer, _, backend_calls = _build_stage_and_spy(monkeypatch)
    del stage  # only the spied transformer is needed here
    try:
        with torch.no_grad():
            out = transformer(
                hidden_states=torch.randn(1, 2, 8, 8, 4),
                encoder_hidden_states=torch.randn(1, TEXT_SEQ_LEN, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([500.0]),
                visual_rope_pos=[torch.arange(2), torch.arange(4), torch.arange(4)],
                text_rope_pos=torch.arange(TEXT_SEQ_LEN),
                scale_factor=(1.0, 2.0, 2.0),
                sparse_params=None,
                return_dict=True,
            ).sample

        assert out.shape == (1, 2, 8, 8, 4)
        assert len(backend_calls) == 0, (
            "expected the missing-forward-context fallback to bypass the backend impl; "
            "if this now fails, Kandinsky5Attention's fallback behavior changed and the "
            "stage guard docs/tests should be revisited")
    finally:
        _cached_get_attn_backend.cache_clear()


def test_dmd_stage_with_context_and_raw_without_share_one_spy(monkeypatch):
    """Same spy, both paths, one process: proves the counter difference
    between the two tests above is the forward-context wrapper itself,
    not some environmental difference."""
    stage, transformer, arch, backend_calls = _build_stage_and_spy(monkeypatch)
    try:
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            transformer(
                hidden_states=torch.randn(1, 2, 8, 8, 4),
                encoder_hidden_states=torch.randn(1, TEXT_SEQ_LEN, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([500.0]),
                visual_rope_pos=[torch.arange(2), torch.arange(4), torch.arange(4)],
                text_rope_pos=torch.arange(TEXT_SEQ_LEN),
                scale_factor=(1.0, 2.0, 2.0),
                sparse_params=None,
                return_dict=True,
            )
        assert len(backend_calls) == 3, "context-wrapped raw call should hit all 3 attention layers"

        stage.forward(_make_batch(), _make_fastvideo_args(arch))
        assert len(backend_calls) == 3 + 3 * len(DMD_STEPS)
    finally:
        _cached_get_attn_backend.cache_clear()
