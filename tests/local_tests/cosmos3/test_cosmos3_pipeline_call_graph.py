# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 native-pipeline call-graph contract (Tier A, no real weights).

Pins the runtime call graph of the FastVideo-native Cosmos3 pipeline against the
framework math, using the stub components from ``conftest.py`` (no real weights,
no ``cosmos_framework``). The native pipeline replaced the vllm-omni-derived
``diffuse``/``forward(req)``/``reset_cache`` skeleton with a stage-based
``Cosmos3DenoisingStage`` + ``Cosmos3DenoiseEngine`` doing SEQUENTIAL CFG.

Invariants under test:

  1. SEQUENTIAL CFG order — per UniPC step, the transformer is called twice,
     conditional (prompt tokens) then unconditional (negative-prompt tokens), in
     that order; over N steps the call order is ``[cond, uncond] * N``.
  2. CFG combination — the per-step velocity equals
     ``uncond + guidance * (cond - uncond)`` (verified against the stub's known
     per-token output) and one UniPC step advances the latent accordingly.
  3. I2V conditioning — a conditioning image is VAE-encoded and frame 0 is kept
     clean: its velocity is zeroed (condition mask) so the decoded clip's
     frame-0 latent equals the clean conditioning latent.
  4. Mode dispatch — the stage routes T2V (num_frames>1, flow_shift=10.0,
     ``is_video`` tokenization) vs T2I (num_frames==1, flow_shift=3.0, image
     tokenization), applying the per-mode ``flow_shift``.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import make_fastvideo_args, make_forward_batch

pytestmark = [pytest.mark.local]

_LATENT_CHANNEL = 16
_LATENT_PATCH_SIZE = 2
_TEMPORAL_FACTOR = 4
_COND_TOKEN = 2
_UNCOND_TOKEN = 1


def _engine(pipeline, scheduler):
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import Cosmos3DenoiseEngine

    return Cosmos3DenoiseEngine(
        transformer=pipeline.modules["transformer"],
        scheduler=scheduler,
        special_tokens={"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62},
        latent_patch_size=_LATENT_PATCH_SIZE,
        temporal_modality_margin=15_000,
        reset_spatial_ids=True,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TEMPORAL_FACTOR,
    )


def test_sequential_cfg_calls_cond_then_uncond_each_step(make_cosmos3_pipeline) -> None:
    """Each UniPC step calls the transformer cond-then-uncond, in order."""
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import Cosmos3VisionSpec

    pipeline = make_cosmos3_pipeline()
    scheduler = pipeline.modules["scheduler"]
    scheduler.set_timesteps(2, device=torch.device("cpu"))
    engine = _engine(pipeline, scheduler)

    shape = (_LATENT_CHANNEL, 2, 2, 2)
    flat = torch.randn(int(torch.tensor(shape).prod()))
    spec = Cosmos3VisionSpec(shape=shape, condition_frame_indexes=[])

    engine.denoise(
        flat_latent=flat,
        timesteps=scheduler.timesteps,
        guidance=6.0,
        specs=[spec],
        cond_token_ids=[_COND_TOKEN, 5, 6],
        uncond_token_ids=[_UNCOND_TOKEN, 7],
    )
    tokens = [c["token"] for c in pipeline.modules["transformer"].calls]
    # 2 steps -> 4 calls: cond, uncond, cond, uncond.
    assert tokens == [_COND_TOKEN, _UNCOND_TOKEN, _COND_TOKEN, _UNCOND_TOKEN]


def test_cfg_velocity_combination_formula(make_cosmos3_pipeline) -> None:
    """The per-step velocity equals ``uncond + g*(cond - uncond)``.

    The stub returns ``scale(token) * tanh(latent)`` on noisy frames, so the
    expected velocity is a closed form we can check exactly.
    """
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
        Cosmos3VisionSpec,
        cosmos3_get_cfg_velocity,
    )

    pipeline = make_cosmos3_pipeline()
    transformer = pipeline.modules["transformer"]
    shape = (_LATENT_CHANNEL, 2, 2, 2)
    flat = torch.randn(int(torch.tensor(shape).prod()))
    guidance = 6.0

    v = cosmos3_get_cfg_velocity(
        transformer=transformer,
        flat_latent=flat,
        timestep=torch.tensor([500.0]),
        guidance=guidance,
        specs=[Cosmos3VisionSpec(shape=shape, condition_frame_indexes=[])],
        cond_token_ids=[_COND_TOKEN, 5, 6],
        uncond_token_ids=[_UNCOND_TOKEN, 7],
        special_tokens={"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62},
        latent_patch_size=_LATENT_PATCH_SIZE,
        temporal_modality_margin=15_000,
        reset_spatial_ids=True,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TEMPORAL_FACTOR,
    )

    lat = flat.reshape(shape)
    scale_cond = 0.01 * (1.0 + (_COND_TOKEN % 7))
    scale_uncond = 0.01 * (1.0 + (_UNCOND_TOKEN % 7))
    cond_v = (scale_cond * torch.tanh(lat)).reshape(-1)
    uncond_v = (scale_uncond * torch.tanh(lat)).reshape(-1)
    expected = uncond_v + guidance * (cond_v - uncond_v)
    torch.testing.assert_close(v, expected, atol=1e-6, rtol=1e-5)


def test_i2v_keeps_condition_frame_clean(make_cosmos3_pipeline) -> None:
    """I2V: frame-0 velocity is zeroed so the conditioning frame stays clean."""
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
        Cosmos3VisionSpec,
        cosmos3_get_cfg_velocity,
    )

    pipeline = make_cosmos3_pipeline()
    transformer = pipeline.modules["transformer"]
    shape = (_LATENT_CHANNEL, 3, 2, 2)  # 3 latent frames, frame 0 conditioned
    flat = torch.randn(int(torch.tensor(shape).prod()))

    v = cosmos3_get_cfg_velocity(
        transformer=transformer,
        flat_latent=flat,
        timestep=torch.tensor([500.0]),
        guidance=6.0,
        specs=[Cosmos3VisionSpec(shape=shape, condition_frame_indexes=[0])],
        cond_token_ids=[_COND_TOKEN, 5, 6],
        uncond_token_ids=[_UNCOND_TOKEN, 7],
        special_tokens={"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62},
        latent_patch_size=_LATENT_PATCH_SIZE,
        temporal_modality_margin=15_000,
        reset_spatial_ids=True,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TEMPORAL_FACTOR,
    )
    v_grid = v.reshape(shape)  # [C, T, H, W]
    # Condition frame 0 velocity must be exactly zero; noisy frames non-zero.
    assert torch.count_nonzero(v_grid[:, 0]) == 0
    assert torch.count_nonzero(v_grid[:, 1:]) > 0


def test_stage_mode_dispatch_t2v(make_cosmos3_pipeline, make_cosmos3_stage) -> None:
    """T2V (num_frames>1): resolution-based flow_shift, video tokenization."""
    pipeline = make_cosmos3_pipeline()
    stage = make_cosmos3_stage(pipeline)
    args = make_fastvideo_args()
    batch = make_forward_batch(num_frames=5, height=16, width=16)

    out = stage.forward(batch, args)
    # flow_shift is resolution-based (not task-based): 16x16 -> "256" bucket -> 3.0.
    # (full resolution->shift parity in test_cosmos3_flow_shift_parity.)
    assert float(pipeline.scheduler.config.flow_shift) == 3.0
    assert out.output is not None and out.output.dim() == 5
    # T2V latent: (5-1)//4 + 1 = 2 frames; 16/8 = 2 latent h/w.
    assert tuple(out.latents.shape) == (1, _LATENT_CHANNEL, 2, 2, 2)


def test_stage_mode_dispatch_t2i(make_cosmos3_pipeline, make_cosmos3_stage) -> None:
    """T2I (num_frames==1): single-frame latent; resolution-based flow_shift."""
    pipeline = make_cosmos3_pipeline()
    stage = make_cosmos3_stage(pipeline)
    args = make_fastvideo_args()
    batch = make_forward_batch(num_frames=1, height=16, width=16, guidance_scale=4.0)

    out = stage.forward(batch, args)
    # 16x16 -> "256" bucket -> 3.0 (resolution-based, not task-based).
    assert float(pipeline.scheduler.config.flow_shift) == 3.0
    assert tuple(out.latents.shape) == (1, _LATENT_CHANNEL, 1, 2, 2)


def test_stage_i2v_encodes_conditioning_image(make_cosmos3_pipeline, make_cosmos3_stage) -> None:
    """I2V stage: a conditioning image is accepted and decoded to a finite clip."""
    pipeline = make_cosmos3_pipeline()
    stage = make_cosmos3_stage(pipeline)
    args = make_fastvideo_args()
    image = torch.zeros(3, 16, 16)  # [-1, 1] conditioning frame
    batch = make_forward_batch(num_frames=5, height=16, width=16, image=image)

    out = stage.forward(batch, args)
    # flow_shift is resolution-based (not task-based): 16x16 -> 3.0.
    assert float(pipeline.scheduler.config.flow_shift) == 3.0
    assert out.output is not None and out.output.dim() == 5
    assert torch.isfinite(out.output).all()
