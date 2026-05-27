# SPDX-License-Identifier: Apache-2.0
"""Regression test: SR latent prep must invalidate the static-packed layout.

The base latent prep stage (`MagiHumanLatentPreparationStage`) precomputes
``batch.magi_static_packed_layout`` for the BASE-resolution latent and stashes
it on the batch so the base denoising loop can reuse it across all denoising
steps (C4 perf optimization, commit 4190c720).

The SR latent prep stage (`MagiHumanSRLatentPreparationStage`) upsamples
``batch.latents`` to a much larger spatial grid (e.g. 256x480 -> 512x896 for
SR-540p), which changes the layout's video_token_num / video_coords / video_mm.
Without invalidating the layout, the SR denoising loop reuses the stale
base-sized layout and crashes inside ``MagiHumanDiT.adapter`` with::

    IndexError: The shape of the mask [3243] at index 0 does not match
    the shape of the indexed tensor [11771, 3584] at index 0

See git f1eeb630 for the fix and a commit-message-level explanation.

This is a pure logic test — no GPU, no model load, no upstream daVinci-MagiHuman
clone needed. It runs in the default CI suite.
"""
from __future__ import annotations

import torch

from fastvideo.pipelines.basic.magi_human.stages.sr_latent_preparation import (
    MagiHumanSRLatentPreparationStage,
    ZeroSNRDDPMDiscretization,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


def _make_stage() -> MagiHumanSRLatentPreparationStage:
    """Bypass __init__: only set the fields the T2V forward() path reads."""
    stage = MagiHumanSRLatentPreparationStage.__new__(
        MagiHumanSRLatentPreparationStage)
    # vae + video_processor are only used by `_encode_image` (TI2V path);
    # T2V skips that branch when `batch.image_latent is None`.
    stage.vae = None
    stage.video_processor = None
    stage.vae_stride = (4, 16, 16)
    stage.patch_size = (1, 2, 2)
    # `noise_value=0` skips the sigma noise-injection branch — keeps the
    # test deterministic and avoids depending on torch.randn.
    stage.noise_value = 0
    stage.sr_audio_noise_scale = 0.7
    stage.sr_height = 512
    stage.sr_width = 896
    stage.sigmas = ZeroSNRDDPMDiscretization()(
        1000, do_append_zero=False, flip=True)
    return stage


def test_sr_latent_prep_invalidates_static_packed_layout():
    stage = _make_stage()

    base_latent = torch.randn(1, 48, 7, 16, 30, dtype=torch.float32)
    audio = torch.randn(1, 26, 64, dtype=torch.float32)

    batch = ForwardBatch(data_type="video")
    batch.latents = base_latent
    batch.audio_latents = audio

    sentinel = object()
    batch.magi_static_packed_layout = sentinel  # type: ignore[attr-defined]

    out = stage.forward(batch, fastvideo_args=None)  # type: ignore[arg-type]

    assert out is batch
    # Sanity: SR actually upsampled to a different spatial grid.
    assert out.latents.shape[-1] != base_latent.shape[-1]
    assert out.latents.shape[-2] != base_latent.shape[-2]
    # The bug-fix invariant: the stale base-sized layout is gone, so the
    # SR denoising loop's `getattr(batch, "magi_static_packed_layout", None)`
    # falls back to None and `build_static_packed_inputs` rebuilds the
    # layout from the new SR-sized latent.
    assert getattr(out, "magi_static_packed_layout", "<missing>") is None
