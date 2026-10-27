# SPDX-License-Identifier: Apache-2.0
"""Regression tests for two ``Kandinsky6LatentPreparationStage.forward`` gaps against the diffusers
reference (``pipeline_kandinsky6_ti2va.py``):

* ``num_frames`` must round to the nearest ``4k+1`` the same way the diffusers reference's
  ``num_frames // temporal_ratio * temporal_ratio + 1`` does: floor for a non-exact multiple of
  ``temporal_ratio``, but round *up* for an exact multiple (120 -> 121, not 117);
* the audio latent length must follow the request's ``fps`` (``sample_fps``), not always the
  pipeline's static default.

Pure CPU harness: a fake transformer/scheduler stand in for real ones (only ``in_visual_dim``/
``in_audio_dim``/``visual_cond`` and an optional ``init_noise_sigma`` are read), so no weights are
needed.
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo.pipelines.stages import kandinsky6 as k6_stages
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6LatentPreparationStage


@pytest.fixture()
def cpu_device(monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setattr(k6_stages, "get_local_torch_device", lambda: cpu)
    return cpu


def _fastvideo_args(*, sample_fps: float = 24.0) -> types.SimpleNamespace:
    pipeline_config = types.SimpleNamespace(
        vae_config=types.SimpleNamespace(
            arch_config=types.SimpleNamespace(temporal_compression_ratio=4, spatial_compression_ratio=8)),
        dit_config=types.SimpleNamespace(
            arch_config=types.SimpleNamespace(patch_size=(1, 2, 2), in_visual_dim=16, in_audio_dim=20)),
        dit_precision="fp32",
        sample_fps=sample_fps,
        audio_sample_rate=44100,
        audio_downsample_factor=1024,
    )
    return types.SimpleNamespace(pipeline_config=pipeline_config)


def _stage() -> Kandinsky6LatentPreparationStage:
    stage = Kandinsky6LatentPreparationStage.__new__(Kandinsky6LatentPreparationStage)
    stage.scheduler = types.SimpleNamespace()  # no init_noise_sigma attribute
    stage.transformer = types.SimpleNamespace(in_visual_dim=16, in_audio_dim=20, visual_cond=False)
    return stage


def _batch(num_frames: int, *, fps=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        height=512,
        width=768,
        num_frames=num_frames,
        prompt="a chef chops vegetables",
        num_videos_per_prompt=1,
        latents=None,
        audio_latents=None,
        generator=None,
        fps=fps,
    )


# --------------------------------------------------------------------------------------------------
# P5: num_frames rounds to the nearest 4k+1 (floor for a non-exact multiple, round up for an exact one)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("requested,expected", [
    (120, 121),  # exact multiple of 4 -> rounds up, not down
    (96, 97),  # ditto
    (121, 121),  # already valid -> unchanged
    (60, 61),  # exact multiple -> rounds up
    (124, 125),  # exact multiple -> rounds up
    (119, 117),  # non-exact multiple -> floors, same as before
    (122, 121),  # non-exact multiple -> floors, same as before
])
def test_num_frames_rounds_to_nearest_4k_plus_1(cpu_device, requested, expected):
    stage = _stage()
    batch = _batch(requested)

    out = stage.forward(batch, _fastvideo_args())

    assert out.num_frames == expected
    assert out.latents.shape[1] == (expected - 1) // 4 + 1


# --------------------------------------------------------------------------------------------------
# P17: audio latent length follows batch.fps
# --------------------------------------------------------------------------------------------------


def test_audio_latent_length_follows_the_requested_fps_not_the_pipeline_default(cpu_device):
    stage = _stage()
    batch = _batch(121, fps=30)

    out = stage.forward(batch, _fastvideo_args(sample_fps=24.0))

    # T_lat=31 ((121-1)//4+1); pixel_frames=(31-1)*4+1=121; ceil(121/30*44100/1024) == 174, matching
    # the diffusers reference's sample_fps=30 case exactly -- a fixed 24fps default would give 218.
    assert out.audio_latents.shape[1] == 174


def test_audio_latent_length_falls_back_to_the_pipeline_default_fps_when_the_request_has_none(cpu_device):
    stage = _stage()
    batch = _batch(121, fps=None)

    out = stage.forward(batch, _fastvideo_args(sample_fps=24.0))

    assert out.audio_latents.shape[1] == 218


def test_audio_latent_length_uses_the_first_entry_of_a_per_prompt_fps_list(cpu_device):
    stage = _stage()
    batch = _batch(121, fps=[30, 30])

    out = stage.forward(batch, _fastvideo_args(sample_fps=24.0))

    assert out.audio_latents.shape[1] == 174
