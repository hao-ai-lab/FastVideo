# SPDX-License-Identifier: Apache-2.0
"""Regression tests for two ``Kandinsky6ImageEncodingStage`` gaps against the diffusers reference's
``encode_i2va_first_frame``:

* the conditioning image must be cover-resized (aspect-preserving) and centre-cropped to (height,
  width), not stretched to fit -- ``_cover_resize_dims`` replicates ``scale=min(src_h/h, src_w/w)``;
  ``resize=(int(src_h/scale), int(src_w/scale))``, ``resize -> [h,w] centre crop``;
* a conditioning image must broadcast across ``num_videos_per_prompt>1`` instead of crashing
  ``torch.cat`` with a batch-size mismatch.

Pure CPU harness: a fake VAE returns zeros of the right shape instead of really encoding, so no
weights are needed.
"""
from __future__ import annotations

import types

import numpy as np
import PIL.Image
import pytest
import torch

from fastvideo.pipelines.stages import kandinsky6 as k6_stages
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6ImageEncodingStage


@pytest.fixture()
def cpu_device(monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setattr(k6_stages, "get_local_torch_device", lambda: cpu)
    return cpu


# --------------------------------------------------------------------------------------------------
# P10: cover-resize + centre crop
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_h,src_w,height,width,expected",
    [
        # The audit's worked example: 1024x1024 source -> 512x768 target. scale=min(1024/512,
        # 1024/768)=1.333(3); resize to (int(1024/1.3333), int(1024/1.3333)) = (768, 768).
        (1024, 1024, 512, 768, (768, 768)),
        # A portrait target from a landscape source: scale=min(480/512, 864/768)=0.9375; resize to
        # (int(480/0.9375), int(864/0.9375)) = (512, 921).
        (480, 864, 512, 768, (512, 921)),
        # Source already exactly the target size: no-op.
        (512, 768, 512, 768, (512, 768)),
    ],
)
def test_cover_resize_dims_matches_the_diffusers_formula(src_h, src_w, height, width, expected):
    assert Kandinsky6ImageEncodingStage._cover_resize_dims(src_h, src_w, height, width) == expected


def test_cover_resize_dims_does_not_underflow_the_target_on_float_error():
    # A 510x510 source at a 480x832 target gives int(510 / (510/832)) == 831 (one pixel short of
    # 832) because of float error in the scale round-trip, which would mean a negative centre-crop
    # offset. Both axes must be >= the target.
    new_h, new_w = Kandinsky6ImageEncodingStage._cover_resize_dims(510, 510, 480, 832)
    assert new_h >= 480
    assert new_w >= 832


def test_preprocess_crops_the_top_and_bottom_instead_of_stretching_the_whole_image():
    # 1024x1024 source with a marker confined to the first few rows only. A cover-resize to (768,768)
    # + centre crop to (512,768) keeps rows [128:640] of the resized image (the audit: "keeps a
    # 1024x683 centre region") -- far from the marker, so it must be cropped away entirely. The old
    # code (plain stretch to (512,768), no crop) would keep *some* of every source row, including the
    # marker, in the output's top rows.
    src = torch.zeros(1, 3, 1024, 1024)
    src[:, :, :4, :] = 1.0
    src = src * 2 - 1  # move to [-1,1] with real negative values so _preprocess skips its [0,1] guess

    out = Kandinsky6ImageEncodingStage._preprocess(src, height=512, width=768)

    assert out.shape[-2:] == (512, 768)
    assert float(out.max()) < 0.0  # the positive marker did not survive the crop


def test_preprocess_pil_path_also_crops_not_stretches():
    arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    arr[:4, :, :] = 255  # marker confined to the first 4 rows, full width
    src = PIL.Image.fromarray(arr)

    out = Kandinsky6ImageEncodingStage._preprocess(src, height=512, width=768)

    assert out.shape[-2:] == (512, 768)
    assert float(out.max()) < 0.0


def test_preprocess_same_aspect_ratio_needs_no_crop_and_matches_a_plain_resize():
    # A source already at the target aspect ratio: cover-resize lands exactly on (height, width), so
    # there is nothing to crop and the result should equal a plain resize.
    src = torch.rand(1, 3, 256, 384) * 2 - 1  # 2:3 aspect, same as height=512,width=768

    out = Kandinsky6ImageEncodingStage._preprocess(src, height=512, width=768)

    expected = torch.nn.functional.interpolate(src, size=(512, 768), mode="bilinear", antialias=True)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5)


# --------------------------------------------------------------------------------------------------
# P11: image conditioning + num_videos_per_prompt > 1
# --------------------------------------------------------------------------------------------------


class _FakeVAEOutput:

    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def sample(self, generator=None) -> torch.Tensor:
        return self._latent


class _FakeVAE:
    scaling_factor = 1.0
    use_tiling = False

    def to(self, device):
        return self

    def encode(self, image: torch.Tensor) -> _FakeVAEOutput:
        b, _c, t, h, w = image.shape
        return _FakeVAEOutput(torch.zeros(b, 4, t, h, w))


def _image_encoding_stage() -> Kandinsky6ImageEncodingStage:
    stage = Kandinsky6ImageEncodingStage.__new__(Kandinsky6ImageEncodingStage)
    stage.vae = _FakeVAE()
    return stage


def _fastvideo_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_loaded={"vae": True},
        pipeline_config=types.SimpleNamespace(vae_precision="fp32"),
        disable_autocast=True,
        vae_cpu_offload=False,
    )


def test_image_plus_multiple_videos_per_prompt_no_longer_crashes(cpu_device):
    stage = _image_encoding_stage()
    num_videos_per_prompt = 2
    latents = torch.zeros(num_videos_per_prompt, 3, 8, 8, 4)  # [B,T,H,W,C], visual_cond off (C==4)
    batch = types.SimpleNamespace(
        pil_image=PIL.Image.new("RGB", (8, 8)),
        height=8,
        width=8,
        generator=None,
        latents=latents,
        extra={},
    )

    out = stage.forward(batch, _fastvideo_args())

    assert out.latents.shape == (num_videos_per_prompt, 4, 8, 8, 4)
    assert out.extra["kandinsky6_tail_cond_active"] is True
    assert out.extra["kandinsky6_visual_token_type_ids"].shape == (num_videos_per_prompt, 4)


def test_image_conditioning_still_works_for_a_single_video(cpu_device):
    stage = _image_encoding_stage()
    latents = torch.zeros(1, 3, 8, 8, 4)
    batch = types.SimpleNamespace(
        pil_image=PIL.Image.new("RGB", (8, 8)),
        height=8,
        width=8,
        generator=None,
        latents=latents,
        extra={},
    )

    out = stage.forward(batch, _fastvideo_args())

    assert out.latents.shape == (1, 4, 8, 8, 4)


def test_image_batch_that_is_neither_one_nor_the_video_batch_size_raises_a_clear_error(cpu_device, monkeypatch):
    stage = _image_encoding_stage()

    class _MismatchedVAE(_FakeVAE):

        def encode(self, image):
            # Pretend the VAE itself produced 3 reference frames -- neither 1 (broadcastable) nor
            # matching the video batch size of 2.
            b, _c, t, h, w = image.shape
            return _FakeVAEOutput(torch.zeros(3, 4, t, h, w))

    stage.vae = _MismatchedVAE()
    latents = torch.zeros(2, 3, 8, 8, 4)
    batch = types.SimpleNamespace(
        pil_image=PIL.Image.new("RGB", (8, 8)),
        height=8,
        width=8,
        generator=None,
        latents=latents,
        extra={},
    )

    with pytest.raises(ValueError, match="not broadcastable"):
        stage.forward(batch, _fastvideo_args())
