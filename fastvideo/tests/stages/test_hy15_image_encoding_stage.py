# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``Hy15ImageEncodingStage``.

The failure this guards against is quiet: with no reference image conditioning,
HunyuanVideo 1.5 still trains and still samples, it just ignores the image. So
these assert the channel layout rather than only that a tensor came back.

Ordering matters and is invisible in T2V. ``DenoisingStage`` appends
``image_latent`` as one block, so the 65 channel input is
``[latents 32][conditioning 32][mask 1]``. For T2V the last 33 are all zero and
either order produces the same tensor, which is why a swapped layout would go
unnoticed until an image is actually supplied.
"""
from __future__ import annotations

from types import SimpleNamespace

import PIL.Image
import pytest
import torch

from fastvideo.pipelines.stages.image_encoding import Hy15ImageEncodingStage

LATENT_SHAPE = (1, 32, 5, 8, 12)  # B, C, T, H, W
REFERENCE_IMAGE = PIL.Image.new("RGB", (LATENT_SHAPE[4] * 8, LATENT_SHAPE[3] * 8), (120, 60, 30))
VISION_TOKENS = 729
VISION_DIM = 1152


class _FakeSiglip(torch.nn.Module):
    """Returns a constant non-zero hidden state, which is all the stage reads."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.ones(1))

    def forward(self, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.full((batch, VISION_TOKENS, VISION_DIM), 0.5))


class _FakeProcessor:

    def preprocess(self, images, **kwargs):
        return SimpleNamespace(to=lambda **_: {"pixel_values": torch.zeros(1, 3, 384, 384)})


class _FakeVAE(torch.nn.Module):
    """Encodes to a constant so the first frame is distinguishable from zero."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=2.0)

    def encode(self, pixels: torch.Tensor):
        _, _, _, height, width = pixels.shape
        latent = torch.full((1, LATENT_SHAPE[1], 1, height // 8, width // 8), 3.0)
        return SimpleNamespace(mode=lambda: latent)

    def to(self, *args, **kwargs):
        return self


def _batch(pil_image):
    return SimpleNamespace(
        pil_image=pil_image,
        raw_latent_shape=LATENT_SHAPE,
        image_embeds=[],
        image_latent=None,
        video_latent=None,
        height=LATENT_SHAPE[3] * 8,
        width=LATENT_SHAPE[4] * 8,
    )


def _args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(vae_precision="fp32"),
        disable_autocast=True,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
    )


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    """Pin the stage to CPU so the fakes and the real code agree on device.

    ``get_local_torch_device`` resolves to MPS on macOS and CUDA elsewhere, and
    nothing here needs an accelerator.
    """
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.image_encoding.get_local_torch_device",
        lambda: torch.device("cpu"),
    )


def _i2v_stage():
    return Hy15ImageEncodingStage(
        image_encoder=_FakeSiglip(),
        image_processor=_FakeProcessor(),
        vae=_FakeVAE(),
    )


class TestTextToVideoPath:
    """No reference image: the transformer must still see the T2V layout."""

    def test_image_embeds_are_zero(self) -> None:
        batch = Hy15ImageEncodingStage().forward(_batch(None), _args())

        assert len(batch.image_embeds) == 1
        assert batch.image_embeds[0].shape == (1, VISION_TOKENS, VISION_DIM)
        # The transformer keys its T2V branch off this being all zero.
        assert torch.all(batch.image_embeds[0] == 0)

    def test_conditioning_stays_on_the_video_latent_slot(self) -> None:
        batch = Hy15ImageEncodingStage().forward(_batch(None), _args())

        # The super-resolution stages read video_latent directly, so T2V keeps
        # populating it rather than switching to image_latent.
        assert batch.video_latent is not None
        assert batch.video_latent.shape == (1, 1) + LATENT_SHAPE[2:]
        assert torch.all(batch.video_latent == 0)
        assert batch.image_latent is None


class TestImageToVideoPath:

    def test_image_embeds_come_from_siglip(self) -> None:
        batch = _i2v_stage().forward(_batch(REFERENCE_IMAGE), _args())

        assert batch.image_embeds[0].shape == (1, VISION_TOKENS, VISION_DIM)
        # Non-zero is the whole point: all-zero would re-select the T2V branch.
        assert not torch.all(batch.image_embeds[0] == 0)

    def test_channel_layout_is_conditioning_then_mask(self) -> None:
        batch = _i2v_stage().forward(_batch(REFERENCE_IMAGE), _args())

        assert batch.image_latent.shape == (1, LATENT_SHAPE[1] + 1) + LATENT_SHAPE[2:]

        conditioning = batch.image_latent[:, :LATENT_SHAPE[1]]
        mask = batch.image_latent[:, LATENT_SHAPE[1]:]

        # A swapped layout would put a 1-wide block first, so check that the
        # trailing channel is the mask and not part of the conditioning.
        assert torch.all(mask[:, :, 0] == 1.0)
        assert torch.all(mask[:, :, 1:] == 0.0)
        assert torch.all(conditioning[:, :, 0] != 0)
        assert torch.all(conditioning[:, :, 1:] == 0)

    def test_scaling_factor_is_applied(self) -> None:
        batch = _i2v_stage().forward(_batch(REFERENCE_IMAGE), _args())

        # Fake VAE emits 3.0 and declares scaling_factor 2.0.
        assert torch.allclose(batch.image_latent[:, :LATENT_SHAPE[1], 0],
                              torch.full((1, LATENT_SHAPE[1]) + LATENT_SHAPE[3:], 6.0))

    def test_video_latent_left_unset(self) -> None:
        batch = _i2v_stage().forward(_batch(REFERENCE_IMAGE), _args())

        # DenoisingStage checks video_latent first and would append the mask
        # ahead of the conditioning, so this slot has to stay empty.
        assert batch.video_latent is None

    def test_missing_encoder_is_an_explicit_error(self) -> None:
        with pytest.raises(ValueError, match="image encoder"):
            Hy15ImageEncodingStage().forward(_batch(REFERENCE_IMAGE), _args())
