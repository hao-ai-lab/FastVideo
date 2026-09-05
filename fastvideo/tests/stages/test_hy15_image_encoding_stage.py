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

import fastvideo.models.vaes.hunyuan15vae as hunyuan15vae
from fastvideo.configs.pipelines.hunyuan15 import (Hunyuan15I2V480PStepDistilledConfig, Hunyuan15I2V720PConfig,
                                                   Hunyuan15SR1080PConfig, Hunyuan15T2V480PConfig,
                                                   Hunyuan15T2V720PConfig)
from fastvideo.models.vaes.hunyuan15vae import AutoencoderKLHunyuanVideo15
from fastvideo.pipelines.stages.image_encoding import Hy15ImageEncodingStage

VAE_SPATIAL_COMPRESSION_RATIO = 16
LATENT_SHAPE = (1, 32, 5, 4, 6)  # B, C, T, H, W
REFERENCE_IMAGE = PIL.Image.new(
    "RGB",
    (LATENT_SHAPE[4] * VAE_SPATIAL_COMPRESSION_RATIO, LATENT_SHAPE[3] * VAE_SPATIAL_COMPRESSION_RATIO),
    (120, 60, 30),
)
VISION_TOKENS = 729
VISION_DIM = 1152


class _FakeSiglip(torch.nn.Module):
    """Returns a constant non-zero hidden state, which is all the stage reads."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.ones(1))
        self.forward_calls = 0

    def forward(self, pixel_values: torch.Tensor):
        self.forward_calls += 1
        batch = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.full((batch, VISION_TOKENS, VISION_DIM), 0.5))


class _FakeProcessor:

    def preprocess(self, images, **kwargs):
        return SimpleNamespace(to=lambda **_: {"pixel_values": torch.zeros(1, 3, 384, 384)})


class _ConstantEncoder(torch.nn.Module):
    """Small encoder used inside the real HunyuanVideo 1.5 VAE wrapper."""

    def __init__(self, out_channels: int, spatial_compression_ratio: int, **kwargs) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.spatial_compression_ratio = spatial_compression_ratio
        self.forward_calls = 0

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        batch_size, _, _, height, width = pixels.shape
        return torch.full(
            (
                batch_size,
                self.out_channels,
                1,
                height // self.spatial_compression_ratio,
                width // self.spatial_compression_ratio,
            ),
            3.0,
            device=pixels.device,
            dtype=pixels.dtype,
        )


class _UnusedDecoder(torch.nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


def _batch(pil_image, batch_size: int = 1):
    raw_latent_shape = (batch_size, ) + LATENT_SHAPE[1:]
    return SimpleNamespace(
        pil_image=pil_image,
        raw_latent_shape=raw_latent_shape,
        image_embeds=[],
        image_latent=None,
        video_latent=None,
        height=LATENT_SHAPE[3] * VAE_SPATIAL_COMPRESSION_RATIO,
        width=LATENT_SHAPE[4] * VAE_SPATIAL_COMPRESSION_RATIO,
    )


def _args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_precision="fp32",
            vae_tiling=True,
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(spatial_compression_ratio=VAE_SPATIAL_COMPRESSION_RATIO)),
        ),
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


@pytest.fixture
def lightweight_vae_factory(monkeypatch):
    monkeypatch.setattr(hunyuan15vae, "HunyuanVideo15Encoder3D", _ConstantEncoder)
    monkeypatch.setattr(hunyuan15vae, "HunyuanVideo15Decoder3D", _UnusedDecoder)

    def build(pipeline_config=None):
        if pipeline_config is None:
            pipeline_config = Hunyuan15I2V480PStepDistilledConfig()
        return AutoencoderKLHunyuanVideo15(pipeline_config.vae_config)

    return build


@pytest.fixture
def i2v_stage(lightweight_vae_factory):
    return Hy15ImageEncodingStage(
        image_encoder=_FakeSiglip(),
        image_processor=_FakeProcessor(),
        vae=lightweight_vae_factory(),
    )


@pytest.mark.parametrize(
    "config_cls",
    [Hunyuan15I2V480PStepDistilledConfig, Hunyuan15I2V720PConfig],
)
def test_i2v_configs_build_the_vae_encoder_and_decoder(config_cls, lightweight_vae_factory) -> None:
    pipeline_config = config_cls()
    vae = lightweight_vae_factory(pipeline_config)

    assert pipeline_config.vae_config.load_encoder is True
    assert pipeline_config.vae_config.load_decoder is True
    assert pipeline_config.text_encoder_configs[0].arch_config.output_hidden_states is True
    assert isinstance(vae.encoder, _ConstantEncoder)
    assert isinstance(vae.decoder, _UnusedDecoder)


@pytest.mark.parametrize(
    "config_cls",
    [Hunyuan15I2V480PStepDistilledConfig, Hunyuan15I2V720PConfig],
)
def test_i2v_configs_reject_disabling_the_vae_encoder(config_cls) -> None:
    pipeline_config = config_cls()
    pipeline_config.update_config_from_dict({"vae_config.load_encoder": False})

    with pytest.raises(ValueError, match="requires the VAE encoder"):
        pipeline_config.check_pipeline_config()


@pytest.mark.parametrize(
    "config_cls",
    [Hunyuan15T2V480PConfig, Hunyuan15T2V720PConfig, Hunyuan15SR1080PConfig],
)
def test_non_i2v_configs_keep_the_vae_encoder_disabled(config_cls) -> None:
    assert config_cls().vae_config.load_encoder is False


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

    def test_placeholders_follow_the_latent_batch(self) -> None:
        batch_size = 2
        batch = Hy15ImageEncodingStage().forward(_batch(None, batch_size=batch_size), _args())

        assert batch.image_embeds[0].shape == (batch_size, VISION_TOKENS, VISION_DIM)
        assert batch.video_latent.shape == (batch_size, 1) + LATENT_SHAPE[2:]


class TestImageToVideoPath:

    def test_image_embeds_come_from_siglip(self, i2v_stage) -> None:
        batch = i2v_stage.forward(_batch(REFERENCE_IMAGE), _args())

        assert batch.image_embeds[0].shape == (1, VISION_TOKENS, VISION_DIM)
        # Non-zero is the whole point: all-zero would re-select the T2V branch.
        assert not torch.all(batch.image_embeds[0] == 0)

    def test_channel_layout_is_conditioning_then_mask(self, i2v_stage) -> None:
        batch = i2v_stage.forward(_batch(REFERENCE_IMAGE), _args())

        assert batch.image_latent.shape == (1, LATENT_SHAPE[1] + 1) + LATENT_SHAPE[2:]

        conditioning = batch.image_latent[:, :LATENT_SHAPE[1]]
        mask = batch.image_latent[:, LATENT_SHAPE[1]:]

        # A swapped layout would put a 1-wide block first, so check that the
        # trailing channel is the mask and not part of the conditioning.
        assert torch.all(mask[:, :, 0] == 1.0)
        assert torch.all(mask[:, :, 1:] == 0.0)
        assert torch.all(conditioning[:, :, 0] != 0)
        assert torch.all(conditioning[:, :, 1:] == 0)

    def test_scaling_factor_is_applied(self, i2v_stage) -> None:
        batch = i2v_stage.forward(_batch(REFERENCE_IMAGE), _args())

        expected = 3.0 * i2v_stage.vae.config.scaling_factor
        assert torch.allclose(batch.image_latent[:, :LATENT_SHAPE[1], 0],
                              torch.full((1, LATENT_SHAPE[1]) + LATENT_SHAPE[3:], expected))

    def test_video_latent_left_unset(self, i2v_stage) -> None:
        batch = i2v_stage.forward(_batch(REFERENCE_IMAGE), _args())

        # DenoisingStage checks video_latent first and would append the mask
        # ahead of the conditioning, so this slot has to stay empty.
        assert batch.video_latent is None

    def test_single_reference_conditioning_repeats_to_the_latent_batch(self, i2v_stage) -> None:
        batch_size = 2
        batch = i2v_stage.forward(_batch(REFERENCE_IMAGE, batch_size=batch_size), _args())

        expected_shape = (batch_size, LATENT_SHAPE[1] + 1) + LATENT_SHAPE[2:]
        assert batch.image_latent.shape == expected_shape
        assert batch.image_embeds[0].shape == (batch_size, VISION_TOKENS, VISION_DIM)
        assert torch.equal(batch.image_latent[0], batch.image_latent[1])
        assert torch.equal(batch.image_embeds[0][0], batch.image_embeds[0][1])

        conditioning = batch.image_latent[:, :LATENT_SHAPE[1]]
        mask = batch.image_latent[:, LATENT_SHAPE[1]:]
        assert torch.all(conditioning[:, :, 0] != 0)
        assert torch.all(conditioning[:, :, 1:] == 0)
        assert torch.all(mask[:, :, 0] == 1)
        assert torch.all(mask[:, :, 1:] == 0)

        assert i2v_stage.vae.encoder.forward_calls == 1
        assert i2v_stage.image_encoder.forward_calls == 1

        noise_latents = torch.zeros((batch_size, ) + LATENT_SHAPE[1:])
        assert torch.cat([noise_latents, batch.image_latent], dim=1).shape == (
            batch_size,
            LATENT_SHAPE[1] * 2 + 1,
        ) + LATENT_SHAPE[2:]

    def test_missing_encoder_is_an_explicit_error(self) -> None:
        with pytest.raises(ValueError, match="image encoder"):
            Hy15ImageEncodingStage().forward(_batch(REFERENCE_IMAGE), _args())
