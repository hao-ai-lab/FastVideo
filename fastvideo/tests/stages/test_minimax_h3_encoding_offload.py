# SPDX-License-Identifier: Apache-2.0
"""CUDA regressions for shared H3 encoding and decoding host-weight caches."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.models import pinned_offload
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAE
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MiniMaxH3LatentPreparationStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def stage(distributed_setup, monkeypatch):
    """Construct finite seeded VAEs with the tiny geometries used in parity tests."""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    torch.manual_seed(20260921)
    video_arch = MiniMaxH3VideoVAEArchConfig(
        latent_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=1,
        latents_mean=(0.0, ) * 4,
        latents_std=(1.0, ) * 4,
    )
    video = AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(
        arch_config=video_arch, use_tiling=False, use_temporal_tiling=False, use_parallel_tiling=False))
    audio_arch = MiniMaxH3AudioVAEArchConfig(
        encoder_dim=8,
        encoder_rates=(2, 2),
        latent_dim=32,
        latent_channels=4,
        num_attention_heads=2,
        decoder_dim=16,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(5, 4),
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, 2), ),
        latents_mean=[0.05 * index for index in range(4)],
        latents_std=[1.0 + 0.1 * index for index in range(4)],
    )
    audio = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=audio_arch))
    # ReplicatedLinear allocates uninitialized parameters without a checkpoint.
    with torch.no_grad():
        for model in (video, audio):
            for name, parameter in model.named_parameters():
                if parameter.ndim > 1:
                    torch.nn.init.xavier_uniform_(parameter)
                elif name.endswith("weight") and "norm" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(0.0, 0.02)
            model.float().eval()
    return MiniMaxH3LatentPreparationStage(video, audio, MiniMaxH3Scheduler())


def _args(pin: bool, offload: bool = True) -> SimpleNamespace:
    """Provide encoding transfer flags and the tiny VAE's packing geometry."""
    return SimpleNamespace(
        pin_cpu_memory=pin,
        vae_cpu_offload=offload,
        vae_parallel_encode=False,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 1, 1)),
            vae_config=SimpleNamespace(arch_config=SimpleNamespace(latent_channels=4)),
        ),
    )


def _batch(mode: str) -> ForwardBatch:
    """Rebuild consumed reference media and reset request noise for every encoding."""
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    image = Image.fromarray(pixels)
    batch = ForwardBatch(data_type="video", raw_latent_shape=(1, 4, 7, 4, 4),
                         generator=torch.Generator("cpu").manual_seed(7))
    if mode == "keyframe":
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = [image]
    else:
        batch.references = [
            MiniMaxH3PreparedReference(media_type="image", image=image),
            MiniMaxH3PreparedReference(media_type="video", frames=np.tile(pixels[None], (22, 1, 1, 1))),
            MiniMaxH3PreparedReference(media_type="audio", has_audio=True,
                                      waveform=torch.linspace(-0.1, 0.1, 401).repeat(2, 1)),
        ]
    return batch


def _encode(stage, mode, args):
    encode = stage._encode_fl2va_conditions if mode == "keyframe" else stage._encode_ref2va_conditions
    return encode(_batch(mode), args, get_local_torch_device())


def _models(stage, mode):
    return (stage.vae, ) if mode == "keyframe" else (stage.vae, stage.audio_vae)


def _tensors(model):
    yield from model.named_parameters()
    for name, buffer in model.named_buffers():
        yield "buffer:" + name, buffer


def _assert_host_storage(model, pin, pointers=None):
    """Check every registered tensor uses the same retained CPU allocation."""
    host = model._frozen_offload_host
    assert set(host) == {name for name, _ in _tensors(model)}
    for name, tensor in _tensors(model):
        assert tensor.device.type == "cpu", name
        assert tensor.data_ptr() == host[name].data_ptr(), name
        assert tensor.is_pinned() == pin, name
    observed = {name: tensor.data_ptr() for name, tensor in host.items()}
    if pointers is not None:
        assert observed == pointers
    return observed


def _decode(stage, mode, args):
    """Exercise decoding between requests using the same frozen-weight cache."""
    device = get_local_torch_device()
    for model in _models(stage, mode):
        pinned_offload.load(model, device, pin=args.pin_cpu_memory)
        try:
            if model is stage.vae:
                latents = torch.zeros(1, 4, 2, 4, 4, device=device)
                with torch.autocast("cuda", dtype=torch.float16):
                    decoded = model.decode(latents).sample
            else:
                decoded = model.decode(torch.zeros(2, 4, 12, device=device)).sample
            assert torch.isfinite(decoded).all()
        finally:
            pinned_offload.unload(model)


@pytest.mark.parametrize("mode", ("keyframe", "references"))
@pytest.mark.parametrize("pin", (False, True))
@torch.no_grad()
def test_encoding_reuses_decode_host_storage(stage, mode, pin):
    """Match ordinary transfers and reuse one cache across encoding and decoding."""
    args = _args(pin)
    with (
        patch.object(pinned_offload, "load", side_effect=lambda model, device, pin: model.to(device)),
        patch.object(pinned_offload, "unload", side_effect=lambda model: model.to("cpu")),
    ):
        expected = _encode(stage, mode, args)
    assert all(not hasattr(model, "_frozen_offload_host") for model in _models(stage, mode))

    # Encoding must also initialize the cache when a request begins with images.
    actual = _encode(stage, mode, args)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    pointers = [_assert_host_storage(model, pin) for model in _models(stage, mode)]
    _decode(stage, mode, args)
    for _ in range(2):
        actual = _encode(stage, mode, args)
        for tensor in actual:
            if tensor is not None:
                assert torch.isfinite(tensor).all()
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        for model, storage in zip(_models(stage, mode), pointers):
            _assert_host_storage(model, pin, storage)


@pytest.mark.parametrize("mode", ("keyframe", "references"))
@torch.no_grad()
def test_encoding_without_offload_keeps_cuda_weights(stage, mode):
    """Preserve GPU residency without creating CPU mirrors when offload is disabled."""
    device = get_local_torch_device()
    for model in _models(stage, mode):
        model.to(device)
    for _ in range(2):
        _encode(stage, mode, _args(pin=True, offload=False))
        for model in _models(stage, mode):
            assert all(tensor.device == device for _, tensor in _tensors(model))
            assert not hasattr(model, "_frozen_offload_host")


@pytest.mark.parametrize("branch", ("keyframe", "visual_reference", "audio_reference"))
@torch.no_grad()
def test_encoding_exception_restores_host_weights(stage, branch):
    """Release CUDA weights through each encoding branch's finally block."""
    mode = "keyframe" if branch == "keyframe" else "references"
    args = _args(pin=True)
    _decode(stage, mode, args)
    pointers = [_assert_host_storage(model, True) for model in _models(stage, mode)]
    model = stage.audio_vae if branch == "audio_reference" else stage.vae
    method = {"keyframe": "encode_keyframe", "visual_reference": "encode_pixels", "audio_reference": "encode"}[branch]
    with patch.object(model, method, side_effect=RuntimeError("injected encoding failure")):
        with pytest.raises(RuntimeError, match="injected encoding failure"):
            _encode(stage, mode, args)
    for model, storage in zip(_models(stage, mode), pointers):
        _assert_host_storage(model, True, storage)
