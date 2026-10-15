# SPDX-License-Identifier: Apache-2.0
"""A shared VAE must follow this request's tiling flag, including True -> False."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.stages import decoding, encoding, image_encoding
from fastvideo.fastvideo_args import ExecutionMode


class StatefulVAE:
    handles_latent_denorm = True
    spatial_compression_ratio = 1
    scaling_factor = 1.0

    def __init__(self, initial):
        self.use_tiling = initial
        self.use_temporal_tiling = True
        self.tile_sample_min_height = 256
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def decode(self, x):
        self.calls.append(self.use_tiling)
        return x + (0.25 if self.use_tiling else 0)

    def encode(self, x):
        return SimpleNamespace(mean=self.decode(x))

    def get_streaming_cache(self):
        return []

    def streaming_decode(self, x, cache, is_first_chunk):
        return self.decode(x), cache


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("boundary", ["decode", "streaming_decode", "encode", "image_encode"])
def test_request_resets_vae_tiling(monkeypatch, initial, boundary):
    for module in (decoding, encoding, image_encoding):
        monkeypatch.setattr(module, "get_local_torch_device", lambda: torch.device("cpu"))
    vae = StatefulVAE(initial)
    if boundary == "image_encode":
        stage = image_encoding.ImageVAEEncodingStage(vae)
    else:
        stage = encoding.EncodingStage(vae) if boundary == "encode" else decoding.DecodingStage(vae)
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(vae_precision="fp32", vae_decode_precision=None, vae_tiling=False),
        mode=ExecutionMode.PREPROCESS,
        disable_autocast=True,
        vae_cpu_offload=False,
    )
    outputs = []
    for flag in (False, True, False):
        args.pipeline_config.vae_tiling = flag
        x = torch.full((1, 3, 1, 2, 2), 0.5)
        if boundary == "image_encode":
            batch = SimpleNamespace(pil_image=x[:, :, 0], height=[2], width=[2], num_frames=[1])
            output = stage.forward(batch, args).image_latent
        elif boundary == "encode":
            output = stage.forward(SimpleNamespace(latents=x), args).latents
        elif boundary == "decode":
            output = stage.decode(x, args)
        else:
            output, _ = stage.streaming_decode(x, args, is_first_chunk=True)
        outputs.append(output)
        assert vae.use_tiling is flag
        # Resetting the request must preserve tile geometry and secondary configuration.
        assert vae.use_temporal_tiling
        assert vae.tile_sample_min_height == 256
    assert vae.calls == [False, True, False]
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
