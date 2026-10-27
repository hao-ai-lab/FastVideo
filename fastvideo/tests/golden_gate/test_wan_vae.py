# SPDX-License-Identifier: Apache-2.0
"""Wan codec golden: FP32 encode, BF16 decode, streaming, and request reset.

One CUDA GPU, only the pinned VAE weights. Nine tiny video frames exercise the
special first frame and two temporal-cache chunks; no DiT or encoder download.
"""

import torch

from fastvideo.tests.golden_gate._harness import DEFAULT_SEED, distributed_runtime
from fastvideo.tests.golden_gate._tensor_golden import assert_tensor_golden, deterministic_forward
from fastvideo.tests.golden_gate._wan_checkpoint import checkpoint_identity, component_path

__all__ = ["distributed_runtime"]


def vae_outputs(device):
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import VAELoader
    from fastvideo.models.wan.vae_config import WanVAEConfig
    from fastvideo.pipelines.stages.decoding import DecodingStage

    path = component_path("vae")
    args = FastVideoArgs(
        model_path=str(path),
        pipeline_config=PipelineConfig(vae_config=WanVAEConfig(), vae_precision="fp32", vae_decode_precision="bf16"),
        vae_cpu_offload=False,
    )
    vae = VAELoader().load(str(path), args)
    assert vae.use_feature_cache and not vae.use_tiling
    generator = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED)
    video = torch.randn(1, 3, 9, 16, 24, generator=generator).clamp(-1, 1).to(device)
    latent = torch.randn(1, 16, 3, 2, 3, generator=generator).to(device)
    with torch.inference_mode():
        posterior = vae.encode(video)
        repeated = vae.encode(video)
        torch.testing.assert_close(repeated.mean, posterior.mean, atol=0, rtol=0)
        torch.testing.assert_close(repeated.logvar, posterior.logvar, atol=0, rtol=0)
        decoder = DecodingStage(vae)
        decoded = decoder.decode(latent, args)
        cache, chunks = None, []
        for index, chunk in enumerate(latent.split(1, dim=2)):
            frames, cache = decoder.streaming_decode(chunk, args, cache=cache, is_first_chunk=index == 0)
            chunks.append(frames)
        streamed = torch.cat(chunks, dim=2)
        torch.testing.assert_close(streamed, decoded, atol=0, rtol=0)
        torch.testing.assert_close(decoder.decode(latent, args), decoded, atol=0, rtol=0)
    return {"mean": posterior.mean, "logvar": posterior.logvar, "decoded": decoded, "streamed": streamed}, {
        **checkpoint_identity(path),
        "video_shape": list(video.shape),
        "latent_shape": list(latent.shape),
        "encode_precision": "fp32",
        "decode_precision": "bf16",
        "normalization": "DecodingStage",
    }


def test_wan_vae_golden_gate(distributed_runtime):
    with deterministic_forward() as device:
        outputs, identity = vae_outputs(device)
        assert_tensor_golden("wan_vae", outputs, identity=identity)
