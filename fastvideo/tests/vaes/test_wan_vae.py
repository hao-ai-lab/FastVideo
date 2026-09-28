# SPDX-License-Identifier: Apache-2.0
"""Wan VAE encode/decode parity against Diffusers (one CUDA GPU, FP32)."""

import os

import pytest
import torch
from diffusers import AutoencoderKLWan
from torch.testing import assert_close

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.models.wan.vae_config import WanVAEConfig
from fastvideo.tests.golden_gate._wan_checkpoint import component_path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29503")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan VAE parity requires one CUDA GPU")
@pytest.mark.usefixtures("distributed_setup")
def test_wan_vae():
    vae_path = str(component_path("vae"))
    device = torch.device("cuda:0")
    precision = torch.float32
    args = FastVideoArgs(model_path=vae_path,
                         pipeline_config=PipelineConfig(vae_config=WanVAEConfig(), vae_precision="fp32"))
    args.device = device
    args.vae_cpu_offload = False

    candidate = VAELoader().load(vae_path, args)
    assert candidate.use_feature_cache  # Preserve the original Wan VAE algorithm.
    reference = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=precision).to(device).eval()
    input_tensor = torch.randn(1, 3, 81, 32, 32, device=device, dtype=precision)

    with torch.no_grad():
        reference_latent = reference.encode(input_tensor).latent_dist
        candidate_latent = candidate.encode(input_tensor)
        assert_close(reference_latent.mean, candidate_latent.mean, atol=1e-4, rtol=1e-4)
        assert_close(reference_latent.logvar, candidate_latent.logvar, atol=1e-4, rtol=1e-4)

        reference_mean = torch.tensor(reference.config.latents_mean, device=device, dtype=precision).view(
            1, reference.config.z_dim, 1, 1, 1)
        reference_scale = (1.0 / torch.tensor(reference.config.latents_std).view(
            1, reference.config.z_dim, 1, 1, 1)).to(device, precision)
        reference_output = reference.decode(reference_latent.mode() / reference_scale + reference_mean).sample

        candidate_mean = candidate.config.arch_config.shift_factor.to(device, precision)
        candidate_scale = candidate.config.arch_config.scaling_factor.to(device, precision)
        decode_input = candidate_latent.mode() / candidate_scale + candidate_mean
        candidate_output = candidate.decode(decode_input)
        assert_close(reference_output, candidate_output, atol=1e-5, rtol=1e-3)

        # The same decoder is also used incrementally by streaming consumers.
        cache = candidate.get_streaming_cache()
        outputs = []
        for index, chunk in enumerate(decode_input.split(1, dim=2)):
            output, cache = candidate.streaming_decode(chunk, cache, is_first_chunk=index == 0)
            outputs.append(output)
        assert_close(torch.cat(outputs, dim=2), candidate_output, atol=1e-5, rtol=1e-3)
