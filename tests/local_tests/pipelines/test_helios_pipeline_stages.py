# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end contracts for Helios custom pipeline stages."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache
def _probe_stages() -> dict:
    script = r"""
import json
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.models.schedulers.scheduling_helios_dmd import HeliosDMDScheduler
from fastvideo.pipelines.basic.helios.stages import (
    HeliosChunkDecodingStage,
    HeliosPyramidDenoisingStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

if not torch.cuda.is_available():
    print(json.dumps({"cuda_available": False}))
    raise SystemExit(0)

device = torch.device("cuda")


class TinyTransformer(nn.Module):
    in_channels = 2
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), device=device, dtype=torch.bfloat16))
        self.calls = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        **kwargs,
    ):
        del kwargs
        self.calls.append({
            "latent_shape": list(hidden_states.shape),
            "history_shapes": [
                list(latents_history_short.shape),
                list(latents_history_mid.shape),
                list(latents_history_long.shape),
            ],
            "indices": [
                indices_hidden_states.tolist(),
                indices_latents_history_short.tolist(),
                indices_latents_history_mid.tolist(),
                indices_latents_history_long.tolist(),
            ],
            "history_means": [
                latents_history_short.float().mean().item(),
                latents_history_mid.float().mean().item(),
                latents_history_long.float().mean().item(),
            ],
            "short_prefix_mean": latents_history_short[:, :, :1].float().mean().item(),
        })
        scalar = (
            timestep.float().view(-1, 1, 1, 1, 1) / 1000
            + encoder_hidden_states.float().mean(dim=(1, 2)).view(-1, 1, 1, 1, 1) * 0.01
            + latents_history_short.float().mean(dim=(1, 2, 3, 4)).view(-1, 1, 1, 1, 1) * 0.02
            + latents_history_mid.float().mean(dim=(1, 2, 3, 4)).view(-1, 1, 1, 1, 1) * 0.03
            + latents_history_long.float().mean(dim=(1, 2, 3, 4)).view(-1, 1, 1, 1, 1) * 0.04
        )
        return (hidden_states.float() * 0.125 + scalar).to(hidden_states.dtype)


def scheduler():
    return HeliosDMDScheduler(
        stages=3,
        stage_range=[0, 1 / 3, 2 / 3, 1],
        gamma=1 / 3,
        shift=1.0,
        use_dynamic_shifting=True,
        time_shift_type="linear",
    )


def block_noise(sched, shape, generator):
    b, c, t, h, w = shape
    block_size = 4
    gamma = sched.config.gamma
    covariance = (
        torch.eye(block_size, device=device) * (1 + gamma)
        - torch.ones(block_size, block_size, device=device) * gamma
    )
    covariance += torch.eye(block_size, device=device) * 1e-8
    cholesky = torch.linalg.cholesky(covariance.float())
    z = torch.randn(
        b * c * t * (h // 2) * (w // 2),
        block_size,
        generator=generator,
        device=generator.device,
    ).to(device)
    noise = z @ cholesky.T
    noise = noise.view(b, c, t, h // 2, w // 2, 2, 2)
    return noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(shape)


def reference_sample(model, sched, generator, prompt):
    history_sizes = [16, 2, 1]
    history = torch.zeros(1, 2, 19, 8, 8, device=device)
    history_long, history_mid, history_one = history.split(history_sizes, dim=2)
    history_short = torch.cat([torch.zeros(1, 2, 1, 8, 8, device=device), history_one], dim=2)

    all_indices = torch.arange(29, device=device)
    prefix, long_idx, mid_idx, one_idx, current_idx = all_indices.split([1, 16, 2, 1, 9])
    indices = (
        current_idx.unsqueeze(0),
        torch.cat([prefix, one_idx]).unsqueeze(0),
        mid_idx.unsqueeze(0),
        long_idx.unsqueeze(0),
    )

    latents = torch.randn((1, 2, 9, 8, 8), generator=generator).to(device)
    flat = latents.permute(0, 2, 1, 3, 4).reshape(9, 2, 8, 8)
    flat = F.interpolate(flat, size=(4, 4), mode="bilinear") * 2
    flat = F.interpolate(flat, size=(2, 2), mode="bilinear") * 2
    latents = flat.reshape(1, 9, 2, 2, 2).permute(0, 2, 1, 3, 4)
    start_points = [latents]

    for stage_index in range(3):
        image_seq_len = math.prod(latents.shape[-3:]) // 4
        mu = image_seq_len * ((1.15 - 0.5) / (4096 - 256)) + (
            0.5 - ((1.15 - 0.5) / (4096 - 256)) * 256
        )
        sched.set_timesteps(
            1,
            stage_index,
            device=device,
            mu=mu,
            is_amplify_first_chunk=True,
        )
        timesteps = sched.timesteps
        if stage_index > 0:
            b, c, t, h, w = latents.shape
            flat = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(h * 2, w * 2), mode="nearest")
            latents = flat.reshape(b, t, c, h * 2, w * 2).permute(0, 2, 1, 3, 4)
            original_signal = 1 - sched.ori_start_sigmas[stage_index]
            gamma = sched.config.gamma
            alpha = 1 / (math.sqrt(1 + 1 / gamma) * (1 - original_signal) + original_signal)
            beta = alpha * (1 - original_signal) / math.sqrt(gamma)
            latents = alpha * latents + beta * block_noise(sched, tuple(latents.shape), generator).to(torch.bfloat16)
            start_points.append(latents)

        for step_index, timestep_value in enumerate(timesteps):
            timestep = timestep_value.expand(1).to(torch.int64)
            prediction = model(
                hidden_states=latents.to(torch.bfloat16),
                timestep=timestep,
                encoder_hidden_states=prompt,
                indices_hidden_states=indices[0],
                indices_latents_history_short=indices[1],
                indices_latents_history_mid=indices[2],
                indices_latents_history_long=indices[3],
                latents_history_short=history_short.to(torch.bfloat16),
                latents_history_mid=history_mid.to(torch.bfloat16),
                latents_history_long=history_long.to(torch.bfloat16),
            )
            latents = sched.step(
                prediction,
                timestep_value,
                latents,
                generator=generator,
                return_dict=False,
                cur_sampling_step=step_index,
                dmd_noisy_tensor=start_points[stage_index],
                dmd_sigmas=sched.sigmas,
                dmd_timesteps=sched.timesteps,
                all_timesteps=timesteps,
            )[0]
    return latents


prompt = torch.ones(1, 5, 4, device=device, dtype=torch.bfloat16)
actual_model = TinyTransformer()
actual_scheduler = scheduler()
batch = ForwardBatch(
    data_type="video",
    prompt_embeds=[prompt],
    generator=[torch.Generator("cpu").manual_seed(321)],
    height=64,
    width=64,
    num_frames=33,
    guidance_scale=1.0,
    pyramid_num_inference_steps_list=[1, 1, 1],
    history_sizes=[16, 2, 1],
    num_latent_frames_per_chunk=9,
    keep_first_frame=True,
    is_amplify_first_chunk=True,
)
args = SimpleNamespace(
    pipeline_config=SimpleNamespace(
        dit_precision="bf16",
        vae_config=SimpleNamespace(
            arch_config=SimpleNamespace(scale_factor_spatial=8, scale_factor_temporal=4)
        ),
    ),
    model_loaded={"transformer": True, "vae": True},
    dit_cpu_offload=False,
    dit_layerwise_offload=False,
    use_fsdp_inference=False,
)
actual_batch = HeliosPyramidDenoisingStage(actual_model, actual_scheduler).forward(batch, args)

autoregressive_model = TinyTransformer()
autoregressive_batch = ForwardBatch(
    data_type="video",
    prompt_embeds=[prompt],
    generator=[torch.Generator("cpu").manual_seed(321)],
    height=64,
    width=64,
    num_frames=65,
    guidance_scale=1.0,
    pyramid_num_inference_steps_list=[1, 1, 1],
    history_sizes=[16, 2, 1],
    num_latent_frames_per_chunk=9,
    keep_first_frame=True,
    is_amplify_first_chunk=True,
)
autoregressive_batch = HeliosPyramidDenoisingStage(autoregressive_model, scheduler()).forward(
    autoregressive_batch, args
)

reference_model = TinyTransformer()
expected = reference_sample(
    reference_model,
    scheduler(),
    torch.Generator("cpu").manual_seed(321),
    prompt,
)


class TinyVAE(nn.Module):
    handles_latent_denorm = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def decode(self, latent):
        self.calls.append(list(latent.shape))
        output_frames = (latent.shape[2] - 1) * 4 + 1
        value = -1.0 if len(self.calls) == 1 else 1.0
        return torch.full(
            (latent.shape[0], 3, output_frames, latent.shape[3] * 8, latent.shape[4] * 8),
            value,
            device=latent.device,
        )


vae = TinyVAE().to(device)
decode_batch = ForwardBatch(data_type="video")
decode_batch.latents = torch.cat([expected, expected], dim=2)
decode_batch.helios_latent_chunks = [expected, expected]
decode_args = SimpleNamespace(
    output_type="video",
    model_loaded={"vae": True},
    pipeline_config=SimpleNamespace(
        vae_decode_precision="fp32",
        vae_precision="fp32",
        vae_tiling=False,
        vae_config=SimpleNamespace(arch_config=SimpleNamespace(scale_factor_temporal=4)),
    ),
    disable_autocast=False,
    vae_cpu_offload=False,
)
decoded = HeliosChunkDecodingStage(vae).forward(decode_batch, decode_args).output

print(json.dumps({
    "cuda_available": True,
    "latent_max_diff": (actual_batch.latents - expected).abs().max().item(),
    "call_shapes": [item["latent_shape"] for item in actual_model.calls],
    "history_shapes": actual_model.calls[0]["history_shapes"],
    "indices": actual_model.calls[0]["indices"],
    "autoregressive_latent_shape": list(autoregressive_batch.latents.shape),
    "autoregressive_call_count": len(autoregressive_model.calls),
    "autoregressive_second_history_means": autoregressive_model.calls[6]["history_means"],
    "autoregressive_second_short_prefix_mean": autoregressive_model.calls[6]["short_prefix_mean"],
    "vae_calls": vae.calls,
    "decoded_shape": list(decoded.shape),
    "decoded_first_mean": decoded[:, :, :33].mean().item(),
    "decoded_second_mean": decoded[:, :, 33:].mean().item(),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _results() -> dict:
    result = _probe_stages()
    if not result["cuda_available"]:
        pytest.skip("Helios tiny stage parity requires CUDA")
    return result


def test_tiny_pyramid_stage_matches_independent_official_loop():
    result = _results()
    assert result["latent_max_diff"] == 0
    assert result["call_shapes"] == [
        [1, 2, 9, 2, 2],
        [1, 2, 9, 2, 2],
        [1, 2, 9, 4, 4],
        [1, 2, 9, 4, 4],
        [1, 2, 9, 8, 8],
        [1, 2, 9, 8, 8],
    ]


def test_tiny_pyramid_stage_passes_exact_history_and_indices():
    result = _results()
    assert result["history_shapes"] == [
        [1, 2, 2, 8, 8],
        [1, 2, 2, 8, 8],
        [1, 2, 16, 8, 8],
    ]
    current, short, mid, long = result["indices"]
    assert current == [list(range(20, 29))]
    assert short == [[0, 19]]
    assert mid == [[17, 18]]
    assert long == [list(range(1, 17))]


def test_tiny_pyramid_stage_uses_history_on_second_chunk():
    result = _results()
    assert result["autoregressive_latent_shape"] == [1, 2, 18, 8, 8]
    assert result["autoregressive_call_count"] == 9
    assert all(abs(value) > 1e-5 for value in result["autoregressive_second_history_means"])
    assert abs(result["autoregressive_second_short_prefix_mean"]) > 1e-5


def test_chunk_decoder_calls_vae_per_chunk_and_matches_frame_rounding():
    result = _results()
    assert result["vae_calls"] == [[1, 2, 9, 8, 8], [1, 2, 9, 8, 8]]
    assert result["decoded_shape"] == [1, 3, 65, 64, 64]
    assert result["decoded_first_mean"] == 0
    assert result["decoded_second_mean"] == 1
