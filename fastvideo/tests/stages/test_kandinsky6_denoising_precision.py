# SPDX-License-Identifier: Apache-2.0
"""Regression tests for three ``Kandinsky6DenoisingStage.forward`` numerics gaps against the diffusers
reference (``pipeline_kandinsky6_ti2va.py`` / ``transformer_kandinsky6.py``):

* the scheduler timestep must reach the transformer as fp32, in both the conditional and the
  unconditional call -- not rounded to ``dit_precision`` (bf16) first;
* the PiFlow (distilled) noisy latent state must stay fp32 across steps, separately from the bf16
  ``video`` buffer;
* the RoPE ``scale_factor`` must come from the transformer's static config, not a height/width
  heuristic.

Pure CPU harness: a fake transformer records the kwargs of every call instead of computing anything,
and a fake scheduler stands in for ``FlowMatchEulerDiscreteScheduler``/``PiflowScheduler`` so no real
weights or scheduler-config parsing is needed.
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo.pipelines.stages import kandinsky6 as k6_stages
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6DenoisingStage


class _RecordingTransformer:
    """Stands in for Kandinsky6Transformer3DModel: records every call's kwargs and returns zeros of
    the right shape instead of computing anything."""

    in_visual_dim = 2
    in_audio_dim = 2

    def __init__(self, scale_factor=(1.0, 2.0, 2.0)):
        self.scale_factor = scale_factor
        # get_sparse_params reads self.transformer.config.attention_engine; "sdpa" (dense) keeps it a
        # no-op, matching every real checkpoint observed so far.
        self.config = types.SimpleNamespace(attention_engine="sdpa")
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        video = kwargs["hidden_states"]
        audio = kwargs["hidden_states_audio"]
        video_vel = torch.zeros_like(video[..., :self.in_visual_dim])
        audio_vel = torch.zeros_like(audio)
        return types.SimpleNamespace(sample=(video_vel, audio_vel))


class _FakeFlowScheduler:
    is_piflow = False
    order = 1

    def __init__(self, num_steps: int):
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    def step(self, vel, timestep, sample, return_dict=False):
        return (sample - 0.01 * vel.to(sample.dtype), )


class _FakePiflowScheduler:
    is_piflow = True
    order = 1

    def __init__(self, num_steps: int):
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        self.step_sample_dtypes: list[torch.dtype] = []

    def step(self, vel, timestep, sample, return_dict=False):
        self.step_sample_dtypes.append(sample.dtype)
        # PiflowScheduler.step returns fp32 regardless of the input sample's dtype (the
        # reference's policy math runs in fp32); mirrored here to make the fake honest.
        return ((sample.float() - 0.01 * vel.float()), )


def _fastvideo_args(scale_factor=(1.0, 2.0, 2.0)) -> types.SimpleNamespace:
    dit_arch = types.SimpleNamespace(in_visual_dim=2, patch_size=(1, 1, 1), scale_factor=scale_factor)
    vae_arch = types.SimpleNamespace(spatial_compression_ratio=1)
    pipeline_config = types.SimpleNamespace(
        dit_precision="bf16",
        dit_config=types.SimpleNamespace(arch_config=dit_arch),
        vae_config=types.SimpleNamespace(arch_config=vae_arch),
    )
    return types.SimpleNamespace(model_loaded={"transformer": True}, pipeline_config=pipeline_config,
                                 disable_autocast=True)


def _batch(timesteps: torch.Tensor, *, guidance_scale: float = 1.0, do_cfg: bool = False,
          height: int = 2, width: int = 2) -> types.SimpleNamespace:
    video = torch.zeros(1, 1, 2, 2, 2, dtype=torch.bfloat16)  # [B,T,H,W,C]
    audio = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    prompt_embeds = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    pooled = torch.zeros(1, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.long)
    return types.SimpleNamespace(
        guidance_scale=guidance_scale,
        timesteps=timesteps,
        latents=video,
        audio_latents=audio,
        extra={},
        prompt_embeds=[prompt_embeds, pooled],
        prompt_attention_mask=[mask],
        do_classifier_free_guidance=do_cfg,
        negative_prompt_embeds=[prompt_embeds, pooled] if do_cfg else [],
        negative_attention_mask=[mask] if do_cfg else None,
        height=height,
        width=width,
        num_inference_steps=len(timesteps),
    )


def _stage(transformer, scheduler) -> Kandinsky6DenoisingStage:
    stage = Kandinsky6DenoisingStage.__new__(Kandinsky6DenoisingStage)
    stage.transformer = transformer
    stage.scheduler = scheduler
    return stage


@pytest.fixture()
def cpu_device(monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setattr(k6_stages, "get_local_torch_device", lambda: cpu)
    return cpu


def test_timestep_reaches_the_transformer_as_fp32_not_bf16(cpu_device):
    transformer = _RecordingTransformer()
    scheduler = _FakeFlowScheduler(num_steps=2)
    stage = _stage(transformer, scheduler)
    # A value that is NOT exactly representable in bf16 (bf16 has an 8-bit mantissa): if the stage
    # rounds it to dit_precision before the transformer sees it, this exact value is lost.
    timesteps = torch.tensor([517.3, 233.7], dtype=torch.float32)
    batch = _batch(timesteps)

    stage.forward(batch, _fastvideo_args())

    assert len(transformer.calls) == 2
    for call in transformer.calls:
        assert call["timestep"].dtype == torch.float32
    # The exact fp32 value survives -- not just the dtype -- confirming the timestep was never
    # rounded through bf16 (517.3 and 233.7 are not exactly bf16-representable) before this point.
    assert torch.equal(transformer.calls[0]["timestep"], torch.tensor([517.3]))
    assert torch.equal(transformer.calls[1]["timestep"], torch.tensor([233.7]))


def test_timestep_is_fp32_in_both_the_cond_and_uncond_call(cpu_device):
    transformer = _RecordingTransformer()
    scheduler = _FakeFlowScheduler(num_steps=1)
    stage = _stage(transformer, scheduler)
    timesteps = torch.tensor([517.3], dtype=torch.float32)
    batch = _batch(timesteps, guidance_scale=5.0, do_cfg=True)

    stage.forward(batch, _fastvideo_args())

    # One cond + one uncond call for the single step.
    assert len(transformer.calls) == 2
    for call in transformer.calls:
        assert call["timestep"].dtype == torch.float32


def test_piflow_latent_state_stays_fp32_across_steps(cpu_device):
    transformer = _RecordingTransformer()
    scheduler = _FakePiflowScheduler(num_steps=3)
    stage = _stage(transformer, scheduler)
    timesteps = torch.tensor([700.0, 400.0, 100.0])
    batch = _batch(timesteps, guidance_scale=1.0, do_cfg=False)

    stage.forward(batch, _fastvideo_args())

    assert scheduler.step_sample_dtypes == [torch.float32] * 3


def test_flow_euler_state_is_unaffected_by_the_piflow_fp32_change(cpu_device):
    # flow-Euler keeps writing through the bf16 `video` buffer every step (matches the diffusers
    # reference, which also downcasts its state to the parameter dtype there) -- only PiFlow gets the
    # separate fp32 tracker.
    transformer = _RecordingTransformer()
    scheduler = _FakeFlowScheduler(num_steps=2)
    stage = _stage(transformer, scheduler)
    timesteps = torch.tensor([700.0, 100.0])
    batch = _batch(timesteps)

    out = stage.forward(batch, _fastvideo_args())

    assert out.latents.dtype == torch.bfloat16


@pytest.mark.parametrize("height,width", [(480, 864), (720, 1280)])
def test_scale_factor_is_read_from_the_transformer_config_not_a_resolution_heuristic(cpu_device, height, width):
    # The old heuristic special-cased 480<=h,w<=854 -> (1,2,2), else (1,3.16,3.16). Both resolutions
    # here take the "else" branch under that heuristic (864 and 1280 both exceed 854), but both
    # official Kandinsky6 checkpoints declare scale_factor=(1,2,2) unconditionally in
    # transformer/config.json, including for the 480x864 docstring example.
    transformer = _RecordingTransformer(scale_factor=(1.0, 2.0, 2.0))
    scheduler = _FakeFlowScheduler(num_steps=1)
    stage = _stage(transformer, scheduler)
    timesteps = torch.tensor([500.0])
    batch = _batch(timesteps, height=height, width=width)

    stage.forward(batch, _fastvideo_args())

    assert transformer.calls[0]["scale_factor"] == (1.0, 2.0, 2.0)


def test_scale_factor_falls_back_to_the_static_arch_config_when_the_transformer_lacks_the_attribute(cpu_device):
    transformer = _RecordingTransformer()
    del transformer.scale_factor  # simulate a not-yet-loaded transformer / older checkpoint object
    scheduler = _FakeFlowScheduler(num_steps=1)
    stage = _stage(transformer, scheduler)
    timesteps = torch.tensor([500.0])
    batch = _batch(timesteps, height=720, width=1280)

    stage.forward(batch, _fastvideo_args(scale_factor=(1.0, 2.0, 2.0)))

    assert transformer.calls[0]["scale_factor"] == (1.0, 2.0, 2.0)
