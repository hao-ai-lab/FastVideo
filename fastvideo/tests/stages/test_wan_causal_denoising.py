# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for causal Wan scheduling and cache lifetime."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.tests.stages._denoising_fixtures import NullProgressBar, _patch_denoising_module
from fastvideo.tests.stages._denoising_fixtures import _args, _batch


class TinyCausalDenoiser(torch.nn.Module):
    hidden_size = 2
    num_attention_heads = 1
    attention_head_dim = 2
    local_attn_size = 3
    sink_size = 1

    def __init__(self):
        super().__init__()
        self.blocks = [None, None]
        self.config = SimpleNamespace(arch_config=SimpleNamespace(
            num_frames_per_block=3, sliding_window_num_frames=6, patch_size=(1, 1, 1)))
        self.calls = []

    def forward(self, latents, prompts, timestep, *, kv_cache, crossattn_cache, current_start, start_frame):
        self.calls.append((timestep.clone(), start_frame, kv_cache, crossattn_cache))
        return latents.float() * 0.125


class RecordingUniPC(FlowUniPCMultistepScheduler):

    def __init__(self):
        super().__init__(shift=3.0)
        self.resets = 0

    def set_timesteps(self, *args, **kwargs):
        self.resets += 1
        return super().set_timesteps(*args, **kwargs)


def test_causal_standard_resets_scheduler_per_block_and_caches_per_request(monkeypatch):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.basic.wan.stages import causal_denoising
    monkeypatch.setattr(causal_denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(causal_denoising, "set_forward_context", lambda **kwargs: nullcontext())
    model, scheduler = TinyCausalDenoiser(), RecordingUniPC()
    stage = causal_denoising.CausalDenoisingStage(model, scheduler)
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    args = _args()
    args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
    args.pipeline_config.context_noise = 0

    outputs = []
    for _ in range(2):
        batch = _batch(steps=50, cfg=False)
        batch.latents = batch.latents.repeat(1, 1, 2, 1, 1)
        outputs.append(stage.forward(batch, args).latents.clone())

    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert scheduler.resets == 4  # Two blocks per request, including a fresh multi-step history.
    assert len(model.calls) == 2 * 2 * (50 + 1)  # Each block also writes its clean context to the cache.
    first_cache, first_cross = model.calls[0][2:]
    assert all(call[2] is first_cache and call[3] is first_cross for call in model.calls[:102])
    assert model.calls[102][2] is not first_cache
    assert model.calls[102][3] is not first_cross
    assert first_cache[0]["k"].shape == (1, 3 * 8, 1, 2)
    assert first_cross[0]["k"].shape == (1, 8, 1, 2)
    assert [model.calls[i][1] for i in (0, 50, 51, 101)] == [0, 0, 3, 3]
    assert model.calls[50][0].eq(0).all() and model.calls[101][0].eq(0).all()


def test_causal_samplers_share_cache_layout_not_sampling_inheritance():
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import (
        CausalDMDDenosingStage, CausalDenoisingStage, WanCausalDenoisingBase,
    )
    assert issubclass(CausalDenoisingStage, WanCausalDenoisingBase)
    assert issubclass(CausalDMDDenosingStage, WanCausalDenoisingBase)
    assert not issubclass(CausalDenoisingStage, CausalDMDDenosingStage)
