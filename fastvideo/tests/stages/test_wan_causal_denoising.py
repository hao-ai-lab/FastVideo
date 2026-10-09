# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for causal Wan scheduling and cache lifetime."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
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


def test_causal_standard_resets_scheduler_per_block_and_caches_per_request(monkeypatch, env_overrides):
    _patch_denoising_module(monkeypatch, env_overrides, "1.0")
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


@pytest.mark.parametrize("active_model", ["high", "low"])
def test_dual_transformer_context_populates_cache_unused_by_denoising(monkeypatch, env_overrides, active_model):
    _patch_denoising_module(monkeypatch, env_overrides, "1.0")
    from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import SelfForcingFlowMatchScheduler
    from fastvideo.models.wan.causal_transformer import CausalWanSelfAttention
    from fastvideo.pipelines.basic.wan.stages import causal_denoising

    monkeypatch.setattr(causal_denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(causal_denoising, "set_forward_context", lambda **kwargs: nullcontext())

    class CacheWritingDenoiser(TinyCausalDenoiser):
        local_attn_size = 9

        def __init__(self):
            super().__init__()
            self.chunk_flags = []

        def forward(self, latents, prompts, timestep, *, kv_cache, crossattn_cache, current_start, start_frame,
                    is_chunk_start=None):
            self.chunk_flags.append(is_chunk_start)
            key = latents.permute(0, 2, 3, 4, 1).reshape(latents.shape[0], -1, 1, 2)
            current_end = current_start + key.shape[1]
            for cache in kv_cache:
                _, end = CausalWanSelfAttention._write_in_place(
                    cache, key, key, current_end=current_end, global_end_index=cache["global_end_index"],
                    local_end_index_prev=cache["local_end_index"], is_chunk_start=is_chunk_start,
                )
                CausalWanSelfAttention._update_cache_counters(cache, current_end, end)
            return super().forward(
                latents, prompts, timestep, kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=current_start, start_frame=start_frame,
            )

    high, low = CacheWritingDenoiser(), CacheWritingDenoiser()
    stage = causal_denoising.CausalDMDDenosingStage(
        high, SelfForcingFlowMatchScheduler(num_inference_steps=1000), transformer_2=low,
    )
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    args = _args()
    args.enable_causal_cuda_graph = True  # Unsupported dual-model execution must preserve ordinary cache writes.
    args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
    args.pipeline_config.dit_config.boundary_ratio = 0.5
    args.pipeline_config.dmd_denoising_steps = [1000, 750] if active_model == "high" else [400, 250]
    args.pipeline_config.warp_denoising_step = False
    batch = _batch(steps=2, cfg=False)
    batch.latents = batch.latents.repeat(1, 1, 3, 1, 1)
    batch.generator = [torch.Generator().manual_seed(123)]
    result = stage.forward(batch, args)
    assert torch.isfinite(result.latents).all()
    assert "causal_cuda_graph" not in result.extra
    active, inactive = (high, low) if active_model == "high" else (low, high)
    assert len(active.calls) == 9  # Three blocks, each with two denoising calls and one context write.
    assert len(inactive.calls) == 3  # Its first cache write in each block is the clean-context call.
    for model in (high, low):
        assert all(flag is None for flag in model.chunk_flags)
        for cache in model.calls[-1][2]:
            assert cache["global_end_index"] == cache["local_end_index"] == 7 * stage.frame_seq_length


@pytest.mark.parametrize("sampler", ["standard", "dmd"])
@pytest.mark.parametrize("interrupt_after_first", [False, True])
def test_interruption_stops_before_context_write(monkeypatch, env_overrides, sampler, interrupt_after_first):
    _patch_denoising_module(monkeypatch, env_overrides, "1.0")
    from fastvideo.pipelines.basic.wan.stages import causal_denoising
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    monkeypatch.setattr(causal_denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(causal_denoising, "set_forward_context", lambda **kwargs: nullcontext())
    model, scheduler = TinyCausalDenoiser(), RecordingUniPC()
    stage = (causal_denoising.CausalDenoisingStage(model, scheduler) if sampler == "standard"
             else causal_denoising.CausalDMDDenosingStage(model, FlowMatchEulerDiscreteScheduler()))
    stage.progress_bar = lambda **kwargs: NullProgressBar()
    stage.interrupt = not interrupt_after_first
    if interrupt_after_first:
        model.register_forward_hook(lambda *args: setattr(stage, "interrupt", True))
    args = _args()
    args.pipeline_config.text_encoder_configs = [SimpleNamespace(arch_config=SimpleNamespace(text_len=8))]
    args.pipeline_config.dmd_denoising_steps = [1000, 750, 500, 250]
    args.pipeline_config.warp_denoising_step = False
    batch = _batch(steps=4, cfg=False)
    batch.generator = [torch.Generator().manual_seed(123)]
    original = batch.latents.clone()
    torch.testing.assert_close(stage.forward(batch, args).latents, original, atol=0, rtol=0)
    assert len(model.calls) == int(interrupt_after_first)
    if sampler == "standard":
        assert scheduler.resets == int(interrupt_after_first)
