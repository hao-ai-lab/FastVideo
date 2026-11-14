# SPDX-License-Identifier: Apache-2.0
"""Causal Wan's default inference path must equal eager over a causal DMD denoising run, bit for bit.

The causal block (fastvideo/models/wan/causal_transformer.py) modulates with
bf16 tables, one row per frame, around plain LayerNorms, and runs under
autocast, so its eager rounding points differ from the dense block's. This
test runs the real SFWan2.1-T2V-1.3B transformer through
``CausalDMDDenosingStage`` (two three-frame chunks, four DMD steps, kv and
cross-attention caches) twice in one process, once with the exact kernels
and once with the eager expressions, and compares the latents with
``torch.equal``. No stored reference is involved.
"""

import os

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.layers import triton_exact_modulation as exact
from fastvideo.tests.golden_gate._harness import DEFAULT_SEED, distributed_runtime
from fastvideo.tests.golden_gate._tensor_golden import deterministic_forward

__all__ = ["distributed_runtime"]

SFWAN_REPO = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"


def causal_denoising_outputs(device):
    from huggingface_hub import snapshot_download

    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import SelfForcingFlowMatchScheduler
    from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig
    from fastvideo.pipelines.basic.wan.stages.causal_denoising import CausalDMDDenosingStage
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    root = snapshot_download(SFWAN_REPO, allow_patterns=["transformer/*", "scheduler/*"])
    path = os.path.join(root, "transformer")
    args = FastVideoArgs(
        model_path=path, pipeline_config=SelfForcingWanT2V480PConfig(),
        num_gpus=1, hsdp_shard_dim=1, use_fsdp_inference=False, dit_cpu_offload=False,
        dit_layerwise_offload=False, enable_torch_compile=False, inference_torch_compile=False,
    )
    model = TransformerLoader().load(path, args)
    args.model_loaded["transformer"] = True
    scheduler = SelfForcingFlowMatchScheduler.from_pretrained(root, subfolder="scheduler")
    generator = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED)
    latents = torch.randn(1, 16, 6, 4, 4, generator=generator).to(device)
    prompt = torch.randn(1, 16, 4096, generator=generator).to(device=device, dtype=torch.bfloat16)
    batch = ForwardBatch(
        data_type="video", latents=latents, prompt_embeds=[prompt],
        timesteps=torch.tensor(args.pipeline_config.dmd_denoising_steps), num_inference_steps=4,
        guidance_scale=1.0, do_classifier_free_guidance=False, height=32, width=32, num_frames=21,
        raw_latent_shape=tuple(latents.shape), save_video=False,
        generator=torch.Generator(device="cpu").manual_seed(DEFAULT_SEED),  # the stage's re-noising draws
    )
    # The pipeline runs its stages under no_grad; inference_mode breaks the scheduler's version counters.
    with torch.no_grad():
        result = CausalDMDDenosingStage(transformer=model, scheduler=scheduler).forward(batch, args)
    return {"latents": result.latents.clone()}


def test_wan_causal_denoising_exact_tier_equals_eager(distributed_runtime, env_overrides, monkeypatch):
    if not exact.HAVE_TRITON:
        pytest.skip("the exact tier requires Triton")
    env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(False))

    kernel_calls = {"gate_residual": 0, "modulate": 0}
    real = {name: getattr(exact, name) for name in kernel_calls}

    def counting(name):
        def call(*args, **kwargs):
            kernel_calls[name] += 1
            return real[name](*args, **kwargs)

        return call

    with deterministic_forward() as device:
        with monkeypatch.context() as spy:
            for name in kernel_calls:
                spy.setattr(exact, name, counting(name))
            actual = causal_denoising_outputs(device)
        with monkeypatch.context() as eager_only:
            eager_only.setattr(exact, "HAVE_TRITON", False)
            expected = causal_denoising_outputs(device)

    # 30 blocks x (2 chunks x (4 steps + 1 context write)) forwards: three
    # gated-residual sites and two modulation sites each.
    assert kernel_calls == {"gate_residual": 900, "modulate": 600}, kernel_calls
    for key in sorted(actual):
        a, b = actual[key], expected[key]
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert torch.equal(a, b), f"{key}: max abs diff {(a.float() - b.float()).abs().max().item():.3e}"


def test_wan_causal_ignores_fused_norm_opt_in(distributed_runtime, env_overrides, monkeypatch):
    """FASTVIDEO_ENABLE_FUSED_NORM=1 must not change a causal rollout: the block opts out of the fused kernel."""
    if not exact.HAVE_TRITON:
        pytest.skip("the exact tier requires Triton")
    from fastvideo.layers import triton_fused_norm as fused

    fused_calls = {"fused_residual_norm_mod": 0, "fused_norm_mod": 0}
    real = {name: getattr(fused, name) for name in fused_calls}

    def counting(name):
        def call(*args, **kwargs):
            fused_calls[name] += 1
            return real[name](*args, **kwargs)

        return call

    with deterministic_forward() as device:
        with monkeypatch.context() as spy:
            for name in fused_calls:
                spy.setattr(fused, name, counting(name))
            env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(True))
            opted_in = causal_denoising_outputs(device)
        with monkeypatch.context() as eager_only:
            env_overrides.enter_context(envs.FASTVIDEO_ENABLE_FUSED_NORM.override(False))
            eager_only.setattr(exact, "HAVE_TRITON", False)
            expected = causal_denoising_outputs(device)
    assert fused_calls == {"fused_residual_norm_mod": 0, "fused_norm_mod": 0}, fused_calls
    assert torch.equal(opted_in["latents"], expected["latents"])
