# SPDX-License-Identifier: Apache-2.0
"""Three UniPC steps with the real 1.3B DiT, fixed embeddings, and tiny latents.

No tokenizer, text encoder, VAE, video encode, or reference-video download. The
per-step tensor reference catches stage/scheduler drift after the block gate.
"""

import torch

from fastvideo.tests.golden_gate._harness import DEFAULT_SEED, distributed_runtime
from fastvideo.tests.golden_gate._tensor_golden import assert_tensor_golden, deterministic_forward
from fastvideo.tests.golden_gate._wan_checkpoint import checkpoint_identity, component_path

__all__ = ["distributed_runtime"]


def denoising_outputs(device):
    from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
    from fastvideo.pipelines.basic.wan.stages.denoising import WanDenoisingStage
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    path = component_path("transformer")
    args = FastVideoArgs(
        model_path=str(path), pipeline_config=WanT2V480PConfig(),
        num_gpus=1, hsdp_shard_dim=1, use_fsdp_inference=False, dit_cpu_offload=False,
        dit_layerwise_offload=False, enable_torch_compile=False, inference_torch_compile=False,
    )
    model = TransformerLoader().load(str(path), args)
    args.model_loaded["transformer"] = True
    scheduler = FlowUniPCMultistepScheduler(shift=3.0)
    scheduler.set_timesteps(3, device=device)
    generator = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED)
    latents = torch.randn(1, 16, 3, 4, 4, generator=generator).to(device)
    prompt = torch.randn(1, 16, 4096, generator=generator).to(device=device, dtype=torch.bfloat16)
    negative = torch.randn(1, 16, 4096, generator=generator).to(device=device, dtype=torch.bfloat16)
    batch = ForwardBatch(
        data_type="video", latents=latents, prompt_embeds=[prompt], negative_prompt_embeds=[negative],
        timesteps=scheduler.timesteps.clone(), num_inference_steps=3, guidance_scale=5.0,
        do_classifier_free_guidance=True, height=32, width=32, num_frames=9,
        raw_latent_shape=tuple(latents.shape), save_video=False, return_trajectory_latents=True,
    )
    with torch.inference_mode():
        result = WanDenoisingStage(model, scheduler).forward(batch, args)
    return {"latents": result.latents, "trajectory": result.trajectory_latents,
            "timesteps": result.trajectory_timesteps}, {
                **checkpoint_identity(path), "latent_shape": list(latents.shape),
                "prompt_shape": list(prompt.shape), "steps": 3, "guidance_scale": 5.0,
                "scheduler": "FlowUniPCMultistepScheduler", "shift": 3.0, "precision": "bf16",
                "cfg_gate_step": 1.0,
            }


def test_wan_denoising_golden_gate(distributed_runtime, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_CFG_GATE_STEP", "1.0")
    with deterministic_forward("FLASH_ATTN") as device:
        outputs, identity = denoising_outputs(device)
        assert_tensor_golden("wan_denoising", outputs, identity=identity, attention_backend="FLASH_ATTN")
