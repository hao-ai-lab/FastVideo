#!/usr/bin/env python3
"""
Generate flux1_step0_dump.pt for comparison.
"""
import torch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader
import os
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.forward_context import set_forward_context

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

maybe_init_distributed_environment_and_model_parallel(
    tp_size=1,
    sp_size=1,
)

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Example: prepare some dummy inputs (match Flux config expectations)
latent_model_input = torch.randn(1, 64, 16, 16, 16, dtype=torch.bfloat16)
prompt_embeds = torch.randn(1, 77, 4096, dtype=torch.bfloat16)
pooled_projections = torch.randn(1, 768, dtype=torch.bfloat16)
timestep = torch.tensor([0], dtype=torch.long)
guidance = torch.tensor([0.0], dtype=torch.float32)

# Load Flux1
fastvideo_args = FastVideoArgs.from_kwargs(
    model_path=MODEL_ID,
    num_gpus=1,
    inference_mode=True,
    use_fsdp_inference=False,
    # precision="bf16",
)
loader = TransformerLoader(device="cuda")
# transformer = loader.load(MODEL_ID, fastvideo_args).to("cuda")
# model_path = "./flux1_model"
model_path = "./flux1_model/transformer"

transformer = loader.load(model_path, fastvideo_args)


# Forward pass to get official noise prediction
try:
    with torch.no_grad(), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=None,
    ):
        noise_pred_official = transformer(
            latent_model_input.to("cuda"),
            prompt_embeds.to("cuda"),
            pooled_projections=pooled_projections.to("cuda"),
            timestep=timestep.to("cuda"),
            guidance=guidance.to("cuda"),
        )
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# Save dump
torch.save({
    "latent_model_input": latent_model_input,
    "prompt_embeds": prompt_embeds,
    "pooled_projections": pooled_projections,
    "timestep": timestep,
    "guidance": guidance,
    "noise_pred_official": noise_pred_official.cpu()
}, "flux1_step0_dump.pt")

print("Dump saved as flux1_step0_dump.pt")
