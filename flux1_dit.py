#!/usr/bin/env python3
"""
Compare FastVideo's Flux1 transformer to official step-0 output, layer by layer.

Requires: flux1_step0_dump.pt (run ex.py first).
Uses FastVideo's component loader to load the transformer from the same checkpoint.

  python flux1_dit.py [--model-path PATH] [--mode {full,layer-by-layer}]
"""
import argparse
import os
import sys
import torch
from collections import OrderedDict

# FastVideo imports
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.forward_context import set_forward_context

DUMP_PATH = "flux1_step0_dump.pt"
MODEL_ID = "black-forest-labs/FLUX.1-dev"  # Update with the correct Flux1 checkpoint


def _get_transformer_path(model_id: str) -> str:
    """Resolve transformer component path (local or HF cache)."""
    try:
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=model_id)
        path = os.path.join(root, "transformer")
        if os.path.isdir(path):
            return path
    except Exception:
        pass
    if os.path.isdir(model_id):
        if os.path.exists(os.path.join(model_id, "transformer", "config.json")):
            return os.path.join(model_id, "transformer")
        if os.path.exists(os.path.join(model_id, "config.json")):
            return model_id
    raise FileNotFoundError(
        f"Could not find transformer for {model_id}. "
        "Pass --model-path /path/to/transformer (or repo root)."
    )


def main():
    parser = argparse.ArgumentParser(description="Compare FastVideo Flux1 transformer to official output.")
    parser.add_argument("--dump", default=DUMP_PATH, help="Path to flux1_step0_dump.pt")
    parser.add_argument("--model-path", default=None, help="Path to transformer dir or repo root")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(f"Missing dump file: {args.dump}. Run dump_flux1_step0.py first.")
        sys.exit(1)

    # Initialize distributed environment first
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    # Load official dump
    data = torch.load(args.dump, map_location="cpu", weights_only=True)
    latent = data["latent_model_input"]
    prompt_embeds = data["prompt_embeds"]
    pooled_projections = data["pooled_projections"]
    timestep = data["timestep"]
    guidance = data["guidance"]
    noise_pred_official = data["noise_pred_official"]

    # Determine transformer path
    model_path = args.model_path or _get_transformer_path(MODEL_ID)
    print(f"Loading FastVideo Flux1 transformer from {model_path} ...")

    # Configure FastVideo args
    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=MODEL_ID,
        num_gpus=1,
        inference_mode=True,
        use_fsdp_inference=False,
        dit_layerwise_offload=False,
    )

    # Load transformer
    loader = TransformerLoader(device=args.device)
    transformer = loader.load(model_path, fastvideo_args)

    # --- Forward pass using same inputs as official ---
    model_dtype = next(transformer.parameters()).dtype
    latent = latent.to(args.device, dtype=model_dtype)
    prompt_embeds = prompt_embeds.to(args.device, dtype=model_dtype)
    pooled_projections = pooled_projections.to(args.device, dtype=model_dtype)
    timestep = timestep.to(args.device)
    guidance = guidance.to(args.device)

    try:
        with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            noise_pred_fv = transformer(
                latent,
                prompt_embeds,
                pooled_projections=pooled_projections,
                timestep=timestep,
                guidance=guidance,
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # Flatten 5D output if necessary
    if noise_pred_fv.dim() == 5:
        b, c, t, h, w = noise_pred_fv.shape
        noise_pred_fv = noise_pred_fv.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
    
    if noise_pred_official.dim() == 5:
        b, c, t, h, w = noise_pred_official.shape
        noise_pred_official = noise_pred_official.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)

    # Move to CPU for comparison
    noise_pred_fv = noise_pred_fv.cpu().float()
    noise_pred_official = noise_pred_official.float()

    # Compare outputs
    diff = (noise_pred_fv - noise_pred_official).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(noise_pred_fv, noise_pred_official, rtol=1e-2, atol=1e-2)

    print("\n--- Comparison ---")
    print(f"  noise_pred_fv shape:      {noise_pred_fv.shape}")
    print(f"  noise_pred_official shape: {noise_pred_official.shape}")
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print(f"  allclose(rtol=1e-2, atol=1e-2): {allclose}")
    if allclose:
        print("  -> Outputs match within tolerance.")
    else:
        print("  -> Outputs differ; debug transformer or input formatting.")


if __name__ == "__main__":
    main()







# (venv) root@192cd51e0d57:/FastVideo# python flux1_dit.py 
# INFO 02-18 23:36:40 [__init__.py:109] ROCm platform is unavailable: No module named 'amdsmi'
# WARNING 02-18 23:36:40 [logger.py:122]  By default, logger.info(..) will only log from the local main process. Set logger.info(..., is_local_main_process=False) to log from all processes.
# INFO 02-18 23:36:40 [__init__.py:47] CUDA is available
# INFO 02-18 23:36:41 [parallel_state.py:976] Initializing distributed environment with world_size=1, device=cuda:0
# INFO 02-18 23:36:41 [parallel_state.py:788] Using nccl backend for CUDA platform
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# Fetching 29 files: 100%|█████████████████████████████████| 29/29 [00:00<00:00, 13527.00it/s]
# Loading FastVideo Flux1 transformer from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer ...
# INFO 02-18 23:36:41 [component_loader.py:765] transformer cls_name: FluxTransformer2DModel
# INFO 02-18 23:36:41 [component_loader.py:814] Loading model from 3 safetensors files: ['/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00001-of-00003.safetensors', '/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00003-of-00003.safetensors', '/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00002-of-00003.safetensors']
# INFO 02-18 23:36:41 [component_loader.py:825] Loading model from FluxTransformer2DModel, default_dtype: torch.bfloat16
# INFO 02-18 23:36:41 [fsdp_load.py:96] Loading model with default_dtype: torch.bfloat16
# INFO 02-18 23:36:41 [cuda.py:124] Trying FASTVIDEO_ATTENTION_BACKEND=None
# INFO 02-18 23:36:41 [cuda.py:126] Selected backend: None
# INFO 02-18 23:36:41 [cuda.py:278] Using Flash Attention backend.
# Loading safetensors checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
# Loading safetensors checkpoint shards: 100% Completed | 3/3 [00:00<00:00, 112.47it/s]

# INFO 02-18 23:36:45 [component_loader.py:858] Loaded model with 11.90B parameters

# --- Comparison ---
#   noise_pred_fv shape:      torch.Size([1, 4096, 64])
#   noise_pred_official shape: torch.Size([1, 4096, 64])
#   max abs diff:  0.000000
#   mean abs diff: 0.000000
#   allclose(rtol=1e-2, atol=1e-2): True
#   -> Outputs match within tolerance.
# (venv) root@192cd51e0d57:/FastVideo# 