#!/usr/bin/env python3
"""
Compare FastVideo's Flux2 DiT to SGLang's transformer output.

Requires: flux2_step0_sglang_dump.pt (run dump_sglang_flux2_step0.py first).
Loads the SGLang dump, runs FastVideo's transformer with the same inputs,
and compares the output to SGLang's noise_pred.

  python compare_flux2_dit_sglang.py [--dump PATH] [--model-path PATH]
"""
import argparse
import os
import sys

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader

DUMP_PATH = "flux2_step0_sglang_dump.pt"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


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
    parser = argparse.ArgumentParser(
        description="Compare FastVideo DiT to SGLang step-0 output."
    )
    parser.add_argument(
        "--dump",
        default=DUMP_PATH,
        help="Path to flux2_step0_sglang_dump.pt",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to transformer dir or repo root",
    )
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(
            f"Missing dump file: {args.dump}. Run dump_sglang_flux2_step0.py first."
        )
        sys.exit(1)

    # Initialize distributed + model parallel
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    data = torch.load(args.dump, map_location="cpu", weights_only=True)
    latent = data["latent_model_input"]
    timestep_scaled = data["timestep_scaled"]
    prompt_embeds = data["prompt_embeds"]
    noise_pred_sglang = data["noise_pred_sglang"]

    model_path = args.model_path or _get_transformer_path(MODEL_ID)
    print(f"Loading FastVideo transformer from {model_path} ...")

    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=MODEL_ID,
        hsdp_shard_dim=1,
        hsdp_replicate_dim=1,
        num_gpus=1,
        inference_mode=True,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        pin_cpu_memory=False,
        dit_precision="bf16",
    )
    loader = TransformerLoader(device=args.device)
    transformer = loader.load(model_path, fastvideo_args)
    transformer = transformer.to(args.device)

    # Forward: same inputs as SGLang dump
    model_dtype = next(transformer.parameters()).dtype
    latent = latent.to(args.device, dtype=model_dtype)
    timestep_scaled = timestep_scaled.to(args.device)
    prompt_embeds = prompt_embeds.to(args.device, dtype=model_dtype)

    with torch.no_grad(), set_forward_context(
        current_timestep=0, attn_metadata=None
    ):
        noise_pred_fv = transformer(
            latent,
            prompt_embeds,
            timestep_scaled,
            guidance=None,
        )

    # FastVideo may return 5D -> reshape to (B, seq, C) for comparison
    if noise_pred_fv.dim() == 5:
        b, c, t, h, w = noise_pred_fv.shape
        noise_pred_fv = noise_pred_fv.permute(0, 2, 3, 4, 1).reshape(
            b, t * h * w, c
        )
    noise_pred_fv = noise_pred_fv.cpu().float()
    noise_pred_sglang = noise_pred_sglang.float()

    diff = (noise_pred_fv - noise_pred_sglang).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(
        noise_pred_fv, noise_pred_sglang, rtol=1e-2, atol=1e-2
    )

    print("\n--- Comparison (FastVideo vs SGLang) ---")
    print(f"  noise_pred_fv shape:    {noise_pred_fv.shape}")
    print(f"  noise_pred_sglang shape: {noise_pred_sglang.shape}")
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print(f"  allclose(rtol=1e-2, atol=1e-2): {allclose}")
    if allclose:
        print("  -> Outputs match within tolerance.")
    else:
        print(
            "  -> Outputs differ; debug DiT or input formatting "
            "(timestep, latent layout, RoPE, etc.)."
        )


if __name__ == "__main__":
    main()
