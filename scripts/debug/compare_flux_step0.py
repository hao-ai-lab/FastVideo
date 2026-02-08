#!/usr/bin/env python3
"""
Load step-0 dump from dump_flux_step0.py, run FastVideo's Flux DiT with the same
inputs, and compare the output to the official noise_pred.

Requires: flux_step0_dump.pt (run dump_flux_step0.py first).
Uses FastVideo's component loader to load the transformer from the same checkpoint.

  python scripts/debug/compare_flux_step0.py [--model-path PATH]
"""
import argparse
import os
import sys
from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TransformerLoader

DUMP_PATH = "flux_step0_dump.pt"
MODEL_ID = "black-forest-labs/FLUX.1-dev"


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
    # Fallback: assume we're given repo root or transformer dir
    if os.path.isdir(model_id):
        if os.path.exists(os.path.join(model_id, "transformer", "config.json")):
            return os.path.join(model_id, "transformer")
        if os.path.exists(os.path.join(model_id, "config.json")):
            return model_id
    raise FileNotFoundError(
        f"Could not find transformer for {model_id}. "
        "Pass --model-path /path/to/transformer (or repo root)."
    )


def _load_dump(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FastVideo Flux DiT to official step-0 output."
    )
    parser.add_argument("--dump", default=DUMP_PATH, help="Path to flux_step0_dump.pt")
    parser.add_argument("--model-path", default=None, help="Path to transformer dir or repo root")
    parser.add_argument("--model-id", default=MODEL_ID, help="HF model ID (default: FLUX.1-dev)")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(f"Missing dump file: {args.dump}. Run dump_flux_step0.py first.")
        sys.exit(1)

    # Initialize distributed + model parallel so ColumnParallelLinear can use get_tp_world_size()
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    data = _load_dump(args.dump)
    latent = data.get("latent_model_input") or data.get("latent")
    timestep_scaled = (
        data.get("timestep_scaled")
        or data.get("timestep")
        or data.get("t")
    )
    prompt_embeds = (
        data.get("prompt_embeds")
        or data.get("encoder_hidden_states")
        or data.get("text_embeds")
    )
    pooled_projections = (
        data.get("pooled_projections")
        or data.get("pooled")
        or data.get("clip_pooled")
        or data.get("prompt_embeds_2")
    )
    noise_pred_official = data.get("noise_pred_official") or data.get("noise_pred")
    text_ids = data.get("text_ids")
    latent_image_ids = data.get("latent_image_ids")

    if latent is None or timestep_scaled is None or prompt_embeds is None or noise_pred_official is None:
        missing = [
            name
            for name, val in (
                ("latent_model_input", latent),
                ("timestep_scaled", timestep_scaled),
                ("prompt_embeds", prompt_embeds),
                ("noise_pred_official", noise_pred_official),
            )
            if val is None
        ]
        raise ValueError(f"Missing required fields in dump: {missing}")

    model_path = args.model_path or _get_transformer_path(args.model_id)
    print(f"Loading FastVideo transformer from {model_path} ...")

    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=args.model_id,
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

    model_dtype = next(transformer.parameters()).dtype
    latent = latent.to(args.device, dtype=model_dtype)
    timestep_scaled = timestep_scaled.to(args.device)

    if isinstance(prompt_embeds, (list, tuple)):
        prompt_embeds = [p.to(args.device, dtype=model_dtype) for p in prompt_embeds]
    else:
        prompt_embeds = prompt_embeds.to(args.device, dtype=model_dtype)

    if pooled_projections is not None:
        pooled_projections = pooled_projections.to(args.device, dtype=model_dtype)

    if text_ids is not None:
        text_ids = text_ids.to(args.device, dtype=model_dtype)
    if latent_image_ids is not None:
        latent_image_ids = latent_image_ids.to(args.device, dtype=model_dtype)

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        noise_pred_fv = transformer(
            hidden_states=latent,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_projections,
            timestep=timestep_scaled,
            guidance=None,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
        )

    # Official returns (B, seq, C); FastVideo may return 5D -> take same slice if needed
    if noise_pred_fv.dim() == 5:
        b, c, t, h, w = noise_pred_fv.shape
        noise_pred_fv = noise_pred_fv.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)

    noise_pred_fv = noise_pred_fv.cpu().float()
    noise_pred_official = noise_pred_official.float()

    diff = (noise_pred_fv - noise_pred_official).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(noise_pred_fv, noise_pred_official, rtol=1e-2, atol=1e-2)

    print("\n--- Comparison ---")
    print(f"  noise_pred_fv shape:       {noise_pred_fv.shape}")
    print(f"  noise_pred_official shape: {noise_pred_official.shape}")
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print(f"  allclose(rtol=1e-2, atol=1e-2): {allclose}")
    if allclose:
        print("  -> Outputs match within tolerance.")
    else:
        print("  -> Outputs differ; debug DiT or input formatting (timestep, latent layout, etc.).")


if __name__ == "__main__":
    main()
