#!/usr/bin/env python3
"""
Compare FastVideo's Flux1 transformer to official step-0 output, layer by layer.

Requires: flux1_step0_dump.pt (run ex.py first).
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
    if os.path.isdir(model_id):
        if os.path.exists(os.path.join(model_id, "transformer", "config.json")):
            return os.path.join(model_id, "transformer")
        if os.path.exists(os.path.join(model_id, "config.json")):
            return model_id
    raise FileNotFoundError(
        f"Could not find transformer for {model_id}. "
        "Pass --model-path /path/to/transformer (or repo root)."
    )


def _register_hooks(model, activations: dict):
    """Register forward hooks on key layers to capture activations."""
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                activations[name] = output[0].detach().cpu()
        return hook
    
    # Hook embedders
    hooks.append(model.x_embedder.register_forward_hook(hook_fn("01_x_embedder")))
    hooks.append(model.context_embedder.register_forward_hook(hook_fn("02_context_embedder")))
    hooks.append(model.time_text_embed.register_forward_hook(hook_fn("03_time_text_embed")))
    
    # Hook transformer blocks
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(block.register_forward_hook(hook_fn(f"04_transformer_block_{i:02d}")))
    
    # Hook single transformer blocks
    for i, block in enumerate(model.single_transformer_blocks):
        hooks.append(block.register_forward_hook(hook_fn(f"05_single_transformer_block_{i:02d}")))
    
    # Hook final norm and projection
    hooks.append(model.norm_out.register_forward_hook(hook_fn("06_norm_out")))
    hooks.append(model.proj_out.register_forward_hook(hook_fn("07_proj_out")))
    
    return hooks


def _compare_layers(activations_fv: dict, activations_official: dict):
    """Compare layer-by-layer activations."""
    print("\n" + "="*90)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*90)
    print(f"{'Layer':<45} {'Shape':<25} {'Max Diff':<12} {'Mean Diff':<12}")
    print("-"*90)
    
    all_layers = sorted(set(activations_fv.keys()) | set(activations_official.keys()))
    
    results = []
    for layer_name in all_layers:
        if layer_name not in activations_fv:
            print(f"{layer_name:<45} {'MISSING in FastVideo':^62}")
            continue
        if layer_name not in activations_official:
            print(f"{layer_name:<45} {'MISSING in Official':^62}")
            continue
        
        fv_out = activations_fv[layer_name].float()
        off_out = activations_official[layer_name].float()
        
        shape_str = str(fv_out.shape)
        
        if fv_out.shape != off_out.shape:
            print(f"{layer_name:<45} Shape mismatch: {fv_out.shape} vs {off_out.shape}")
            results.append((layer_name, False, None, None))
            continue
        
        diff = (fv_out - off_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        match = torch.allclose(fv_out, off_out, rtol=1e-2, atol=1e-2)
        status = "✓" if match else "✗"
        
        print(f"{status} {layer_name:<43} {shape_str:<25} {max_diff:<12.2e} {mean_diff:<12.2e}")
        results.append((layer_name, match, max_diff, mean_diff))
    
    print("="*90)
    n_match = sum(1 for _, m, _, _ in results if m)
    n_total = len(results)
    print(f"\nSummary: {n_match}/{n_total} layers match within tolerance (rtol=1e-2, atol=1e-2)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer comparison of FastVideo Flux1 transformer.")
    parser.add_argument("--dump", default=DUMP_PATH, help="Path to flux1_step0_dump.pt")
    parser.add_argument("--model-path", default=None, help="Path to transformer dir or repo root")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(f"Missing dump file: {args.dump}. Run ex.py first.")
        sys.exit(1)

    # Initialize distributed environment first
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    # Load official dump
    print("Loading dump...")
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

    # Prepare inputs
    model_dtype = next(transformer.parameters()).dtype
    latent = latent.to(args.device, dtype=model_dtype)
    prompt_embeds = prompt_embeds.to(args.device, dtype=model_dtype)
    pooled_projections = pooled_projections.to(args.device, dtype=model_dtype)
    timestep = timestep.to(args.device)
    guidance = guidance.to(args.device)

    # ===== OFFICIAL FORWARD PASS (with hooks) =====
    print("\n[Step 1/2] Running official forward pass with hooks...")
    activations_official = {}
    hooks_official = _register_hooks(transformer, activations_official)
    
    try:
        with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            noise_pred_official_fv = transformer(
                latent,
                prompt_embeds,
                pooled_projections=pooled_projections,
                timestep=timestep,
                guidance=guidance,
            )
    finally:
        # Remove hooks
        for h in hooks_official:
            h.remove()

    # ===== RERUN WITH FRESH TRANSFORMER FOR CLEAN COMPARISON =====
    print("[Step 2/2] Running second forward pass with hooks for comparison...")
    
    # Reload transformer
    loader2 = TransformerLoader(device=args.device)
    transformer2 = loader2.load(model_path, fastvideo_args)
    
    activations_fv = {}
    hooks_fv = _register_hooks(transformer2, activations_fv)
    
    try:
        with torch.no_grad(), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            noise_pred_fv = transformer2(
                latent,
                prompt_embeds,
                pooled_projections=pooled_projections,
                timestep=timestep,
                guidance=guidance,
            )
    finally:
        # Remove hooks
        for h in hooks_fv:
            h.remove()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # Compare layers
    _compare_layers(activations_fv, activations_official)

    # Flatten outputs for final comparison
    if noise_pred_fv.dim() == 5:
        b, c, t, h, w = noise_pred_fv.shape
        noise_pred_fv = noise_pred_fv.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
    
    if noise_pred_official_fv.dim() == 5:
        b, c, t, h, w = noise_pred_official_fv.shape
        noise_pred_official_fv = noise_pred_official_fv.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)

    # Move to CPU for comparison
    noise_pred_fv = noise_pred_fv.cpu().float()
    noise_pred_official_fv = noise_pred_official_fv.cpu().float()

    # Final output comparison
    diff = (noise_pred_fv - noise_pred_official_fv).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(noise_pred_fv, noise_pred_official_fv, rtol=1e-2, atol=1e-2)

    print("\n" + "="*90)
    print("FINAL OUTPUT COMPARISON")
    print("="*90)
    print(f"  noise_pred shape:      {noise_pred_fv.shape}")
    print(f"  max abs diff:          {max_diff:.6e}")
    print(f"  mean abs diff:         {mean_diff:.6e}")
    print(f"  allclose match:        {allclose}")
    if allclose:
        print("  ✓ Final outputs match within tolerance.")
    else:
        print("  ✗ Final outputs differ.")
    print("="*90)


if __name__ == "__main__":
    main()
