#!/usr/bin/env python3
"""
Block-by-block comparison: find the first layer where FastVideo DiT diverges from official.
Runs both transformers with the same step-0 inputs, captures activations after each block,
and reports max/mean diff per block.

Requires: flux2_step0_dump.pt (run dump_flux2_step0.py first).
  python compare_flux2_dit_blocks.py [--model-path PATH]
"""
import argparse
import os
import sys

import torch

# FastVideo imports
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader

DUMP_PATH = "flux2_step0_dump.pt"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
THRESHOLD_MEAN = 0.1  # report block as diverged if mean abs diff > this


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


def _collect_fv_activations(transformer, latent, prompt_embeds, timestep_scaled, device, num_txt_tokens):
    """Run FastVideo transformer with hooks; return list of (block_name, tensor) per block."""
    activations = []

    def make_double_hook(name):
        def hook(_module, _inputs, outputs):
            # Flux2TransformerBlock returns (encoder_hidden_states, hidden_states)
            activations.append((name, outputs[1].detach().clone()))
        return hook

    def make_single_hook(name, ntxt):
        def hook(_module, _inputs, outputs):
            # Flux2SingleTransformerBlock returns full hidden_states; keep image part only
            out = outputs[0][:, ntxt:, :].detach().clone()
            activations.append((name, out))
        return hook

    for i, block in enumerate(transformer.transformer_blocks):
        block.register_forward_hook(make_double_hook(f"double_{i}"))
    for i, block in enumerate(transformer.single_transformer_blocks):
        block.register_forward_hook(make_single_hook(f"single_{i}", num_txt_tokens))

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        transformer(latent, prompt_embeds, timestep_scaled, guidance=None)

    return activations


def _collect_official_activations(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, num_txt_tokens):
    """Run official (diffusers) transformer with hooks; return list of (block_name, tensor) per block."""
    activations = []

    def make_double_hook(name):
        def hook(_module, _inputs, outputs):
            activations.append((name, outputs[1].detach().clone().float()))
        return hook

    def make_single_hook(name, ntxt):
        def hook(_module, _inputs, outputs):
            out = outputs[0][:, ntxt:, :].detach().clone().float()
            activations.append((name, out))
        return hook

    trans = pipe.transformer
    if not hasattr(trans, "transformer_blocks") or not hasattr(trans, "single_transformer_blocks"):
        print("Warning: official transformer has no transformer_blocks/single_transformer_blocks; skipping block capture.")
        return []

    for i, block in enumerate(trans.transformer_blocks):
        block.register_forward_hook(make_double_hook(f"double_{i}"))
    for i, block in enumerate(trans.single_transformer_blocks):
        block.register_forward_hook(make_single_hook(f"single_{i}", num_txt_tokens))

    dtype = next(pipe.transformer.parameters()).dtype
    latent = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    text_ids = text_ids.to(device)
    latent_ids = latent_ids.to(device)

    with torch.no_grad():
        with trans.cache_context("cond"):
            trans(
                hidden_states=latent,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=getattr(pipe, "attention_kwargs", None) or {},
                return_dict=False,
            )

    return activations


def main():
    parser = argparse.ArgumentParser(description="Compare FastVideo vs official DiT block-by-block.")
    parser.add_argument("--dump", default=DUMP_PATH, help="Path to flux2_step0_dump.pt")
    parser.add_argument("--model-path", default=None, help="Path to transformer dir or repo root")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(f"Missing dump file: {args.dump}. Run dump_flux2_step0.py first.")
        sys.exit(1)

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
    text_ids = data["text_ids"]
    latent_ids = data["latent_ids"]
    num_txt_tokens = prompt_embeds.shape[1]

    device = args.device
    model_path = args.model_path or _get_transformer_path(MODEL_ID)

    # 1. Load official transformer (via pipeline) and collect activations
    print("Loading official Flux2KleinPipeline and running forward with hooks ...")
    try:
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            from diffusers.pipelines.flux2 import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        official_activations = _collect_official_activations(
            pipe, latent, prompt_embeds, timestep_scaled,
            text_ids, latent_ids, device, num_txt_tokens,
        )
    except Exception as e:
        print(f"Official pipeline/hooks failed: {e}")
        official_activations = []

    # 2. Load FastVideo transformer and collect activations
    print("Loading FastVideo transformer and running forward with hooks ...")
    pipeline_config_cls = get_pipeline_config_cls_from_name(MODEL_ID)
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
    loader = TransformerLoader(device=device)
    transformer = loader.load(model_path, fastvideo_args)
    transformer = transformer.to(device)
    model_dtype = next(transformer.parameters()).dtype
    latent_fv = latent.to(device, dtype=model_dtype)
    prompt_embeds_fv = prompt_embeds.to(device, dtype=model_dtype)
    timestep_fv = timestep_scaled.to(device)

    fv_activations = _collect_fv_activations(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, num_txt_tokens,
    )
    # Cast to float for comparison
    fv_activations = [(n, t.cpu().float()) for n, t in fv_activations]

    # 3. Compare block-by-block
    if len(official_activations) == 0 or len(fv_activations) != len(official_activations):
        print(f"Block count mismatch: official={len(official_activations)}, FastVideo={len(fv_activations)}")
        if len(fv_activations) > 0:
            print("FastVideo block names:", [a[0] for a in fv_activations])
        return

    print("\n--- Block-by-block comparison ---")
    first_diverged = None
    for i, ((name_o, t_o), (name_fv, t_fv)) in enumerate(zip(official_activations, fv_activations)):
        if t_o.shape != t_fv.shape:
            print(f"  {i} {name_o} vs {name_fv}: SHAPE MISMATCH {t_o.shape} vs {t_fv.shape}")
            if first_diverged is None:
                first_diverged = i
            continue
        diff = (t_fv - t_o).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        ok = mean_d <= THRESHOLD_MEAN
        status = "ok" if ok else "DIVERGED"
        print(f"  {i} {name_o}: max_diff={max_d:.4f} mean_diff={mean_d:.4f} [{status}]")
        if first_diverged is None and not ok:
            first_diverged = i

    if first_diverged is not None:
        print(f"\n-> First diverged block index: {first_diverged} ({official_activations[first_diverged][0]})")
        print("  Debug that block (and inputs to it) in FastVideo vs official.")
    else:
        print("\n-> All block outputs within threshold (mean diff <= %s)." % THRESHOLD_MEAN)


if __name__ == "__main__":
    main()
