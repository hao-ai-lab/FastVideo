#!/usr/bin/env python3
"""
Compare Diffusers vs FastVideo **context (text) branch linears** in double block 0:
  add_q_proj, add_k_proj, add_v_proj, to_add_out

Hooks the same modules on both sides after a step-0 forward. Inputs to block 0
should already match if the dump is consistent; this isolates ColumnParallelLinear
(add_*) and to_add_out vs nn.Linear.

  python compare_flux2_context_branch_double0.py [--dump PATH] [--device cuda]

Requires: flux2_step0_dump.pt (with text_ids, latent_ids), diffusers Flux2KleinPipeline,
editable FastVideo.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

DUMP_PATH = "flux2_step0_dump.pt"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _get_transformer_path(model_id: str) -> str:
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
    raise FileNotFoundError(f"Could not find transformer for {model_id}")


def _linear_out(module, x: torch.Tensor) -> torch.Tensor:
    """Match ColumnParallelLinear (tuple) vs nn.Linear (tensor)."""
    out = module(x)
    if isinstance(out, tuple):
        return out[0]
    return out


def _register_io_hooks(attn_module, storage: dict, prefix: str) -> list:
    handles = []

    def wire(name: str, child: torch.nn.Module):
        def pre(_m, args, _kwargs=None):
            if args and args[0] is not None:
                storage[f"{prefix}_{name}_in"] = args[0].detach().clone()

        def post(_m, _args, out):
            o = out[0] if isinstance(out, tuple) else out
            storage[f"{prefix}_{name}_out"] = o.detach().clone()

        handles.append(child.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(child.register_forward_hook(post))

    for name in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
        if hasattr(attn_module, name):
            wire(name, getattr(attn_module, name))
    return handles


def _max_mean(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        return float("nan"), float("nan")
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


def _flinear_vs_module(
    x: torch.Tensor, weight: torch.Tensor, bias, module_forward
) -> tuple[float, float]:
    """max/mean |module(x) - F.linear(x, w, b)| in float32."""
    with torch.no_grad():
        y_mod = module_forward(x)
        if isinstance(y_mod, tuple):
            y_mod = y_mod[0]
        y_man = F.linear(
            x.float(),
            weight.float(),
            None if bias is None else bias.float(),
        ).to(dtype=y_mod.dtype)
        d = (y_mod.float() - y_man.float()).abs()
        return d.max().item(), d.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare context-branch linears (double block 0) FV vs Diffusers."
    )
    parser.add_argument("--dump", default=DUMP_PATH)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isfile(args.dump):
        print(f"Missing dump: {args.dump}")
        sys.exit(1)

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
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
    text_ids = data.get("text_ids")
    latent_ids = data.get("latent_ids")
    if text_ids is None or latent_ids is None:
        print("Dump needs text_ids and latent_ids.")
        sys.exit(1)

    device = args.device
    dtype = torch.bfloat16

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers.pipelines.flux2 import Flux2KleinPipeline

    print("Loading Diffusers pipeline ...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    off_attn = pipe.transformer.transformer_blocks[0].attn
    off_store: dict = {}
    off_handles = _register_io_hooks(off_attn, off_store, "off")

    latent_d = latent.to(device, dtype=dtype)
    te_d = data["timestep_scaled"].to(device, dtype=dtype)
    if te_d.dim() == 1:
        te_d = te_d.view(1).expand(latent_d.shape[0])
    pe_d = prompt_embeds.to(device, dtype=dtype)

    with torch.no_grad():
        with pipe.transformer.cache_context("cond"):
            pipe.transformer(
                hidden_states=latent_d,
                timestep=te_d,
                guidance=None,
                encoder_hidden_states=pe_d,
                txt_ids=text_ids.to(device),
                img_ids=latent_ids.to(device),
                return_dict=False,
            )
    for h in off_handles:
        h.remove()

    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.flux_2 import compute_flux2_freqs_cis_from_ids
    from fastvideo.models.loader.component_loader import TransformerLoader

    model_path = args.model_path or _get_transformer_path(MODEL_ID)
    print("Loading FastVideo transformer ...")
    fv_args = FastVideoArgs.from_kwargs(
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
    fv = TransformerLoader(device=device).load(model_path, fv_args).to(device)
    model_dtype = next(fv.parameters()).dtype
    latent_f = latent.to(device, dtype=model_dtype)
    te_f = timestep_scaled.to(device)
    pe_f = prompt_embeds.to(device, dtype=model_dtype)
    freqs = compute_flux2_freqs_cis_from_ids(
        fv.rotary_emb, text_ids, latent_ids, device, dtype=model_dtype
    )

    fv_attn = fv.transformer_blocks[0].attn
    fv_store: dict = {}
    fv_handles = _register_io_hooks(fv_attn, fv_store, "fv")

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        fv(latent_f, pe_f, te_f, guidance=None, freqs_cis=freqs)
    for h in fv_handles:
        h.remove()

    print("\n--- Context branch linear I/O (double block 0) ---")
    for name in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
        ki, ko = f"off_{name}_in", f"off_{name}_out"
        fi, fo = f"fv_{name}_in", f"fv_{name}_out"
        if ki not in off_store or fi not in fv_store:
            print(f"  {name}: missing hook data (off_in={ki in off_store}, fv_in={fi in fv_store})")
            continue
        mi, ai = _max_mean(off_store[ki], fv_store[fi])
        mo, ao = _max_mean(off_store[ko], fv_store[fo])
        print(
            f"  {name}: input max|diff|={mi:.6f} mean={ai:.6f} | "
            f"output max|diff|={mo:.6f} mean={ao:.6f}"
        )

    print("\n--- F.linear sanity (FastVideo weights vs FV module output) ---")
    x_in = fv_store.get("fv_add_q_proj_in")
    if x_in is not None:
        m = fv_attn.add_q_proj
        w = m.weight
        b = getattr(m, "bias", None)
        mmx, mmn = _flinear_vs_module(x_in, w, b, lambda t: _linear_out(m, t))
        print(f"  add_q_proj F.linear vs module: max={mmx:.6e} mean={mmn:.6e}")

    print("\n--- Weight vs Diffusers (add_q_proj) ---")
    if hasattr(off_attn, "add_q_proj") and hasattr(fv_attn, "add_q_proj"):
        ow = off_attn.add_q_proj.weight.data.float()
        fw = fv_attn.add_q_proj.weight.data.float()
        d = (ow - fw).abs()
        print(f"  weight max|diff|={d.max().item():.6e} mean={d.mean().item():.6e}")


if __name__ == "__main__":
    main()
