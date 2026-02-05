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


def _compute_freqs_cis(transformer, text_ids, latent_ids, device, dtype=None):
    """Build RoPE (cos, sin) from text and image position IDs using the model's rotary_emb.
    Position IDs are concatenated along sequence: [text_tokens; image_tokens].
    Returns (cos, sin) for use as freqs_cis in transformer forward.
    """
    # Concat along sequence dim: [B, T+L, ...]
    if text_ids.dim() == 2:
        # [B, T] -> treat as 1 axis, expand to n_axes
        text_ids = text_ids.unsqueeze(-1)  # [B, T, 1]
    if latent_ids.dim() == 2:
        latent_ids = latent_ids.unsqueeze(-1)  # [B, L, 1]
    combined = torch.cat([text_ids, latent_ids], dim=1)  # [B, T+L, n_axes]
    n_axes = transformer.rotary_emb.axes_dim
    if combined.shape[-1] != len(n_axes):
        # Pad or repeat last axis to match expected n_axes
        need = len(n_axes) - combined.shape[-1]
        if need > 0:
            combined = torch.cat([combined, combined[..., -1:].expand(-1, -1, need)], dim=-1)
        else:
            combined = combined[..., : len(n_axes)]
    # [num_tokens, n_axes] — keep on CPU so get_1d_rotary_pos_embed (no device arg) doesn't mix devices
    pos = combined.reshape(-1, combined.shape[-1]).float()
    with torch.no_grad():
        cos, sin = transformer.rotary_emb.forward_uncached(pos=pos)
    cos, sin = cos.to(device=device), sin.to(device)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return (cos, sin)


def _capture_double0_inputs_fv(transformer, latent, prompt_embeds, timestep_scaled, device, freqs_cis):
    """Run FastVideo transformer and capture (hidden_states, encoder_hidden_states) going into the first double block."""
    captured = {}

    def pre_hook(_module, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        hs = kwargs.get("hidden_states")
        enc = kwargs.get("encoder_hidden_states")
        if hs is None and len(args) > 0:
            hs = args[0]
        if enc is None and len(args) > 1:
            enc = args[1]
        if hs is not None:
            captured["hidden_states"] = hs.detach().clone().cpu().float()
        if enc is not None:
            captured["encoder_hidden_states"] = enc.detach().clone().cpu().float()

    handle = transformer.transformer_blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            transformer(
                latent,
                prompt_embeds,
                timestep_scaled,
                guidance=None,
                freqs_cis=freqs_cis,
            )
    finally:
        handle.remove()
    return captured


def _capture_double0_attn_output_fv(transformer, latent, prompt_embeds, timestep_scaled, device, freqs_cis):
    """Capture (attn_out_img, attn_out_txt) from the first double block's attention."""
    captured = {}

    def hook(_module, _inputs, outputs):
        try:
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                # Flux2Attention returns (hidden_states img, encoder_hidden_states txt)
                captured["attn_out_img"] = outputs[0].detach().clone().cpu().float()
                captured["attn_out_txt"] = outputs[1].detach().clone().cpu().float()
        except Exception:
            pass

    block0 = transformer.transformer_blocks[0]
    if not hasattr(block0, "attn"):
        return captured
    handle = block0.attn.register_forward_hook(hook)
    try:
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            transformer(
                latent,
                prompt_embeds,
                timestep_scaled,
                guidance=None,
                freqs_cis=freqs_cis,
            )
    finally:
        handle.remove()
    return captured


def _capture_double0_attn_output_official(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device):
    """Capture (attn_out_img, attn_out_txt) from the first double block's attention."""
    captured = {}
    trans = pipe.transformer

    def hook(_module, _inputs, outputs):
        try:
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                captured["attn_out_img"] = outputs[0].detach().clone().cpu().float()
                captured["attn_out_txt"] = outputs[1].detach().clone().cpu().float()
        except Exception:
            pass

    if not hasattr(trans, "transformer_blocks") or len(trans.transformer_blocks) == 0:
        return captured
    block0 = trans.transformer_blocks[0]
    if not hasattr(block0, "attn"):
        return captured
    handle = block0.attn.register_forward_hook(hook)
    dtype = next(trans.parameters()).dtype
    latent_d = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds_d = prompt_embeds.to(device, dtype=dtype)
    text_ids_d = text_ids.to(device)
    latent_ids_d = latent_ids.to(device)
    try:
        with torch.no_grad():
            with trans.cache_context("cond"):
                trans(
                    hidden_states=latent_d,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds_d,
                    txt_ids=text_ids_d,
                    img_ids=latent_ids_d,
                    joint_attention_kwargs=getattr(pipe, "attention_kwargs", None) or {},
                    return_dict=False,
                )
    finally:
        handle.remove()
    return captured


def _capture_double0_inputs_official(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device):
    """Run official transformer and capture (hidden_states, encoder_hidden_states) going into the first double block."""
    captured = {}
    trans = pipe.transformer

    def pre_hook(_module, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        hs = kwargs.get("hidden_states")
        enc = kwargs.get("encoder_hidden_states")
        if hs is None and len(args) > 0:
            hs = args[0]
        if enc is None and len(args) > 1:
            enc = args[1]
        if hs is not None:
            captured["hidden_states"] = hs.detach().clone().cpu().float()
        if enc is not None:
            captured["encoder_hidden_states"] = enc.detach().clone().cpu().float()

    if not hasattr(trans, "transformer_blocks") or len(trans.transformer_blocks) == 0:
        return captured
    handle = trans.transformer_blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    dtype = next(trans.parameters()).dtype
    latent_d = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds_d = prompt_embeds.to(device, dtype=dtype)
    text_ids_d = text_ids.to(device)
    latent_ids_d = latent_ids.to(device)
    try:
        with torch.no_grad():
            with trans.cache_context("cond"):
                trans(
                    hidden_states=latent_d,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds_d,
                    txt_ids=text_ids_d,
                    img_ids=latent_ids_d,
                    joint_attention_kwargs=getattr(pipe, "attention_kwargs", None) or {},
                    return_dict=False,
                )
    finally:
        handle.remove()
    return captured


def _collect_fv_activations(transformer, latent, prompt_embeds, timestep_scaled, device, num_txt_tokens, freqs_cis=None):
    """Run FastVideo transformer with hooks; return list of (block_name, tensor) per block."""
    activations = []

    def make_double_hook(name):
        def hook(_module, _inputs, outputs):
            # Flux2TransformerBlock returns (encoder_hidden_states, hidden_states)
            activations.append((name, outputs[1].detach().clone()))
        return hook

    def make_single_hook(name, ntxt):
        def hook(_module, _inputs, outputs):
            try:
                out = outputs[0]
                if out.dim() == 3 and out.shape[1] > ntxt:
                    out = out[:, ntxt:, :].detach().clone()
                elif out.dim() == 2:
                    out = out.unsqueeze(0).detach().clone()
                else:
                    out = out.detach().clone()
                activations.append((name, out))
            except Exception:
                activations.append((name, None))
        return hook

    for i, block in enumerate(transformer.transformer_blocks):
        block.register_forward_hook(make_double_hook(f"double_{i}"))
    for i, block in enumerate(transformer.single_transformer_blocks):
        block.register_forward_hook(make_single_hook(f"single_{i}", num_txt_tokens))

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        transformer(
            latent,
            prompt_embeds,
            timestep_scaled,
            guidance=None,
            freqs_cis=freqs_cis,
        )

    return activations


def _collect_official_activations(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, num_txt_tokens):
    """Run official (diffusers) transformer with hooks; return list of (block_name, tensor) per block."""
    activations = []

    def make_double_hook(name):
        def hook(_module, _inputs, outputs):
            try:
                t = outputs[1] if len(outputs) >= 2 else outputs[0]
                if t.dim() == 3:
                    activations.append((name, t.detach().clone().float()))
                elif t.dim() == 2:
                    activations.append((name, t.unsqueeze(0).detach().clone().float()))
                else:
                    activations.append((name, None))
            except Exception:
                activations.append((name, None))
        return hook

    def make_single_hook(name, ntxt):
        def hook(_module, _inputs, outputs):
            try:
                out = outputs[0]
                if out.dim() == 3 and out.shape[1] > ntxt:
                    activations.append((name, out[:, ntxt:, :].detach().clone().float()))
                elif out.dim() == 2:
                    activations.append((name, out.unsqueeze(0).detach().clone().float()))
                else:
                    activations.append((name, None))
            except Exception:
                activations.append((name, None))
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

    # RoPE: compute freqs_cis from text + image position IDs (same as official pipeline)
    freqs_cis = _compute_freqs_cis(
        transformer, text_ids, latent_ids, device, dtype=model_dtype
    )

    # Capture inputs to first double block to see where divergence starts
    print("\n--- Inputs to double_0 (before first block) ---")
    in_fv = _capture_double0_inputs_fv(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, freqs_cis
    )
    in_official = {}
    if len(official_activations) > 0:
        in_official = _capture_double0_inputs_official(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device
        )
    for key in ("hidden_states", "encoder_hidden_states"):
        a_fv = in_fv.get(key)
        a_o = in_official.get(key)
        if a_fv is None or a_o is None:
            print(f"  {key}: missing in one model (fv={a_fv is not None}, official={a_o is not None})")
            continue
        if a_fv.shape != a_o.shape:
            print(f"  {key}: SHAPE MISMATCH {a_fv.shape} vs {a_o.shape}")
            continue
        diff = (a_fv - a_o).abs()
        print(f"  {key}: shape={a_fv.shape} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")

    # Sub-layer: attention output of double_0 (narrows down where divergence is)
    print("\n--- double_0 attention output (inside first block) ---")
    attn_fv = _capture_double0_attn_output_fv(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, freqs_cis
    )
    attn_official = {}
    if len(official_activations) > 0:
        attn_official = _capture_double0_attn_output_official(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device
        )
    for key in ("attn_out_img", "attn_out_txt"):
        a_fv = attn_fv.get(key)
        a_o = attn_official.get(key)
        if a_fv is None or a_o is None:
            print(f"  {key}: missing (fv={a_fv is not None}, official={a_o is not None})")
            continue
        if a_fv.shape != a_o.shape:
            print(f"  {key}: SHAPE MISMATCH {a_fv.shape} vs {a_o.shape}")
            continue
        diff = (a_fv - a_o).abs()
        print(f"  {key}: shape={a_fv.shape} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")

    fv_activations = _collect_fv_activations(
        transformer,
        latent_fv,
        prompt_embeds_fv,
        timestep_fv,
        device,
        num_txt_tokens,
        freqs_cis=freqs_cis,
    )
    # Cast to float for comparison
    fv_activations = [(n, t.cpu().float()) for n, t in fv_activations]

    # 3. Compare block-by-block (compare first N blocks where N = min of the two)
    if len(official_activations) == 0 or len(fv_activations) == 0:
        print("No block activations to compare.")
        return
    n_compare = min(len(official_activations), len(fv_activations))
    if n_compare < len(official_activations) or n_compare < len(fv_activations):
        print(f"Block count: official={len(official_activations)}, FastVideo={len(fv_activations)} -> comparing first {n_compare} blocks")

    print("\n--- Block-by-block comparison ---")
    first_diverged = None
    for i in range(n_compare):
        name_o, t_o = official_activations[i]
        name_fv, t_fv = fv_activations[i]
        if t_o is None or t_fv is None:
            print(f"  {i} {name_o}: N/A (missing activation from one model)")
            if first_diverged is None:
                first_diverged = i
            continue
        if t_o.shape != t_fv.shape:
            print(f"  {i} {name_o} vs {name_fv}: SHAPE MISMATCH {t_o.shape} vs {t_fv.shape}")
            if first_diverged is None:
                first_diverged = i
            continue
        t_o = t_o.cpu() if t_o.is_cuda else t_o
        t_fv = t_fv.cpu() if t_fv.is_cuda else t_fv
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
