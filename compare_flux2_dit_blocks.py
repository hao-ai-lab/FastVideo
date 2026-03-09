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


def _capture_double0_qk_after_rope_fv(transformer, latent, prompt_embeds, timestep_scaled, device, freqs_cis):
    """Capture Q, K passed to the attention op (after RoPE) in the first double block."""
    captured = {}
    block0 = transformer.transformer_blocks[0]
    inner_attn = getattr(block0.attn, "attn", None)
    if inner_attn is None:
        return captured

    def pre_hook(_module, args, kwargs=None):
        try:
            if args and len(args) >= 2:
                captured["q"] = args[0].detach().clone().cpu().float()
                captured["k"] = args[1].detach().clone().cpu().float()
        except Exception:
            pass

    handle = inner_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
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


def _capture_double0_qk_after_rope_official(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device):
    """Capture Q, K passed to the attention op (after RoPE) in the first double block."""
    captured = {}
    trans = pipe.transformer
    if not hasattr(trans, "transformer_blocks") or len(trans.transformer_blocks) == 0:
        return captured
    block0 = trans.transformer_blocks[0]
    inner_attn = getattr(block0.attn, "attn", None)
    if inner_attn is None:
        return captured

    def pre_hook(_module, args, kwargs=None):
        try:
            if args and len(args) >= 2:
                captured["q"] = args[0].detach().clone().cpu().float()
                captured["k"] = args[1].detach().clone().cpu().float()
        except Exception:
            pass

    handle = inner_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
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


def _capture_single_block_input_fv(transformer, latent, prompt_embeds, timestep_scaled, device, single_block_index, freqs_cis):
    """Capture input and after norm+mod to the given single-stream block (one forward pass)."""
    captured = {}
    if single_block_index >= len(transformer.single_transformer_blocks):
        return captured

    def block_pre_hook(_module, args, kwargs=None):
        hs = (kwargs or {}).get("hidden_states")
        if hs is None and args and len(args) > 0:
            hs = args[0]
        if hs is not None:
            captured["input"] = hs.detach().clone().cpu().float()

    def attn_pre_hook(_module, args, kwargs=None):
        hs = (kwargs or {}).get("hidden_states")
        if hs is None and args and len(args) > 0:
            hs = args[0]
        if hs is not None:
            captured["after_norm_mod"] = hs.detach().clone().cpu().float()

    block = transformer.single_transformer_blocks[single_block_index]
    h_block = block.register_forward_pre_hook(block_pre_hook, with_kwargs=True)
    h_attn = block.attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
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
        h_block.remove()
        h_attn.remove()
    return captured


def _capture_single_block_input_official(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, single_block_index):
    """Capture input and after norm+mod to the given single-stream block (one forward pass)."""
    captured = {}
    trans = pipe.transformer
    if not hasattr(trans, "single_transformer_blocks") or single_block_index >= len(trans.single_transformer_blocks):
        return captured

    def block_pre_hook(_module, args, kwargs=None):
        hs = (kwargs or {}).get("hidden_states")
        if hs is None and args and len(args) > 0:
            hs = args[0]
        if hs is not None:
            captured["input"] = hs.detach().clone().cpu().float()

    def attn_pre_hook(_module, args, kwargs=None):
        hs = (kwargs or {}).get("hidden_states")
        if hs is None and args and len(args) > 0:
            hs = args[0]
        if hs is not None:
            captured["after_norm_mod"] = hs.detach().clone().cpu().float()

    block = trans.single_transformer_blocks[single_block_index]
    h_block = block.register_forward_pre_hook(block_pre_hook, with_kwargs=True)
    h_attn = block.attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
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
        h_block.remove()
        h_attn.remove()
    return captured


def _collect_fv_double_encoder_states(transformer, latent, prompt_embeds, timestep_scaled, device, freqs_cis):
    """Run FastVideo transformer; return list of (block_name, encoder_hidden_states) after each double block."""
    enc_states = []

    def make_hook(name):
        def hook(_module, _inputs, outputs):
            if len(outputs) >= 1 and outputs[0] is not None:
                enc_states.append((name, outputs[0].detach().clone()))
            else:
                enc_states.append((name, None))
        return hook

    for i, block in enumerate(transformer.transformer_blocks):
        block.register_forward_hook(make_hook(f"double_{i}"))

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        transformer(
            latent,
            prompt_embeds,
            timestep_scaled,
            guidance=None,
            freqs_cis=freqs_cis,
        )
    return enc_states


def _collect_official_double_encoder_states(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device):
    """Run official transformer; return list of (block_name, encoder_hidden_states) after each double block."""
    enc_states = []
    trans = pipe.transformer

    def make_hook(name):
        def hook(_module, _inputs, outputs):
            try:
                enc = outputs[0] if len(outputs) >= 1 else None
                if enc is not None and enc.dim() >= 2:
                    enc_states.append((name, enc.detach().clone().float()))
                else:
                    enc_states.append((name, None))
            except Exception:
                enc_states.append((name, None))
        return hook

    if not hasattr(trans, "transformer_blocks"):
        return []
    for i, block in enumerate(trans.transformer_blocks):
        block.register_forward_hook(make_hook(f"double_{i}"))

    dtype = next(trans.parameters()).dtype
    latent_d = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds_d = prompt_embeds.to(device, dtype=dtype)
    text_ids_d = text_ids.to(device)
    latent_ids_d = latent_ids.to(device)
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
    return enc_states


def _collect_fv_double4_intermediates(transformer, latent, prompt_embeds, timestep_scaled, device, freqs_cis, block_index=4):
    """Run FastVideo transformer with _debug_double_enc; return after_attn, after_ff, context_attn_output, and FF path tensors."""
    debug = {
        "block_index": block_index,
        "after_attn": [],
        "after_ff": [],
        "context_attn_output": [],
        "before_ff": [],
        "context_ff_output": [],
        "context_ff_update": [],
        "c_gate_mlp": [],
    }
    joint_kwargs = {"_debug_double_enc": debug}

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        transformer(
            latent,
            prompt_embeds,
            timestep_scaled,
            guidance=None,
            freqs_cis=freqs_cis,
            joint_attention_kwargs=joint_kwargs,
        )
    def first(key):
        return debug[key][0] if len(debug[key]) > 0 else None
    return (
        first("after_attn"),
        first("after_ff"),
        first("context_attn_output"),
        first("before_ff"),
        first("context_ff_output"),
        first("context_ff_update"),
        first("c_gate_mlp"),
    )


def _capture_official_double4_context_attn(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=4):
    """Run official transformer with hook on block 4's attn; return the context update (c_gate_msa * context_attn_output) for the text stream."""
    try:
        from diffusers.models.transformers import transformer_flux2
        Flux2Modulation = transformer_flux2.Flux2Modulation
    except Exception:
        return None
    trans = pipe.transformer
    if not hasattr(trans, "transformer_blocks") or block_index >= len(trans.transformer_blocks):
        return None
    captured_context_part = []

    def attn_hook(_module, _inputs, outputs):
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            captured_context_part.append(outputs[1].detach().clone().float())

    block4 = trans.transformer_blocks[block_index]
    handle = block4.attn.register_forward_hook(attn_hook)
    dtype = next(trans.parameters()).dtype
    latent_d = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds_d = prompt_embeds.to(device, dtype=dtype)
    text_ids_d = text_ids.to(device)
    latent_ids_d = latent_ids.to(device)
    try:
        timestep_1000 = timestep.to(latent_d.dtype) * 1000
        temb = trans.time_guidance_embed(timestep_1000, None)
        double_stream_mod_txt = trans.double_stream_modulation_txt(temb)
        (c_shift_msa, c_scale_msa, c_gate_msa), _ = Flux2Modulation.split(double_stream_mod_txt, 2)
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
    if not captured_context_part:
        return None
    context_before_gate = captured_context_part[0]
    c_gate_msa = c_gate_msa.to(device=context_before_gate.device, dtype=context_before_gate.dtype)
    if c_gate_msa.dim() == 2:
        c_gate_msa = c_gate_msa.unsqueeze(1)
    official_context_update = (c_gate_msa * context_before_gate).float()
    return official_context_update


def _capture_official_double4_context_ff(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=4):
    """Run official transformer with hook on block 4's ff_context; return (context_ff_output, context_ff_update, c_gate_mlp) for the text stream."""
    try:
        from diffusers.models.transformers import transformer_flux2
        Flux2Modulation = transformer_flux2.Flux2Modulation
    except Exception:
        return None, None, None
    trans = pipe.transformer
    if not hasattr(trans, "transformer_blocks") or block_index >= len(trans.transformer_blocks):
        return None, None, None
    captured_ff_output = []

    def ff_hook(_module, _inputs, outputs):
        captured_ff_output.append(outputs.detach().clone().float())

    block4 = trans.transformer_blocks[block_index]
    handle = block4.ff_context.register_forward_hook(ff_hook)
    dtype = next(trans.parameters()).dtype
    latent_d = latent.to(device, dtype=dtype)
    timestep = timestep_scaled.to(device)
    prompt_embeds_d = prompt_embeds.to(device, dtype=dtype)
    text_ids_d = text_ids.to(device)
    latent_ids_d = latent_ids.to(device)
    try:
        timestep_1000 = timestep.to(latent_d.dtype) * 1000
        temb = trans.time_guidance_embed(timestep_1000, None)
        double_stream_mod_txt = trans.double_stream_modulation_txt(temb)
        _, (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(double_stream_mod_txt, 2)
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
    if not captured_ff_output:
        return None, None, None
    ctx_ff_out = captured_ff_output[0]
    c_gate_mlp_raw = c_gate_mlp.to(device=ctx_ff_out.device, dtype=ctx_ff_out.dtype)
    c_gate_mlp = c_gate_mlp_raw
    if c_gate_mlp.dim() == 2:
        c_gate_mlp = c_gate_mlp.unsqueeze(1)
    official_context_ff_update = (c_gate_mlp * ctx_ff_out).float()
    return ctx_ff_out.float(), official_context_ff_update, c_gate_mlp_raw.float()


def _capture_official_double4_before_ff(pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=4):
    """Run official transformer with pre_hook on block 4's ff_context; return the input to ff_context (before_ff = after norm+mod)."""
    trans = pipe.transformer
    if not hasattr(trans, "transformer_blocks") or block_index >= len(trans.transformer_blocks):
        return None
    captured_before_ff = []

    def ff_pre_hook(_module, args):
        if args and len(args) > 0:
            captured_before_ff.append(args[0].detach().clone().float())

    block4 = trans.transformer_blocks[block_index]
    handle = block4.ff_context.register_forward_pre_hook(ff_pre_hook)
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
    return captured_before_ff[0] if captured_before_ff else None


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

    # Enforce same attention backend (SDPA) for reproducible comparison
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

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

    # Text stream (encoder_hidden_states) after each double block — find where 768 appears
    print("\n--- Text stream (encoder_hidden_states) after each double block ---")
    enc_fv = _collect_fv_double_encoder_states(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, freqs_cis
    )
    enc_official = []
    if len(official_activations) > 0:
        enc_official = _collect_official_double_encoder_states(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device
        )
    n_double = min(len(enc_fv), len(enc_official))
    for i in range(n_double):
        name_fv, t_fv = enc_fv[i]
        name_o, t_o = enc_official[i]
        if t_fv is None or t_o is None:
            print(f"  {name_fv}: missing (fv={t_fv is not None}, official={t_o is not None})")
            continue
        t_fv = t_fv.cpu().float() if t_fv.is_cuda else t_fv.float()
        t_o = t_o.cpu().float() if t_o.is_cuda else t_o.float()
        if t_fv.shape != t_o.shape:
            print(f"  {name_fv}: SHAPE MISMATCH {t_fv.shape} vs {t_o.shape}")
            continue
        diff = (t_fv - t_o).abs()
        print(f"  {name_fv}: shape={t_fv.shape} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")
    if n_double == 0 and (enc_fv or enc_official):
        print("  (one model has no double-block encoder states)")

    # double_4 intermediates: after context attn vs after context FF (find whether 768 is attn or FF path)
    TARGET_DOUBLE = 4
    if n_double > TARGET_DOUBLE and len(enc_official) > TARGET_DOUBLE:
        print(f"\n--- double_{TARGET_DOUBLE} intermediates (context attn vs context FF) ---")
        fv_after_attn, fv_after_ff, fv_context_attn, fv_before_ff, fv_context_ff_output, fv_context_ff_update, fv_c_gate_mlp = _collect_fv_double4_intermediates(
            transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, freqs_cis, block_index=TARGET_DOUBLE
        )
        _, off_enc4 = enc_official[TARGET_DOUBLE]
        if fv_after_attn is not None and fv_after_ff is not None and off_enc4 is not None:
            fv_after_attn = fv_after_attn.cpu().float() if fv_after_attn.is_cuda else fv_after_attn.float()
            fv_after_ff = fv_after_ff.cpu().float() if fv_after_ff.is_cuda else fv_after_ff.float()
            off_enc4 = off_enc4.cpu().float() if off_enc4.is_cuda else off_enc4.float()
            if fv_after_attn.shape == off_enc4.shape and fv_after_ff.shape == off_enc4.shape:
                d_attn_vs_off = (fv_after_attn - off_enc4).abs()
                d_ff_vs_off = (fv_after_ff - off_enc4).abs()
                print(f"  FV after_attn vs official double_{TARGET_DOUBLE} output: max_diff={d_attn_vs_off.max().item():.4f} mean_diff={d_attn_vs_off.mean().item():.4f}")
                print(f"  FV after_ff    vs official double_{TARGET_DOUBLE} output: max_diff={d_ff_vs_off.max().item():.4f} mean_diff={d_ff_vs_off.mean().item():.4f}")
                if d_attn_vs_off.max().item() > 100 and d_ff_vs_off.max().item() > 100:
                    print("  -> Bug likely in context ATTENTION path (both after_attn and after_ff already far from official).")
                elif d_attn_vs_off.max().item() < 50 and d_ff_vs_off.max().item() > 100:
                    print("  -> Bug likely in context FF path (after_attn is close; after_ff diverges).")
                else:
                    print("  -> Compare context_attn_output / to_add_out and context_ff_output for scale or bias.")
            else:
                print(f"  Shape mismatch: fv_after_attn={fv_after_attn.shape} fv_after_ff={fv_after_ff.shape} official={off_enc4.shape}")
        else:
            print("  Missing one of fv_after_attn, fv_after_ff, or official double_4 output.")

        # Compare context_attn update (the tensor added to encoder_hidden_states) FV vs official
        if fv_context_attn is not None:
            off_context_attn = _capture_official_double4_context_attn(
                pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=TARGET_DOUBLE
            )
            if off_context_attn is not None:
                fv_ctx = fv_context_attn.cpu().float() if fv_context_attn.is_cuda else fv_context_attn.float()
                off_ctx = off_context_attn.cpu().float() if off_context_attn.is_cuda else off_context_attn.float()
                if fv_ctx.shape == off_ctx.shape:
                    d_ctx = (fv_ctx - off_ctx).abs()
                    print(f"  FV context_attn_output (update) vs official: max_diff={d_ctx.max().item():.4f} mean_diff={d_ctx.mean().item():.4f}")
                else:
                    print(f"  context_attn_output shape mismatch: FV={fv_ctx.shape} official={off_ctx.shape}")
            else:
                print("  Could not capture official double_4 context_attn update.")

        # Context FF path: before_ff, context_ff_output, context_ff_update, c_gate_mlp
        print(f"\n--- double_{TARGET_DOUBLE} context FF path ---")
        off_ctx_ff_out, off_ctx_ff_update, off_c_gate_mlp = _capture_official_double4_context_ff(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=TARGET_DOUBLE
        )
        off_before_ff = _capture_official_double4_before_ff(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, block_index=TARGET_DOUBLE
        )
        if fv_before_ff is not None:
            print(f"  FV before_ff (after norm+mod): shape={fv_before_ff.shape}")
        if fv_before_ff is not None and off_before_ff is not None:
            fv_bf = fv_before_ff.cpu().float() if fv_before_ff.is_cuda else fv_before_ff.float()
            off_bf = off_before_ff.cpu().float() if off_before_ff.is_cuda else off_before_ff.float()
            if fv_bf.shape == off_bf.shape:
                d = (fv_bf - off_bf).abs()
                print(f"  FV before_ff vs official (input to ff_context): max_diff={d.max().item():.4f} mean_diff={d.mean().item():.4f}")
            else:
                print(f"  before_ff shape mismatch: FV={fv_bf.shape} official={off_bf.shape}")
        if fv_context_ff_output is not None and off_ctx_ff_out is not None:
            fv_o = fv_context_ff_output.cpu().float() if fv_context_ff_output.is_cuda else fv_context_ff_output.float()
            off_o = off_ctx_ff_out.cpu().float() if off_ctx_ff_out.is_cuda else off_ctx_ff_out.float()
            if fv_o.shape == off_o.shape:
                d = (fv_o - off_o).abs()
                print(f"  FV context_ff_output vs official: max_diff={d.max().item():.4f} mean_diff={d.mean().item():.4f}")
            else:
                print(f"  context_ff_output shape mismatch: FV={fv_o.shape} official={off_o.shape}")
        if fv_context_ff_update is not None and off_ctx_ff_update is not None:
            fv_u = fv_context_ff_update.cpu().float() if fv_context_ff_update.is_cuda else fv_context_ff_update.float()
            off_u = off_ctx_ff_update.cpu().float() if off_ctx_ff_update.is_cuda else off_ctx_ff_update.float()
            if fv_u.shape == off_u.shape:
                d = (fv_u - off_u).abs()
                print(f"  FV context_ff_update (c_gate_mlp*ff_out) vs official: max_diff={d.max().item():.4f} mean_diff={d.mean().item():.4f}")
            else:
                print(f"  context_ff_update shape mismatch: FV={fv_u.shape} official={off_u.shape}")
        if fv_c_gate_mlp is not None and off_c_gate_mlp is not None:
            fv_g = fv_c_gate_mlp.cpu().float() if fv_c_gate_mlp.is_cuda else fv_c_gate_mlp.float()
            off_g = off_c_gate_mlp.cpu().float() if off_c_gate_mlp.is_cuda else off_c_gate_mlp.float()
            print(f"  FV c_gate_mlp shape={fv_g.shape} official c_gate_mlp shape={off_g.shape}")
            if fv_g.shape == off_g.shape:
                d = (fv_g - off_g).abs()
                print(f"  FV c_gate_mlp vs official: max_diff={d.max().item():.4f} mean_diff={d.mean().item():.4f}")
            else:
                print("  c_gate_mlp shape mismatch (check broadcast in block).")

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

    # Q,K after RoPE (inputs to attention op) — if these differ, bug is in RoPE or Q/K proj; if match, bug is in attn op or output proj
    print("\n--- double_0 Q,K after RoPE (inputs to attention op) ---")
    qk_fv = _capture_double0_qk_after_rope_fv(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, freqs_cis
    )
    qk_official = {}
    if len(official_activations) > 0:
        qk_official = _capture_double0_qk_after_rope_official(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device
        )
    for key in ("q", "k"):
        a_fv = qk_fv.get(key)
        a_o = qk_official.get(key)
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

    # single_0 and double_7 diagnostics: where does the 768 diff appear?
    num_double = len(transformer.transformer_blocks)
    print("\n--- single_0: input (concat), after_norm_mod, output (image part) ---")
    in0_fv = _capture_single_block_input_fv(
        transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, 0, freqs_cis
    )
    in0_official = {}
    if len(official_activations) > 0:
        in0_official = _capture_single_block_input_official(
            pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, 0
        )
    for stage in ("input", "after_norm_mod"):
        a_fv = in0_fv.get(stage)
        a_o = in0_official.get(stage)
        if a_fv is None or a_o is None:
            print(f"  single_0 {stage}: missing (fv={a_fv is not None}, official={a_o is not None})")
            continue
        if a_fv.shape != a_o.shape:
            print(f"  single_0 {stage}: SHAPE MISMATCH {a_fv.shape} vs {a_o.shape}")
        else:
            diff = (a_fv - a_o).abs()
            ntxt = num_txt_tokens
            if a_fv.shape[1] > ntxt:
                d_txt = (a_fv[:, :ntxt] - a_o[:, :ntxt]).abs()
                d_img = (a_fv[:, ntxt:] - a_o[:, ntxt:]).abs()
                print(f"  single_0 {stage}: max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f} (text: max={d_txt.max().item():.4f}, image: max={d_img.max().item():.4f})")
            else:
                print(f"  single_0 {stage}: max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")
    if num_double < len(fv_activations) and num_double < len(official_activations):
        _, t0_fv = fv_activations[num_double]
        _, t0_o = official_activations[num_double]
        if t0_fv is not None and t0_o is not None and t0_fv.shape == t0_o.shape:
            t0_o = t0_o.cpu().float() if t0_o.is_cuda else t0_o.float()
            diff0 = (t0_fv - t0_o).abs()
            print(f"  single_0 output (image): max_diff={diff0.max().item():.4f} mean_diff={diff0.mean().item():.4f}")
    if num_double >= 1:
        _, d7_fv = fv_activations[num_double - 1]
        _, d7_o = official_activations[num_double - 1]
        if d7_fv is not None and d7_o is not None and d7_fv.shape == d7_o.shape:
            d7_o = d7_o.cpu().float() if d7_o.is_cuda else d7_o.float()
            diff7 = (d7_fv - d7_o).abs()
            print(f"  double_7 output (image): max_diff={diff7.max().item():.4f} mean_diff={diff7.mean().item():.4f}")

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
        # If first diverged is a single block, capture and compare INPUT to that block ( = output of previous block)
        num_double = len(transformer.transformer_blocks)
        if first_diverged >= num_double:
            single_idx = first_diverged - num_double  # e.g. first_diverged=8 -> single_3 -> index 3
            print(f"\n--- Input to first-diverged block (single_{single_idx} = output of single_{single_idx - 1}) ---")
            in_fv = _capture_single_block_input_fv(
                transformer, latent_fv, prompt_embeds_fv, timestep_fv, device, single_idx, freqs_cis
            )
            in_official = {}
            if len(official_activations) > 0:
                in_official = _capture_single_block_input_official(
                    pipe, latent, prompt_embeds, timestep_scaled, text_ids, latent_ids, device, single_idx
                )
            for stage in ("input", "after_norm_mod"):
                a_fv = in_fv.get(stage)
                a_o = in_official.get(stage)
                if a_fv is None or a_o is None:
                    print(f"  {stage}: missing (fv={a_fv is not None}, official={a_o is not None})")
                    continue
                if a_fv.shape != a_o.shape:
                    print(f"  {stage}: SHAPE MISMATCH {a_fv.shape} vs {a_o.shape}")
                else:
                    diff = (a_fv - a_o).abs()
                    print(f"  {stage}: shape={a_fv.shape} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")
            # Output of this block (from main activation collection)
            t_fv = fv_activations[first_diverged][1]
            t_o = official_activations[first_diverged][1]
            if t_fv is not None and t_o is not None and t_fv.shape == t_o.shape:
                t_fv = t_fv.cpu().float() if t_fv.is_cuda else t_fv.float()
                t_o = t_o.cpu().float() if t_o.is_cuda else t_o.float()
                diff = (t_fv - t_o).abs()
                print(f"  output: shape={t_fv.shape} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")
            else:
                print(f"  output: missing or shape mismatch")
            # Interpret: where does the diff jump?
            in_fv_a = in_fv.get("input")
            in_o_a = in_official.get("input")
            anm_fv = in_fv.get("after_norm_mod")
            anm_o = in_official.get("after_norm_mod")
            if (in_fv_a is not None and in_o_a is not None and anm_fv is not None and anm_o is not None
                    and t_fv is not None and t_o is not None and t_fv.shape == t_o.shape):
                d_in = (in_fv_a - in_o_a).abs().mean().item()
                d_anm = (anm_fv - anm_o).abs().mean().item()
                d_out = (t_fv.cpu().float() - t_o.cpu().float()).abs().mean().item()
                if d_anm > d_in * 1.5:
                    print("  -> Diff jumps at after_norm_mod: check norm / modulation (scale, shift).")
                elif d_out > d_anm * 1.2:
                    print("  -> Diff jumps at output: check attention + residual path (attn, gate, clip).")
                else:
                    print("  -> Diff grows gradually across the block.")
            if in_fv_a is not None and in_o_a is not None and (in_fv_a - in_o_a).abs().mean().item() > THRESHOLD_MEAN:
                print("  -> Input already diverged (drift from earlier blocks).")
            elif in_fv_a is not None and in_o_a is not None:
                print("  -> Input is close; divergence happens inside this block.")
    else:
        print("\n-> All block outputs within threshold (mean diff <= %s)." % THRESHOLD_MEAN)


if __name__ == "__main__":
    main()
