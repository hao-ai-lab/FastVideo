# SPDX-License-Identifier: Apache-2.0
"""Safetensors files for exported NVFP4 MiniMax-H3 video decoders.

Two formats share one file layout: tensors are prefixed ``decoder.`` or
``post_quant_conv.``; the header holds ``format`` and ``nvfp4_metadata`` (JSON).

* ``fastvideo_h3_decoder_nvfp4_deploy_v1``: NVFP4 linears as bf16 master
  weights, packed at load (``NVFP4DecoderLinear.freeze``).
* ``fastvideo_h3_decoder_nvfp4_packed_v1``: the frozen layers' buffers as
  written by ``freeze`` on a Blackwell GPU, loaded as is. Per NVFP4 linear
  ``<name>``: ``<name>.packed_weight`` (uint8 ``[out, in/2]``, two E2M1 codes per
  byte, FlashInfer ``fp4_quantize`` order), ``<name>.weight_inv_scale`` (E4M3
  per-16 block scales, FlashInfer ``SfLayout.layout_128x4`` swizzle),
  ``<name>.weight_global_sf`` (fp32 scalar), ``<name>.bias`` (in the compute
  dtype, which is exactly how inference consumes it) and ``<name>.input_amax``
  (fp32 static activation range). Every other tensor keeps the dtype it is
  stored with in the trained decoder, so outputs stay bit-identical.
"""
from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn as nn

from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (
    NVFP4_DECODER_DEPLOY_FORMAT,
    NVFP4DecoderLinear,
    check_num_linears,
    enable_fused_blocks,
    nvfp4_decoder_linear_names,
)

NVFP4_DECODER_PACKED_FORMAT = "fastvideo_h3_decoder_nvfp4_packed_v1"
NVFP4_DECODER_FORMATS = (NVFP4_DECODER_DEPLOY_FORMAT, NVFP4_DECODER_PACKED_FORMAT)
PACKED_BUFFERS = ("packed_weight", "weight_inv_scale", "weight_global_sf")
GROUPS = ("decoder", "post_quant_conv")
PACKED_LAYOUT = {
    "packed_weight": "uint8 [out_features, in_features/2], two E2M1 codes per byte (FlashInfer fp4_quantize)",
    "weight_inv_scale": "E4M3 per-16-element block scales, FlashInfer SfLayout.layout_128x4 (swizzled)",
    "weight_global_sf": "fp32 scalar, (448 * 6) / amax(bf16 weight)",
    "bias": "compute dtype (bf16)",
    "input_amax": "fp32 scalar static activation range (unused when act_scale == 'unit')",
    "other_tensors": "stored dtype of the trained decoder (fp32 norms, projections, registers, post_quant_conv)",
}


def write_nvfp4_decoder_safetensors(path: str, checkpoint: dict[str, Any]) -> None:
    """Write ``{"format", "metadata", "decoder", "post_quant_conv"}`` as one safetensors file."""
    from safetensors.torch import save_file

    if checkpoint["format"] not in NVFP4_DECODER_FORMATS:
        raise ValueError(f"unknown NVFP4 decoder format {checkpoint['format']!r}")
    tensors = {
        f"{group}.{key}": value.detach().contiguous().cpu().clone()
        for group in GROUPS for key, value in checkpoint[group].items()
    }
    header = {"format": checkpoint["format"], "nvfp4_metadata": json.dumps(checkpoint["metadata"])}
    save_file(tensors, path, metadata=header)


def read_nvfp4_decoder_safetensors(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Read either safetensors format; packed tensors go straight to ``device`` (bf16 masters stay on CPU)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt") as handle:
        header = handle.metadata() or {}
    fmt = header.get("format")
    if fmt not in NVFP4_DECODER_FORMATS:
        raise ValueError(f"{path} is not an NVFP4 decoder safetensors file (format: {fmt!r}); "
                         f"expected one of {NVFP4_DECODER_FORMATS}")
    if "nvfp4_metadata" not in header:
        raise ValueError(f"{path} is missing the nvfp4_metadata header")
    tensors = load_file(path, device=str(device) if fmt == NVFP4_DECODER_PACKED_FORMAT else "cpu")
    checkpoint: dict[str, Any] = {"format": fmt, "metadata": json.loads(header["nvfp4_metadata"])}
    for group in GROUPS:
        prefix = f"{group}."
        checkpoint[group] = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
    unknown = [key for key in tensors if not key.startswith(tuple(f"{group}." for group in GROUPS))]
    if unknown:
        raise ValueError(f"{path} has tensors outside {GROUPS}: {unknown[:5]}")
    return checkpoint


def packed_nvfp4_decoder_checkpoint(vae: nn.Module, metadata: dict[str, Any]) -> dict[str, Any]:
    """The packed-format checkpoint of a VAE whose NVFP4 decoder linears are all frozen."""
    decoder = vae.decoder
    state = dict(decoder.state_dict())
    layers = {name: module for name, module in decoder.named_modules() if isinstance(module, NVFP4DecoderLinear)}
    if not layers:
        raise ValueError("the VAE decoder has no NVFP4 linears")
    for name, layer in layers.items():
        if not layer.frozen:
            raise ValueError(f"NVFP4 linear {name} is not frozen; freeze the decoder before exporting packed weights")
        for buffer in PACKED_BUFFERS:
            state[f"{name}.{buffer}"] = getattr(layer, buffer)
        if layer.bias is not None:
            state[f"{name}.bias"] = layer.bias.detach().to(layer.compute_dtype)
    packed_metadata = {**metadata, "packed_layout": PACKED_LAYOUT, "num_linears": len(layers)}
    return {
        "format": NVFP4_DECODER_PACKED_FORMAT,
        "metadata": packed_metadata,
        "decoder": state,
        "post_quant_conv": dict(vae.post_quant_conv.state_dict()),
    }


def apply_packed_nvfp4_decoder(vae: nn.Module, checkpoint: dict[str, Any]) -> None:
    """Replace the (already depth-cut) dense decoder linears with frozen layers built from packed tensors.

    The dense linears are dropped one at a time, so no full-precision copy of a quantized weight is
    ever made. Everything else is loaded strictly.
    """
    metadata = checkpoint["metadata"]
    decoder = vae.decoder
    state = dict(checkpoint["decoder"])
    names = nvfp4_decoder_linear_names(decoder, tuple(metadata.get("skip_blocks") or ()))
    check_num_linears(metadata, len(names))
    for name in names:
        missing = [f"{name}.{buffer}" for buffer in PACKED_BUFFERS if f"{name}.{buffer}" not in state]
        if missing:
            raise ValueError(f"the packed NVFP4 decoder does not fit this VAE's architecture: missing {missing}")
        parent_name, _, child = name.rpartition(".")
        parent = decoder.get_submodule(parent_name)
        dense = parent[int(child)] if child.isdigit() else getattr(parent, child)
        packed = state.pop(f"{name}.packed_weight")
        if packed.shape[0] != dense.out_features:
            raise ValueError(f"the packed NVFP4 decoder does not fit this VAE's architecture: {name} has "
                             f"{packed.shape[0]} outputs, the VAE {dense.out_features}")
        layer = NVFP4DecoderLinear.from_packed(packed,
                                               state.pop(f"{name}.weight_inv_scale"),
                                               state.pop(f"{name}.weight_global_sf"),
                                               state.get(f"{name}.bias"),
                                               state.get(f"{name}.input_amax", torch.zeros((), device=packed.device)),
                                               in_features=dense.in_features,
                                               rotation_group=metadata.get("rotation_group"),
                                               compute_dtype=torch.bfloat16,
                                               act_scale=metadata.get("act_scale", "dynamic"))
        if child.isdigit():
            parent[int(child)] = layer
        else:
            setattr(parent, child, layer)
        del dense
    enable_fused_blocks(decoder)
    try:
        decoder.load_state_dict(state, strict=True)
        vae.post_quant_conv.load_state_dict(checkpoint["post_quant_conv"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"the packed NVFP4 decoder does not fit this VAE's architecture: {error}") from error
    vae.requires_grad_(False)
