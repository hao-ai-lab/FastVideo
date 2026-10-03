# SPDX-License-Identifier: Apache-2.0
"""Convert a ModelOpt-unified NVFP4 MiniMax-H3 transformer into FastVideo's packed export.

ModelOpt's unified Hugging Face layout (e.g. ``FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4``)
stores each quantized linear as::

    <prefix>.weight          uint8          [out, in // 2]   two E2M1 values per byte
    <prefix>.weight_scale    float8_e4m3fn  [out, in // 16]  linear layout
    <prefix>.weight_scale_2  float32        []               amax(|W|) / (6 * 448)
    <prefix>.input_scale     float32        []               static activation scale

FastVideo's packed NVFP4 DiT export (``nvfp4_weights.safetensors``, read by
``load_minimax_h3_nvfp4_dit_export``) stores ``<module>::_nvfp4_weight`` (same
bytes), ``::_nvfp4_weight_scale`` (the same E4M3 bytes in FlashInfer's 128x4
swizzled layout), ``::_nvfp4_alpha`` (= ``weight_scale_2``) and
``::_weight_global_sf`` (= 1 / ``weight_scale_2``). The calibrated weights are
therefore carried over bit for bit; only the activation scale changes, because
FastVideo quantizes activations per call with a unit global scale and the
static ``input_scale`` is dropped.

``--quantize-attention`` additionally quantizes the dense BF16 attention
projections (``attn.to_{q,k,v,out}``) of every main block exactly as
``convert_model_to_nvfp4`` would at runtime, producing the full
``layer_profile="h3_dit"`` set. ``--quantize-gate`` (with it) also quantizes
each block's VSA compression gate, for ``layer_profile="h3_dit_vsa"``.
Without either the export holds the FFN linears only and must be loaded with
``layer_profile="h3_dit_ffn"``.

``--quantize-ffn`` takes the FFN linears from a dense BF16 source instead of a
ModelOpt export, with the same round-to-nearest weight math as the runtime
(ModelOpt's max calibration produces the same weight codes). ``--act-amax``
adds a calibrated static activation scale per linear
(``_nvfp4_input_global_sf`` = 448 * 6 / amax) from a JSON of input amax keyed
``b<block>.<module>`` (e.g. ``b3.ff.fc_in``); without it activations use the
unit global scale, which saturates inputs above 2688 (H3's ``ff.fc_out``).

Every exported linear is probed on random BF16 rows through the same
``mm_fp4`` path the loader runs; the relative error against a BF16 matmul with
the dequantized weight must stay under ``--max-probe-error`` (genuine W4A4
noise on random inputs is about 0.1; a wrong nibble order or scale layout
reads near 1.0). Needs a Blackwell GPU with FlashInfer.

Usage::

    python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py \\
        --src /path/to/FastH3-V2-NVFP4/transformer --dst /path/to/out/transformer [--quantize-attention]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPORT_FILENAME = "nvfp4_weights.safetensors"
_BLOCK_ATTN = re.compile(r"^transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0)$")
_BLOCK_FFN = re.compile(r"^transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)$")
_BLOCK_GATE = re.compile(r"^transformer_blocks\.\d+\.attn\.to_gate_compress$")
_RENAMES = ((re.compile(r"\.ff\.net\.0\.proj$"), ".ff.fc_in"), (re.compile(r"\.ff\.net\.2$"), ".ff.fc_out"),
            (re.compile(r"\.attn\.to_out\.0$"), ".attn.to_out"))
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fastvideo_module_name(diffusers_prefix: str) -> str:
    name = diffusers_prefix
    for pattern, replacement in _RENAMES:
        name = pattern.sub(replacement, name)
    return name


def dequantize_modelopt(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """E2M1 low nibble = even column, high nibble = odd column (ModelOpt / CUTLASS order)."""
    lut = _E2M1.to(packed.device)
    low = lut[(packed & 0x0F).long()]
    high = lut[(packed >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    block = scale.to(torch.float32).repeat_interleave(16, dim=1)
    return values * block * scale_2.to(torch.float32)


def _flashinfer():
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    try:
        from flashinfer import block_scale_interleave
    except ImportError:
        from flashinfer.fp4_quantization import block_scale_interleave
    return SfLayout, mm_fp4, nvfp4_quantize, block_scale_interleave


def probe(buffers: dict[str, torch.Tensor], reference: torch.Tensor, rows: int = 256) -> float:
    """Relative error of the loader's mm_fp4 path against BF16 x @ W_ref^T."""
    SfLayout, mm_fp4, nvfp4_quantize, _ = _flashinfer()
    device = buffers["_nvfp4_weight"].device
    x = torch.randn(rows, reference.shape[1], device=device, dtype=torch.bfloat16)
    unit = torch.tensor(1.0, device=device, dtype=torch.float32)
    x_fp4, x_scale = nvfp4_quantize(x, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    out = mm_fp4(x_fp4, buffers["_nvfp4_weight"].T, x_scale, buffers["_nvfp4_weight_scale"].T,
                 buffers["_nvfp4_alpha"] / unit, torch.bfloat16, None, backend="auto")
    ref = x.float() @ reference.float().T
    return ((out.float() - ref).norm() / ref.norm()).item()


def convert_modelopt_linear(weight, scale, scale_2, device) -> tuple[dict[str, torch.Tensor], torch.Tensor, float]:
    """Carry the calibrated bytes over; return (buffers, dequantized weight, scale-byte agreement).

    The agreement compares the swizzled ModelOpt scales with the scales
    ``nvfp4_quantize`` derives from the dequantized weight. It is a layout check
    (identical swizzle and shape) and is near 1.0 when ModelOpt used max
    calibration; blocks whose largest code is below 6 legitimately differ.
    """
    SfLayout, _, nvfp4_quantize, block_scale_interleave = _flashinfer()
    weight = weight.to(device)
    scale = scale.to(device)
    scale_2 = scale_2.to(device=device, dtype=torch.float32).reshape(())
    reference = dequantize_modelopt(weight, scale, scale_2)
    _, layout_ref = nvfp4_quantize(reference.to(torch.bfloat16), 1.0 / scale_2, sfLayout=SfLayout.layout_128x4,
                                   do_shuffle=False)
    swizzled = block_scale_interleave(scale.view(torch.uint8).contiguous()).reshape(layout_ref.shape)
    agreement = (swizzled.view(torch.uint8) == layout_ref.view(torch.uint8)).float().mean().item()
    buffers = {
        "_nvfp4_weight": weight.contiguous(),
        "_nvfp4_weight_scale": swizzled.view(layout_ref.dtype).contiguous(),
        "_nvfp4_alpha": scale_2.clone(),
        "_weight_global_sf": (1.0 / scale_2).to(torch.bfloat16),
    }
    return buffers, reference, agreement


def quantize_dense_linear(weight: torch.Tensor, device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Same math as ``nvfp4_config.convert_model_to_nvfp4``."""
    SfLayout, _, nvfp4_quantize, _ = _flashinfer()
    weight = weight.to(device=device, dtype=torch.bfloat16)
    global_sf = (448 * 6) / weight.float().abs().nan_to_num().max()
    fp4_w, fp4_s = nvfp4_quantize(weight, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    global_sf = torch.as_tensor(global_sf, device=device, dtype=torch.float32)
    buffers = {"_nvfp4_weight": fp4_w, "_nvfp4_weight_scale": fp4_s, "_nvfp4_alpha": (1.0 / global_sf).float(),
               "_weight_global_sf": global_sf.to(torch.bfloat16)}
    return buffers, weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--quantize-attention", action="store_true")
    parser.add_argument("--quantize-gate", action="store_true", help="also quantize attn.to_gate_compress")
    parser.add_argument("--quantize-ffn", action="store_true", help="quantize the FFN linears from a bf16 source")
    parser.add_argument("--act-amax", type=Path, help="JSON of calibrated input amax per linear")
    parser.add_argument("--max-probe-error", type=float, default=0.3)
    parser.add_argument("--dense-shard-gb", type=float, default=5.0)
    args = parser.parse_args()
    device = torch.device("cuda")
    args.dst.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    modelopt = sorted(k[:-len(".weight_scale_2")] for k in weight_map if k.endswith(".weight_scale_2"))
    modelopt_keys = {f"{p}.{s}" for p in modelopt for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
    if args.quantize_gate and not args.quantize_attention:
        parser.error("--quantize-gate requires --quantize-attention")
    selected = ([_BLOCK_ATTN.pattern] if args.quantize_attention else []) + (
        [_BLOCK_GATE.pattern] if args.quantize_gate else []) + ([_BLOCK_FFN.pattern] if args.quantize_ffn else [])
    dense_pattern = re.compile("|".join(selected)) if selected else None
    attention = sorted(k[:-len(".weight")] for k in weight_map
                       if k.endswith(".weight") and dense_pattern.match(k[:-len(".weight")])) if dense_pattern else []
    if args.quantize_ffn and any(_BLOCK_FFN.match(p) for p in modelopt):
        parser.error("--quantize-ffn needs a bf16 source; this one already holds ModelOpt FFN linears")
    amax_table = None
    if args.act_amax:
        raw = json.loads(args.act_amax.read_text())
        amax_table = {k: float(v["all"] if isinstance(v, dict) else v) for k, v in raw.items()}
    attention_keys = {f"{p}.weight" for p in attention}

    readers = {shard: safe_open(str(args.src / shard), framework="pt", device="cpu") for shard in set(weight_map.values())}
    tensor = lambda key: readers[weight_map[key]].get_tensor(key)

    export: dict[str, torch.Tensor] = {}
    worst = 0.0
    agreements: list[float] = []
    attention_set = set(attention)
    for prefix in modelopt + attention:
        if prefix in attention_set:
            buffers, reference = quantize_dense_linear(tensor(f"{prefix}.weight"), device)
        else:
            buffers, reference, agreement = convert_modelopt_linear(tensor(f"{prefix}.weight"),
                                                                    tensor(f"{prefix}.weight_scale"),
                                                                    tensor(f"{prefix}.weight_scale_2"), device)
            agreements.append(agreement)
        error = probe(buffers, reference)
        worst = max(worst, error)
        if error > args.max_probe_error:
            raise SystemExit(f"probe error {error:.3f} on {prefix} exceeds {args.max_probe_error}; nothing written")
        module = fastvideo_module_name(prefix)
        if amax_table is not None:
            block = re.match(r"transformer_blocks\.(\d+)\.(.+)$", module)
            key = f"b{block.group(1)}.{block.group(2)}"
            if key not in amax_table:
                raise SystemExit(f"--act-amax has no entry {key!r} for {module}; nothing written")
            buffers["_nvfp4_input_global_sf"] = torch.tensor((448.0 * 6.0) / max(amax_table[key], 1e-12),
                                                             dtype=torch.float32)
        for name, value in buffers.items():
            export[f"{module}::{name}"] = value.cpu()
    save_file(export, str(args.dst / EXPORT_FILENAME))

    dense_keys = [k for k in weight_map if k not in modelopt_keys and k not in attention_keys]
    shard, shard_bytes, shards = {}, 0, []
    for key in dense_keys:
        value = tensor(key)
        shard[key] = value
        shard_bytes += value.numel() * value.element_size()
        if shard_bytes >= args.dense_shard_gb * 1e9:
            shards.append(shard)
            shard, shard_bytes = {}, 0
    if shard:
        shards.append(shard)
    new_map = {}
    for i, part in enumerate(shards, 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(part, str(args.dst / name))
        new_map.update({k: name for k in part})
    (args.dst / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": new_map}, indent=2) + "\n")
    config = json.loads((args.src / "config.json").read_text())
    config.pop("quantization_config", None)
    (args.dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for extra in args.src.iterdir():
        if extra.suffix not in (".safetensors", ".json"):
            shutil.copy2(extra, args.dst / extra.name)
    print(json.dumps({"exported_linears": len(modelopt) + len(attention), "modelopt_linears": len(modelopt),
                      "quantized_dense_linears": len(attention), "worst_probe_error": round(worst, 4),
                      "static_activation_scales": len(attention) + len(modelopt) if amax_table else 0,
                      "gate_linears": sum(1 for p in attention if _BLOCK_GATE.match(p)),
                      "min_scale_byte_agreement": round(min(agreements), 4) if agreements else None,
                      "mean_scale_byte_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
                      "dense_tensors": len(dense_keys), "dense_shards": len(shards)}))


if __name__ == "__main__":
    main()
