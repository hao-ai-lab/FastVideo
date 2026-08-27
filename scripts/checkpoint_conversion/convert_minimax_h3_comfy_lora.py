"""Convert a ComfyUI-layout MiniMax-H3 LoRA into the layout FastVideo loads.

Most published H3 adapters (larryvrh's Turbo, Kijai's repacks, lightx2v's ComfyUI
builds) target `Comfy-Org/MiniMax-H3`, whose transformer is the same trained model as
the diffusers release under different names. FastVideo resolves adapter keys with the
model's own ``param_names_mapping``, which speaks diffusers, so those files reach no
layer at all.

Renaming is most of the job, but two rules cannot be expressed as a rename and are the
reason this is a script rather than a regex table in the arch config:

**Fused QKV.** ComfyUI keeps one ``attn.qkv_proj`` of shape ``[21504, 5376]``; diffusers
keeps three ``[7168, 5376]`` matrices. A LoRA states that delta as ``B @ A`` with
``A [r, 5376]`` shared and ``B [21504, r]``. Splitting ``B`` into three row blocks and
pairing each with the same ``A`` reproduces each projection's delta exactly -- the
factorization is over the input dimension, which the split leaves alone. Verified
against the two published checkpoints: the thirds are q, k, v in that order.

**Swapped SwiGLU halves.** ``mlp.fc1`` and ``ff.net.0.proj`` are both ``[28672, 5376]``,
so a rename looks correct and type-checks, but the two 14336-row halves are stored in
the opposite order in the two repacks (comfy row 0 equals diffusers row 14336, measured
on the released weights). Renaming without swapping applies the gate delta to the up
projection and vice versa: no error, no shape mismatch, just a quietly wrong model.

    python scripts/checkpoint_conversion/convert_minimax_h3_comfy_lora.py \\
        --input minimax_h3_turbo_v4_step600_ema.safetensors \\
        --output converted/adapter_model.safetensors
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOG = logging.getLogger("convert_minimax_h3_comfy_lora")

# MiniMax-H3 geometry. Named rather than inferred so a mismatched file fails loudly.
HEAD_DIM_TOTAL = 7168          # per-projection output width of q, k and v
FFN_HALF = 14336               # one SwiGLU half of the fc1 output

# Applied in order, first match wins. ``token_refiner.blocks`` is listed before the bare
# ``blocks`` rule, though the anchors already keep them apart.
#
# The lookahead matches either a following path segment or the end of the name: these run
# against module stems, and a top-level module like ``final_layer.adaln_proj.linear`` is
# the entire stem with nothing after it. Requiring a literal trailing dot silently skips
# exactly those, which is a miss that shows up only as one unmatched key at load.
PREFIX_RENAMES: list[tuple[str, str]] = [
    (r"^token_refiner\.blocks\.(\d+)(?=\.|$)", r"token_refiner.refiner_blocks.\1"),
    (r"^blocks\.(\d+)(?=\.|$)", r"transformer_blocks.\1"),
    (r"^final_layer\.adaln_proj\.linear(?=\.|$)", "norm_out.linear"),
    (r"^final_layer\.video_out(?=\.|$)", "proj_out"),
    (r"^final_layer\.audio_out(?=\.|$)", "audio_proj_out"),
    (r"^video_patch_proj(?=\.|$)", "proj_in"),
    (r"^audio_patch_proj(?=\.|$)", "audio_proj_in"),
    (r"^condition_proj(?=\.|$)", "context_embedder"),
    (r"^time_embedder\.proj_in(?=\.|$)", "time_embedder.linear_1"),
    (r"^time_embedder\.proj_out(?=\.|$)", "time_embedder.linear_2"),
]

# Module-level renames applied after the prefix rules. ``qkv_proj`` and ``fc1`` are
# absent here because they need tensor surgery, handled separately.
MODULE_RENAMES: list[tuple[str, str]] = [
    (r"\.attn\.out_proj$", ".attn.to_out.0"),
    (r"\.mlp\.fc2$", ".ff.net.2"),
]


def rename_module(name: str) -> str:
    for pattern, replacement in PREFIX_RENAMES:
        new = re.sub(pattern, replacement, name)
        if new != name:
            name = new
            break
    for pattern, replacement in MODULE_RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


def convert(input_path: Path, output_path: Path) -> dict:
    """Rewrite one ComfyUI-layout adapter. Returns a summary of what was emitted."""
    factors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    passthrough: dict[str, torch.Tensor] = {}

    with safe_open(input_path, framework="pt") as handle:
        source_meta = handle.metadata() or {}
        for key in handle.keys():
            if ".lora_A" in key or ".lora_down" in key:
                factors[key.split(".lora_A")[0].split(".lora_down")[0]]["A"] = handle.get_tensor(key)
            elif ".lora_B" in key or ".lora_up" in key:
                factors[key.split(".lora_B")[0].split(".lora_up")[0]]["B"] = handle.get_tensor(key)
            else:
                passthrough[key] = handle.get_tensor(key)

    out: dict[str, torch.Tensor] = {}
    stats = {"renamed": 0, "qkv_split": 0, "swiglu_swapped": 0, "skipped": 0}

    for module, pair in sorted(factors.items()):
        if "A" not in pair or "B" not in pair:
            LOG.warning("skipping %s: has only %s", module, "".join(sorted(pair)))
            stats["skipped"] += 1
            continue
        a, b = pair["A"], pair["B"]
        target = rename_module(module)

        if module.endswith(".attn.qkv_proj"):
            if b.shape[0] != 3 * HEAD_DIM_TOTAL:
                raise ValueError(f"{module}: expected lora_B rows {3 * HEAD_DIM_TOTAL}, got {b.shape[0]}")
            stem = target[:-len(".attn.qkv_proj")] + ".attn"
            for i, projection in enumerate(("to_q", "to_k", "to_v")):
                rows = b[i * HEAD_DIM_TOTAL:(i + 1) * HEAD_DIM_TOTAL]
                out[f"{stem}.{projection}.lora_A.weight"] = a.clone().contiguous()
                out[f"{stem}.{projection}.lora_B.weight"] = rows.contiguous()
            stats["qkv_split"] += 1
            continue

        if module.endswith(".mlp.fc1"):
            if b.shape[0] != 2 * FFN_HALF:
                raise ValueError(f"{module}: expected lora_B rows {2 * FFN_HALF}, got {b.shape[0]}")
            # The halves are stored in the opposite order in the two repacks.
            b = torch.cat([b[FFN_HALF:], b[:FFN_HALF]], dim=0)
            target = target[:-len(".mlp.fc1")] + ".ff.net.0.proj"
            stats["swiglu_swapped"] += 1

        out[f"{target}.lora_A.weight"] = a.contiguous()
        out[f"{target}.lora_B.weight"] = b.contiguous()
        stats["renamed"] += 1

    for key, tensor in passthrough.items():
        LOG.warning("carrying through unrecognized key unchanged: %s", key)
        out[rename_module(key)] = tensor

    metadata = {
        "format": "fastvideo-lora-v2",
        "converted_from": input_path.name,
        "source_layout": "comfyui-minimax-h3",
        "source_base_model": source_meta.get("base_model", "Comfy-Org/MiniMax-H3"),
        "application": "W = W_base + lora_B @ lora_A",
        "conversion": ("blocks->transformer_blocks; qkv_proj split into to_q/to_k/to_v sharing lora_A; "
                       "mlp.fc1 SwiGLU halves swapped for ff.net.0.proj; out_proj->to_out.0; fc2->ff.net.2"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output_path), metadata=metadata)
    output_path.chmod(0o644)

    LOG.info("%s -> %s", input_path.name, output_path)
    LOG.info("  %d modules renamed, %d QKV split into 3, %d SwiGLU halves swapped, %d skipped",
             stats["renamed"], stats["qkv_split"], stats["swiglu_swapped"], stats["skipped"])
    LOG.info("  %d tensors out (%.2f GiB)", len(out), output_path.stat().st_size / 2**30)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
