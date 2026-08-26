# SPDX-License-Identifier: Apache-2.0
"""Repack the Wan2.2-Animate relighting LoRA for FastVideo's --lora-path.

The relighting LoRA (replace mode only: matches the character's lighting and
color tone to the target scene) ships in the *native* checkpoint repo
(``Wan-AI/Wan2.2-Animate-14B``) -- the official Diffusers-format repo does not
carry it. It comes in PEFT format, whose key style FastVideo's LoRA loader
does not read directly:

    base_model.model.blocks.0.self_attn.q.lora_A.weight   (adapter_model.safetensors)
    blocks.0.self_attn.q.lora_A.default.weight            (relighting_lora.ckpt)

This script only renames keys -- it does NOT modify any tensor. It strips the
PEFT ``base_model.model.`` prefix and ``.default.`` infix, leaving native Wan
names (``blocks.N.self_attn.q.lora_A``) that the loader resolves through the
model's ``lora_param_names_mapping`` -> ``param_names_mapping`` two-hop, and
emits one ``<layer>.lora_alpha`` scalar per target layer from
``adapter_config.json`` so the alpha/rank scaling matches PEFT.

Usage:
    python wan_animate_relight_lora.py --output ./wan_animate_relight_lora.safetensors
    # then: generator.set_lora_adapter("relight", "./wan_animate_relight_lora.safetensors")
    # (replace mode only -- do not load it for animation mode)

``--validate`` additionally maps every repacked key onto the FastVideo model's
parameter names on the meta device (needs torch + fastvideo importable).
"""
import argparse
import json
import os
import re

SOURCE_REPO = "Wan-AI/Wan2.2-Animate-14B"


def normalize_key(key: str) -> str:
    key = key.removeprefix("base_model.model.")
    key = key.replace(".default.", ".")
    return key


def load_state_dict(source_dir: str) -> tuple[dict, float | None]:
    """Prefer the PEFT safetensors; fall back to the torch .ckpt."""
    peft_weights = os.path.join(source_dir, "relighting_lora", "adapter_model.safetensors")
    peft_config = os.path.join(source_dir, "relighting_lora", "adapter_config.json")
    if os.path.isfile(peft_weights):
        from safetensors.torch import load_file
        alpha = None
        if os.path.isfile(peft_config):
            with open(peft_config) as f:
                alpha = json.load(f).get("lora_alpha")
        return load_file(peft_weights), alpha

    ckpt = os.path.join(source_dir, "relighting_lora.ckpt")
    assert os.path.isfile(ckpt), f"neither relighting_lora/ nor relighting_lora.ckpt under {source_dir}"
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    return state.get("state_dict", state), None


def repack(source: str, output: str, validate: bool) -> None:
    import torch
    from safetensors.torch import save_file

    if not os.path.isdir(source):
        from huggingface_hub import snapshot_download
        source = snapshot_download(source, allow_patterns=["relighting_lora*", "relighting_lora/*"])

    state_dict, alpha = load_state_dict(source)
    out: dict[str, torch.Tensor] = {}
    for key, weight in state_dict.items():
        new_key = normalize_key(key)
        assert ".lora_A." in new_key or ".lora_B." in new_key, f"unexpected LoRA key: {key} -> {new_key}"
        assert new_key not in out, f"key collision after normalisation: {new_key}"
        out[new_key] = weight.contiguous()

    layers = sorted({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in out})
    if alpha is not None:
        # One explicit alpha per layer: PEFT scales by alpha/rank, and leaving
        # it implicit would silently change the LoRA's strength.
        for layer in layers:
            out[layer + ".lora_alpha"] = torch.tensor(float(alpha))

    rank = next(v.shape[0] for k, v in out.items() if k.endswith(".lora_A.weight"))
    print(f"{len(layers)} target layers | rank {rank} | alpha {alpha}")
    for sample in layers[:3]:
        print(f"  e.g. {sample}")

    if validate:
        _validate(layers)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    save_file(out, output)
    print(f"wrote {output}")


def _validate(layers: list[str]) -> None:
    """Every target layer must resolve onto a FastVideo model parameter."""
    import torch

    from fastvideo.configs.models.dits.wan_animate import WanAnimateConfig
    from fastvideo.models.dits.wan_animate import WanAnimateTransformer3DModel
    from fastvideo.models.loader.utils import get_param_names_mapping

    lora_map = get_param_names_mapping(WanAnimateConfig().lora_param_names_mapping)
    param_map = get_param_names_mapping(WanAnimateConfig().param_names_mapping)
    with torch.device("meta"):
        model = WanAnimateTransformer3DModel(config=WanAnimateConfig(), hf_config={})
    model_params = {name.removesuffix(".weight") for name, _ in model.named_parameters()}

    unresolved = []
    for layer in layers:
        hf_name, _, _ = lora_map(layer + ".weight")
        target, _, _ = param_map(hf_name)
        if target.removesuffix(".weight") not in model_params:
            unresolved.append(f"{layer} -> {target}")
    if unresolved:
        for line in unresolved[:10]:
            print(f"  UNRESOLVED: {line}")
        raise SystemExit(f"validation FAILED: {len(unresolved)}/{len(layers)} layers do not resolve")
    print(f"validation PASSED: all {len(layers)} layers resolve to model parameters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SOURCE_REPO,
                        help="Official native checkpoint: HF repo id or local snapshot directory")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--validate", action="store_true",
                        help="Meta-device check that every layer resolves onto the FastVideo model")
    args = parser.parse_args()
    repack(args.source, args.output, args.validate)


if __name__ == "__main__":
    main()
