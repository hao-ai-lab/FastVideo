# SPDX-License-Identifier: Apache-2.0
"""Convert a `stable_audio_tools` raw checkpoint into a Diffusers-style
per-component layout consumable by FastVideo's standard
`ComposedPipelineBase` loader.

Why: the published Stability AI Stable Audio Open repos ship a single
monolithic `model.safetensors` containing DiT (`model.model.*`),
VAE / pretransform (`pretransform.model.*`), and NumberConditioner
(`conditioner.conditioners.*`) state dicts under shared prefixes.
FastVideo's standard loader expects the Diffusers per-subfolder layout
(`model_index.json` + per-component subfolders). This script splits the
monolithic checkpoint, drops the host-pipeline prefixes, normalizes
LayerNorm `gamma`/`beta` keys to `weight`/`bias`, and writes a
Diffusers-style tree.

Usage::

    python scripts/checkpoint_conversion/stable_audio_to_diffusers.py \
        --src stabilityai/stable-audio-open-1.0 \
        --dst converted_weights/stable_audio_open_1_0_diffusers

    # Or with a local snapshot directory:
    python scripts/checkpoint_conversion/stable_audio_to_diffusers.py \
        --src /path/to/stable-audio-open-1.0 \
        --dst converted_weights/stable_audio_open_small_diffusers

The output tree is::

    <dst>/
    ├── model_index.json
    ├── transformer/{config.json, diffusion_pytorch_model.safetensors}
    ├── vae/{config.json, diffusion_pytorch_model.safetensors}
    ├── conditioner/{config.json, diffusion_pytorch_model.safetensors}
    ├── text_encoder/  (copied from src if present)
    ├── tokenizer/     (copied from src if present)
    └── scheduler/     (copied from src if present)

Push to the FastVideo HF org with `huggingface-cli upload`. Confirm
visibility (`--private` for staging, omit for public release)::

    huggingface-cli upload --private FastVideo/stable-audio-open-1.0-Diffusers \
        converted_weights/stable-audio-open-1.0-Diffusers
    huggingface-cli upload --private FastVideo/stable-audio-open-small-Diffusers \
        converted_weights/stable-audio-open-small-Diffusers
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

# Map host-pipeline prefix → output subfolder. Order matters only for
# logging.
_COMPONENT_PREFIXES: dict[str, str] = {
    "model.model.": "transformer",
    "pretransform.model.": "vae",
    "conditioner.": "conditioner",
}

# Subfolders we copy verbatim from the source repo. The `vae/` subfolder
# in `stabilityai/stable-audio-open-1.0` already ships in Diffusers
# format with the key naming our `OobleckVAE` expects (`decoder.block.X.
# conv_t1.weight_g` etc.) — preferred over the raw `pretransform.model.*`
# extraction below, which uses a different module nesting (`decoder.
# layers.X.layers.Y.*`) that would need a separate remap.
_PASSTHROUGH_SUBFOLDERS = ("text_encoder", "tokenizer", "scheduler", "vae")

_MODEL_INDEX = {
    "_class_name": "StableAudioPipeline",
    # `ComposedPipelineBase._load_config` requires this field; the value
    # is informational (we don't pin a Diffusers version).
    "_diffusers_version": "0.30.0",
    "_fastvideo_converted_from": None,  # filled in at write time
    "transformer": ["fastvideo.models.dits.stable_audio", "StableAudioDiT"],
    "vae": ["fastvideo.models.vaes.oobleck", "OobleckVAE"],
    "conditioner": [
        "fastvideo.models.encoders.stable_audio_conditioner",
        "StableAudioMultiConditioner",
    ],
    "text_encoder": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "scheduler": ["diffusers", "CosineDPMSolverMultistepScheduler"],
}


def _resolve_src(src: str) -> Path:
    """Accept either a HF repo id or a local directory."""
    if os.path.isdir(src):
        return Path(src)
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
             or os.environ.get("HF_API_KEY"))
    return Path(snapshot_download(repo_id=src, token=token))


def _rename_layernorm_keys(state: dict[str, Any]) -> dict[str, Any]:
    """`nn.LayerNorm` ships `gamma`/`beta` in the official checkpoint;
    torch's `nn.LayerNorm` uses `weight`/`bias`."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k.endswith(".gamma"):
            out[k[: -len(".gamma")] + ".weight"] = v
        elif k.endswith(".beta"):
            out[k[: -len(".beta")] + ".bias"] = v
        else:
            out[k] = v
    return out


def _split_state_dict(src_safetensors: Path) -> dict[str, dict[str, Any]]:
    """Read `model.safetensors` and bucket its keys by the host-pipeline
    prefix listed in `_COMPONENT_PREFIXES`."""
    buckets: dict[str, dict[str, Any]] = {name: {} for name in _COMPONENT_PREFIXES.values()}
    skipped = 0
    with safe_open(str(src_safetensors), framework="pt") as f:
        for k in f.keys():
            for prefix, subfolder in _COMPONENT_PREFIXES.items():
                if k.startswith(prefix):
                    buckets[subfolder][k[len(prefix):]] = f.get_tensor(k)
                    break
            else:
                skipped += 1
    if skipped:
        print(f"  skipped {skipped} keys (no matching prefix)")
    for subfolder, state in buckets.items():
        if subfolder == "transformer":
            buckets[subfolder] = _rename_layernorm_keys(state)
        print(f"  {subfolder}: {len(buckets[subfolder])} keys")
    return buckets


def _component_config(model_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull per-component sizing out of the official `model_config.json`
    so the converted repo records the architecture authoritatively."""
    diff = model_config["model"]["diffusion"]
    pre = model_config["model"]["pretransform"]["config"]
    cond = model_config["model"]["conditioning"]
    # Translate upstream `model_config.json` field names to FastVideo's
    # `StableAudioArchConfig` fields. The standard `TransformerLoader`
    # calls `dit_config.update_model_arch(json_dict)` which raises on
    # unknown keys.
    diff_cfg = dict(diff["config"])
    if "num_heads" in diff_cfg:
        diff_cfg["num_attention_heads"] = diff_cfg.pop("num_heads")
    diff_cfg.pop("transformer_type", None)  # always "continuous_transformer"
    diff_cfg.pop("attn_kwargs", None)  # small variant; surfaced via `qk_norm` instead
    # `cross_attention_cond_ids` / `global_cond_ids` belong to the
    # conditioner, not the DiT — keep them out of transformer/config.json.
    transformer_cfg = {
        "_class_name": "StableAudioDiT",
        **diff_cfg,
    }
    vae_cfg = {
        "_class_name": "OobleckVAE",
        "io_channels": pre["io_channels"],
        "latent_dim": pre["latent_dim"],
        "downsampling_ratio": pre["downsampling_ratio"],
        "encoder_config": pre["encoder"]["config"],
        "decoder_config": pre["decoder"]["config"],
    }
    conditioner_cfg = {
        "_class_name": "StableAudioMultiConditioner",
        "cond_dim": cond["cond_dim"],
        "configs": cond["configs"],
        "cross_attention_cond_ids": diff["cross_attention_cond_ids"],
        "global_cond_ids": diff["global_cond_ids"],
    }
    return {
        "transformer": transformer_cfg,
        "vae": vae_cfg,
        "conditioner": conditioner_cfg,
    }


def _write_component(dst_dir: Path, name: str, state: dict[str, Any], config: dict[str, Any]) -> None:
    sub = dst_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    save_file(state, str(sub / "diffusion_pytorch_model.safetensors"))
    with open(sub / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  wrote {sub}/")


def _copy_passthrough(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for sub in _PASSTHROUGH_SUBFOLDERS:
        s = src_dir / sub
        if not s.is_dir():
            continue
        d = dst_dir / sub
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("*.bin", "*.ckpt"))
        copied.append(sub)
        print(f"  copied {sub}/")
    return copied


def convert(src: str, dst: str) -> None:
    src_dir = _resolve_src(src)
    dst_dir = Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"src: {src_dir}")
    print(f"dst: {dst_dir}")

    monolithic = src_dir / "model.safetensors"
    if not monolithic.is_file():
        raise FileNotFoundError(f"Expected monolithic safetensors at {monolithic}")
    cfg_path = src_dir / "model_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Expected model_config.json at {cfg_path} (Stable Audio's authoritative "
            f"per-component arch config).")
    with open(cfg_path) as f:
        model_config = json.load(f)

    print("\n[1/4] Splitting monolithic state dict by component prefix:")
    buckets = _split_state_dict(monolithic)

    print("\n[2/4] Copying passthrough subfolders (text_encoder/tokenizer/scheduler/vae):")
    copied = _copy_passthrough(src_dir, dst_dir)

    print("\n[3/4] Writing per-component configs + safetensors:")
    component_cfgs = _component_config(model_config)
    for name in ("transformer", "vae", "conditioner"):
        if name in copied:
            print(f"  skipped {name}/ (already copied from source in Diffusers format)")
            continue
        _write_component(dst_dir, name, buckets[name], component_cfgs[name])

    print("\n[4/4] Writing model_index.json:")
    index = dict(_MODEL_INDEX)
    # Prefer the HF repo id over a snapshot directory (which is
    # machine-local and full of hash gunk).
    index["_fastvideo_converted_from"] = src if "/" in src and not os.path.isdir(src) else src_dir.name
    # Drop entries for subfolders we didn't actually populate.
    available = {"transformer", "vae", "conditioner", *copied}
    index = {k: v for k, v in index.items() if not k or k.startswith("_") or k in available}
    with open(dst_dir / "model_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"  wrote {dst_dir}/model_index.json")
    print(f"\nDone. Push to HF with:")
    print(f"  huggingface-cli upload <FastVideo/repo-name> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True,
                        help="HF repo id or local directory containing model.safetensors + model_config.json")
    parser.add_argument("--dst", required=True, help="Output directory for the Diffusers-format repo")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
