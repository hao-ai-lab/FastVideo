#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false
"""Convert FLUX.2 Klein weights into a FastVideo-loadable layout.

The published ``black-forest-labs/FLUX.2-klein-4B`` repo contains two useful
transformer surfaces:

* ``flux-2-klein-4b.safetensors``: BFL's compact raw transformer checkpoint.
  Its double-stream attention projections are fused as ``img_attn.qkv`` and
  ``txt_attn.qkv``.
* ``transformer/diffusion_pytorch_model.safetensors``: Diffusers/FastVideo
  names. These keys already match ``Flux2Transformer2DModel.state_dict()``.

By default this script prefers the raw root checkpoint when present, converts it
to the FastVideo native transformer key surface, reloads/resaves the VAE, copies
the HF-backed Qwen3 text encoder/tokenizer and scheduler, and emits a standard
Diffusers-style FastVideo repo:

    <dst>/
      model_index.json
      transformer/{config.json,diffusion_pytorch_model*.safetensors}
      vae/{config.json,diffusion_pytorch_model*.safetensors}
      text_encoder/...
      tokenizer/...
      scheduler/scheduler_config.json

Qwen3 and Mistral3 are intentionally copied as Transformers passthrough
components in the standard layout: ``qwen3.py`` and ``mistral3.py`` both route
Flux2 text encoding through ``from_pretrained_local()`` for exact HF parity.

Example:
    python scripts/checkpoint_conversion/convert_flux2_klein.py \
        --src black-forest-labs/FLUX.2-klein-4B \
        --dst converted_weights/flux2-klein-4b-fastvideo \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

try:
    from huggingface_hub import save_torch_state_dict, snapshot_download
except ImportError:
    save_torch_state_dict = None
    snapshot_download = None


DEFAULT_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
RAW_TRANSFORMER_FILENAME = "flux-2-klein-4b.safetensors"
DIFFUSION_WEIGHTS_BASENAME = "diffusion_pytorch_model"

BASE_SNAPSHOT_ALLOW_PATTERNS = (
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/*.safetensors",
    "vae/*.safetensors.index.json",
    "text_encoder/*",
    "tokenizer/*",
    "scheduler/*",
)

PASSTHROUGH_SUBFOLDERS = ("text_encoder", "tokenizer", "scheduler")

DEFAULT_MODEL_INDEX: dict[str, Any] = {
    "_class_name": "Flux2KleinPipeline",
    "_diffusers_version": "0.37.0.dev0",
    "is_distilled": True,
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3ForCausalLM"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "transformer": ["diffusers", "Flux2Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}

DEFAULT_SCHEDULER_CONFIG: dict[str, Any] = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

TRANSFORMER_REQUIRED_KEYS = (
    "x_embedder.weight",
    "context_embedder.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.linear.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)

VAE_REQUIRED_KEYS = (
    "encoder.conv_in.weight",
    "decoder.conv_out.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
    "bn.running_mean",
    "bn.running_var",
)


class ConversionError(RuntimeError):
    pass


def _resolve_hf_token() -> str | bool | None:
    try:
        from fastvideo.utils import resolve_hf_token

        token = resolve_hf_token()
        return token if token else None
    except Exception:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None


def _snapshot_allow_patterns(transformer_source: str) -> list[str]:
    patterns: list[str] = list(BASE_SNAPSHOT_ALLOW_PATTERNS)
    if transformer_source == "diffusers":
        patterns.extend(("transformer/*.safetensors", "transformer/*.safetensors.index.json"))
    else:
        patterns.append(RAW_TRANSFORMER_FILENAME)
    return patterns


def _resolve_src(src: str, revision: str | None, cache_dir: str | None, transformer_source: str) -> Path:
    local = Path(src).expanduser()
    if local.exists():
        return local

    if snapshot_download is None:
        raise ConversionError("huggingface_hub is required when --src is a repo id")

    return Path(
        snapshot_download(
            repo_id=src,
            revision=revision,
            cache_dir=cache_dir,
            token=_resolve_hf_token(),
            allow_patterns=_snapshot_allow_patterns(transformer_source),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _prepare_output_dir(dst: Path, overwrite: bool) -> None:
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def _find_safetensors_files(component_dir: Path, basename: str = DIFFUSION_WEIGHTS_BASENAME) -> list[Path]:
    if component_dir.is_file():
        return [component_dir]

    index_path = component_dir / f"{basename}.safetensors.index.json"
    if index_path.exists():
        index = _read_json(index_path)
        return sorted({component_dir / shard for shard in index["weight_map"].values()})

    single = component_dir / f"{basename}.safetensors"
    if single.exists():
        return [single]

    return sorted(component_dir.glob("*.safetensors"))


def _load_safetensors(files: list[Path]) -> OrderedDict[str, torch.Tensor]:
    if not files:
        raise FileNotFoundError("No safetensors files found")

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in state:
                    raise ConversionError(f"Duplicate tensor key {key!r} while reading {file}")
                state[key] = handle.get_tensor(key)
    return state


def _write_state_dict(
    state: OrderedDict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: str,
) -> None:
    if save_torch_state_dict is None:
        raise ConversionError("huggingface_hub.save_torch_state_dict is required to write sharded safetensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(
        state,
        output_dir,
        filename_pattern=f"{DIFFUSION_WEIGHTS_BASENAME}" + "{suffix}.safetensors",
        max_shard_size=max_shard_size,
        metadata={"format": "pt"},
        safe_serialization=True,
    )


def _split_qkv(weight: torch.Tensor, source_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[0] % 3 != 0:
        raise ConversionError(f"Expected first dim divisible by 3 for {source_key}, got {tuple(weight.shape)}")
    q_weight, k_weight, v_weight = torch.chunk(weight, 3, dim=0)
    return q_weight, k_weight, v_weight


def _convert_raw_transformer_key(
    key: str,
    value: torch.Tensor,
    output: OrderedDict[str, torch.Tensor],
) -> bool:
    literal_map = {
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
    }
    if key in literal_map:
        output[literal_map[key]] = value
        return True

    match = re.fullmatch(r"double_blocks\.(\d+)\.(img|txt)_attn\.qkv\.weight", key)
    if match:
        block, stream = match.groups()
        q_weight, k_weight, v_weight = _split_qkv(value, key)
        if stream == "img":
            prefix = f"transformer_blocks.{block}.attn"
            output[f"{prefix}.to_q.weight"] = q_weight
            output[f"{prefix}.to_k.weight"] = k_weight
            output[f"{prefix}.to_v.weight"] = v_weight
        else:
            prefix = f"transformer_blocks.{block}.attn"
            output[f"{prefix}.add_q_proj.weight"] = q_weight
            output[f"{prefix}.add_k_proj.weight"] = k_weight
            output[f"{prefix}.add_v_proj.weight"] = v_weight
        return True

    double_rewrites = (
        (r"double_blocks\.(\d+)\.img_attn\.proj\.weight", r"transformer_blocks.\1.attn.to_out.0.weight"),
        (r"double_blocks\.(\d+)\.txt_attn\.proj\.weight", r"transformer_blocks.\1.attn.to_add_out.weight"),
        (r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale", r"transformer_blocks.\1.attn.norm_q.weight"),
        (r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale", r"transformer_blocks.\1.attn.norm_k.weight"),
        (r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale", r"transformer_blocks.\1.attn.norm_added_q.weight"),
        (r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale", r"transformer_blocks.\1.attn.norm_added_k.weight"),
        (r"double_blocks\.(\d+)\.img_mlp\.0\.weight", r"transformer_blocks.\1.ff.linear_in.weight"),
        (r"double_blocks\.(\d+)\.img_mlp\.2\.weight", r"transformer_blocks.\1.ff.linear_out.weight"),
        (r"double_blocks\.(\d+)\.txt_mlp\.0\.weight", r"transformer_blocks.\1.ff_context.linear_in.weight"),
        (r"double_blocks\.(\d+)\.txt_mlp\.2\.weight", r"transformer_blocks.\1.ff_context.linear_out.weight"),
        (r"single_blocks\.(\d+)\.linear1\.weight", r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.weight"),
        (r"single_blocks\.(\d+)\.linear2\.weight", r"single_transformer_blocks.\1.attn.to_out.weight"),
        (r"single_blocks\.(\d+)\.norm\.query_norm\.scale", r"single_transformer_blocks.\1.attn.norm_q.weight"),
        (r"single_blocks\.(\d+)\.norm\.key_norm\.scale", r"single_transformer_blocks.\1.attn.norm_k.weight"),
    )
    for pattern, replacement in double_rewrites:
        if re.fullmatch(pattern, key):
            output[re.sub(pattern, replacement, key)] = value
            return True

    return False


def convert_raw_transformer(raw_state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    unexpected: list[str] = []
    for key, value in raw_state.items():
        if not _convert_raw_transformer_key(key, value, converted):
            unexpected.append(key)

    if unexpected:
        raise ConversionError(
            "Unmapped raw Flux2 Klein transformer keys:\n" + "\n".join(f"  - {key}" for key in unexpected)
        )
    return converted


def convert_diffusers_transformer(state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        if key.startswith("transformer."):
            key = key[len("transformer."):]
        converted[key] = value
    return converted


def convert_vae(state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(state.items())


def _infer_block_count(keys: set[str], prefix: str) -> int:
    count = 0
    for key in keys:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            first = suffix.split(".", 1)[0]
            if first.isdigit():
                count = max(count, int(first) + 1)
    return count


def _validate_required_keys(component: str, state: OrderedDict[str, torch.Tensor], required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in state]
    if missing:
        raise ConversionError(f"{component} conversion missing required keys: {missing}")


def _validate_transformer(state: OrderedDict[str, torch.Tensor]) -> None:
    _validate_required_keys("transformer", state, TRANSFORMER_REQUIRED_KEYS)
    raw_leftovers = [key for key in state if key.startswith(("double_blocks.", "single_blocks."))]
    if raw_leftovers:
        raise ConversionError(f"Raw transformer keys leaked into output: {raw_leftovers[:8]}")
    keys = set(state)
    double_blocks = _infer_block_count(keys, "transformer_blocks.")
    single_blocks = _infer_block_count(keys, "single_transformer_blocks.")
    print(
        "  transformer keys: " +
        f"{len(state)} total, {double_blocks} double blocks, {single_blocks} single blocks"
    )


def _validate_vae(state: OrderedDict[str, torch.Tensor]) -> None:
    _validate_required_keys("vae", state, VAE_REQUIRED_KEYS)
    print(f"  vae keys: {len(state)} total")


def _component_config(src_dir: Path, component: str, class_name: str) -> dict[str, Any]:
    config_path = src_dir / component / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {component} config: {config_path}")
    config = _read_json(config_path)
    config["_class_name"] = class_name
    config.pop("_name_or_path", None)
    return config


def _copy_passthrough_subfolder(src_dir: Path, dst_dir: Path, subfolder: str) -> None:
    src = src_dir / subfolder
    if not src.is_dir():
        raise FileNotFoundError(f"Missing passthrough subfolder: {src}")
    dst = dst_dir / subfolder
    shutil.copytree(src, dst)
    print(f"  copied {subfolder}/")


def _copy_or_write_scheduler(src_dir: Path, dst_dir: Path) -> None:
    src = src_dir / "scheduler"
    dst = dst_dir / "scheduler"
    if src.is_dir():
        shutil.copytree(src, dst)
        print("  copied scheduler/")
        return
    _write_json(dst / "scheduler_config.json", DEFAULT_SCHEDULER_CONFIG)
    print("  wrote default scheduler/scheduler_config.json")


def _build_model_index(src_dir: Path, source_label: str) -> dict[str, Any]:
    index_path = src_dir / "model_index.json"
    if index_path.exists():
        index = _read_json(index_path)
    else:
        index = dict(DEFAULT_MODEL_INDEX)
    index.update(
        {
            "_class_name": "Flux2KleinPipeline",
            "is_distilled": True,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
            "_fastvideo_converted_from": source_label,
        }
    )
    return index


def _select_transformer_source(src_dir: Path, mode: str) -> tuple[str, Path]:
    if src_dir.is_file():
        if mode == "diffusers":
            raise FileNotFoundError(f"Cannot use --transformer-source diffusers with a file source: {src_dir}")
        return "raw", src_dir

    raw_path = src_dir / RAW_TRANSFORMER_FILENAME
    diffusers_dir = src_dir / "transformer"

    if mode == "raw":
        if not raw_path.exists():
            raise FileNotFoundError(f"Requested raw transformer but missing {raw_path}")
        return "raw", raw_path
    if mode == "diffusers":
        if not diffusers_dir.is_dir():
            raise FileNotFoundError(f"Requested diffusers transformer but missing {diffusers_dir}")
        return "diffusers", diffusers_dir
    if raw_path.exists():
        return "raw", raw_path
    if diffusers_dir.is_dir():
        return "diffusers", diffusers_dir
    raise FileNotFoundError(f"Missing transformer weights under {src_dir}")


def convert(src: str, dst: str, revision: str | None, cache_dir: str | None, max_shard_size: str,
            transformer_source: str, overwrite: bool) -> None:
    resolved_src = _resolve_src(src, revision=revision, cache_dir=cache_dir, transformer_source=transformer_source)
    src_dir = resolved_src.parent if resolved_src.is_file() else resolved_src
    dst_dir = Path(dst).expanduser()
    _prepare_output_dir(dst_dir, overwrite=overwrite)

    print(f"src: {src_dir}")
    print(f"dst: {dst_dir}")

    print("\n[1/5] Converting transformer:")
    source_kind, source_path = _select_transformer_source(resolved_src, transformer_source)
    if source_kind == "raw":
        print(f"  using raw BFL transformer: {source_path.name}")
        transformer_state = convert_raw_transformer(_load_safetensors([source_path]))
    else:
        print("  using Diffusers transformer subfolder")
        transformer_state = convert_diffusers_transformer(_load_safetensors(_find_safetensors_files(source_path)))
    _validate_transformer(transformer_state)
    transformer_dir = dst_dir / "transformer"
    _write_state_dict(transformer_state, transformer_dir, max_shard_size=max_shard_size)
    _write_json(transformer_dir / "config.json", _component_config(src_dir, "transformer", "Flux2Transformer2DModel"))

    print("\n[2/5] Converting VAE:")
    vae_dir = src_dir / "vae"
    vae_state = convert_vae(_load_safetensors(_find_safetensors_files(vae_dir)))
    _validate_vae(vae_state)
    output_vae_dir = dst_dir / "vae"
    _write_state_dict(vae_state, output_vae_dir, max_shard_size=max_shard_size)
    _write_json(output_vae_dir / "config.json", _component_config(src_dir, "vae", "AutoencoderKLFlux2"))

    print("\n[3/5] Copying HF-backed encoders/tokenizer/scheduler:")
    # Current FastVideo Flux2 text encoders intentionally use HF passthrough:
    # Qwen3ForCausalLM.from_pretrained_local() and Mistral3ForConditionalGeneration.from_pretrained_local().
    # Rewriting their weights to native fused QKV names would make the standard loader call the wrong HF class.
    _copy_passthrough_subfolder(src_dir, dst_dir, "text_encoder")
    _copy_passthrough_subfolder(src_dir, dst_dir, "tokenizer")
    _copy_or_write_scheduler(src_dir, dst_dir)

    print("\n[4/5] Writing model_index.json:")
    _write_json(dst_dir / "model_index.json", _build_model_index(src_dir, src))
    print("  wrote model_index.json")

    print("\n[5/5] Summary:")
    if source_kind == "raw":
        print("  transformer mapping: raw BFL -> FastVideo native")
    else:
        print("  transformer mapping: Diffusers/FastVideo identity")
    print("  vae mapping: Diffusers/FastVideo identity")
    print("  text encoder mapping: HF Qwen3 passthrough (FastVideo loader uses from_pretrained_local)")
    print(f"  max shard size: {max_shard_size}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("--src", default=DEFAULT_REPO_ID, help="HF repo id or local Flux2 Klein snapshot directory")
    parser.add_argument("--dst", required=True, help="Output directory for the FastVideo Diffusers-style repo")
    parser.add_argument("--revision", default=None, help="Optional HF revision for --src repo id")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output shard size")
    parser.add_argument(
        "--transformer-source",
        choices=("auto", "raw", "diffusers"),
        default="auto",
        help="auto prefers flux-2-klein-4b.safetensors when present, else transformer/",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    args = parser.parse_args()

    convert(
        src=args.src,
        dst=args.dst,
        revision=args.revision,
        cache_dir=args.cache_dir,
        max_shard_size=args.max_shard_size,
        transformer_source=args.transformer_source,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
