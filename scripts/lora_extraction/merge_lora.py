"""Merge LoRA adapter into base model weights.

Usage:
    python merge_lora_updated.py \
        --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --adapter adapter.safetensors \
        --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
        --output ./merged_model
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import logging
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable
from typing import Any

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

_FASTVIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastvideo_pr", "FastVideo"))
if _FASTVIDEO_PATH not in sys.path:
    sys.path.insert(0, _FASTVIDEO_PATH)

import torch
from safetensors.torch import load_file, save_file

from extract_lora import load_transformer_state_dict_from_model, get_pipeline_class_for_model
from fastvideo.training.training_utils import custom_to_hf_state_dict
from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

LOG = logging.getLogger(__name__)


def configure_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)


def fix_adapter_naming(adapter: dict) -> dict:
    fixed = {}
    renamed = 0

    for key, tensor in adapter.items():
        new_key = key

        if ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")
            if not base.endswith(".weight"):
                new_key = base + ".weight.lora_A.weight"
                renamed += 1
        elif ".lora_B.weight" in key:
            base = key.replace(".lora_B.weight", "")
            if not base.endswith(".weight"):
                new_key = base + ".weight.lora_B.weight"
                renamed += 1
        elif ".lora_rank" in key:
            base = key.replace(".lora_rank", "")
            if not base.endswith(".weight"):
                new_key = base + ".weight.lora_rank"
                renamed += 1
        elif ".lora_alpha" in key:
            base = key.replace(".lora_alpha", "")
            if not base.endswith(".weight"):
                new_key = base + ".weight.lora_alpha"
                renamed += 1

        fixed[new_key] = tensor

    if renamed > 0:
        LOG.info(f"Fixed {renamed} adapter key names")

    return fixed


def load_adapter(adapter_path: str) -> dict:
    abs_path = os.path.abspath(adapter_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Adapter file not found: {abs_path}")
    if not abs_path.endswith('.safetensors'):
        raise ValueError(f"Adapter must be .safetensors: {abs_path}")

    LOG.info(f"Loading adapter: {abs_path}")
    adapter = load_file(abs_path)
    file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    LOG.info(f"Loaded {len(adapter)} tensors ({file_size_mb:.1f} MB)")

    return fix_adapter_naming(adapter)


# Suffix -> the parameter suffix it targets, mirroring fastvideo.models.loader.lora_patch.
# An empty target keeps the full name, for standalone nn.Parameters.
ADDITIVE_SUFFIXES: dict[str, str] = {".diff_param": "", ".diff_b": ".bias", ".diff": ".weight"}
REPLACEMENT_SUFFIXES: dict[str, str] = {".set_weight": ".weight", ".set_param": ""}
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_rank", ".lora_alpha")


def group_dense_keys(adapter: dict) -> tuple[dict, dict, list]:
    """Split the non-factorized half of an adapter into additive and replacement params.

    ``--exact-tensor-pattern`` keeps selected matrices as exact dense deltas instead of
    LoRA factors, so an adapter merged without these is missing those tensors entirely.
    """
    additive: dict = {}
    replacement: dict = {}
    unrecognized: list = []

    for key, tensor in adapter.items():
        if key.endswith(LORA_SUFFIXES):
            continue
        for suffix, param_suffix in ADDITIVE_SUFFIXES.items():
            if key.endswith(suffix):
                additive[key.removesuffix(suffix) + param_suffix] = tensor
                break
        else:
            for suffix, param_suffix in REPLACEMENT_SUFFIXES.items():
                if key.endswith(suffix):
                    replacement[key.removesuffix(suffix) + param_suffix] = tensor
                    break
            else:
                unrecognized.append(key)

    return additive, replacement, unrecognized


ParamMapping = Callable[[str], tuple[str, Any, Any]]


def map_adapter_parameter(
    name: str,
    lora_param_names_mapping: ParamMapping | None,
    param_names_mapping: ParamMapping | None,
) -> tuple[str, int | None, int | None]:
    """Apply the same official-LoRA -> HF -> FastVideo mapping order as runtime loading."""
    if lora_param_names_mapping is not None:
        name, merge_index, total = lora_param_names_mapping(name)
        if merge_index is not None:
            raise NotImplementedError(f"Adapter-specific mapping unexpectedly fused {name} ({merge_index}/{total})")
    if param_names_mapping is None:
        return name, None, None
    return param_names_mapping(name)


def _mapped_slice(target: torch.Tensor, merge_index: int | None,
                  total: int | None) -> tuple[Any, tuple[int, ...]] | None:
    if merge_index is None:
        return Ellipsis, tuple(target.shape)
    if total is None or total < 1 or target.ndim < 1 or target.shape[0] % total:
        return None
    chunk = target.shape[0] // total
    return slice(merge_index * chunk, (merge_index + 1) * chunk), (chunk, *target.shape[1:])


def merge_dense_into_base(
    merged_sd: dict,
    additive: dict,
    replacement: dict,
    lora_param_names_mapping: ParamMapping | None = None,
    param_names_mapping: ParamMapping | None = None,
) -> tuple[int, list[str]]:
    """Apply exact dense deltas and replacement parameters in place."""
    merged_count = 0
    skipped: list[str] = []

    for source_name, tensor in additive.items():
        param_name, merge_index, total = map_adapter_parameter(source_name, lora_param_names_mapping,
                                                               param_names_mapping)
        target = merged_sd.get(param_name)
        target_slice = _mapped_slice(target, merge_index, total) if target is not None else None
        if target is None or target_slice is None or target_slice[1] != tuple(tensor.shape):
            skipped.append(f"{source_name} -> {param_name}")
            continue
        index, _ = target_slice
        updated = target.to(torch.float32).clone()
        updated[index] += tensor.to(torch.float32)
        merged_sd[param_name] = updated.to(target.dtype)
        merged_count += 1

    for source_name, tensor in replacement.items():
        param_name, merge_index, total = map_adapter_parameter(source_name, lora_param_names_mapping,
                                                               param_names_mapping)
        target = merged_sd.get(param_name)
        if target is None:
            if merge_index is not None:
                skipped.append(f"{source_name} -> {param_name}")
                continue
            merged_sd[param_name] = tensor
            merged_count += 1
            continue
        target_slice = _mapped_slice(target, merge_index, total)
        if target_slice is None or target_slice[1] != tuple(tensor.shape):
            skipped.append(f"{source_name} -> {param_name}")
            continue
        index, _ = target_slice
        updated = target.clone()
        updated[index] = tensor.to(target.dtype)
        merged_sd[param_name] = updated
        merged_count += 1

    return merged_count, skipped


def group_adapter_keys(adapter: dict) -> dict:
    grouped = defaultdict(dict)

    for key, tensor in adapter.items():
        if key.endswith(".lora_A.weight"):
            grouped[key.replace(".lora_A.weight", "")]["A"] = tensor
        elif key.endswith(".lora_B.weight"):
            grouped[key.replace(".lora_B.weight", "")]["B"] = tensor
        elif key.endswith(".lora_rank"):
            grouped[key.replace(".lora_rank", "")]["rank"] = tensor
        elif key.endswith(".lora_alpha"):
            grouped[key.replace(".lora_alpha", "")]["alpha"] = tensor

    LOG.info(f"Grouped {len(grouped)} LoRA layers")
    return grouped


def get_reverse_param_mapping(base_model_path: str) -> tuple[dict, ParamMapping, ParamMapping]:
    LOG.info("Loading base model for parameter mapping")

    pipeline_cls = get_pipeline_class_for_model(base_model_path)
    pipeline = pipeline_cls.from_pretrained(
        base_model_path,
        num_gpus=1,
        inference_mode=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    transformer = None
    if hasattr(pipeline, "transformer"):
        transformer = pipeline.transformer
    elif hasattr(pipeline, "modules") and isinstance(pipeline.modules, dict):
        if "transformer" in pipeline.modules:
            transformer = pipeline.modules["transformer"]

    if transformer is None:
        raise RuntimeError("Could not find transformer in pipeline")

    param_names_mapping_fn = get_param_names_mapping(transformer.param_names_mapping)
    lora_param_names_mapping_fn = get_param_names_mapping(transformer.lora_param_names_mapping)

    if getattr(transformer, "reverse_param_names_mapping", None):
        reverse_mapping = transformer.reverse_param_names_mapping
    elif hasattr(transformer, "config") and hasattr(transformer.config, "arch_config"):
        arch_config = transformer.config.arch_config
        if getattr(arch_config, "reverse_param_names_mapping", None):
            reverse_mapping = arch_config.reverse_param_names_mapping
        else:
            from diffusers import DiffusionPipeline
            from huggingface_hub import snapshot_download

            if not os.path.exists(base_model_path) or not os.path.isdir(base_model_path):
                model_path = snapshot_download(repo_id=base_model_path, ignore_patterns=["*.onnx", "*.msgpack"])
            else:
                model_path = base_model_path

            hf_pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
            hf_transformer = hf_pipeline.transformer
            hf_sd = hf_transformer.state_dict()

            _, reverse_mapping = hf_to_custom_state_dict(hf_sd, param_names_mapping_fn)

            del hf_pipeline
            del hf_transformer
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("Could not find reverse_param_names_mapping in transformer or config")

    del pipeline
    del transformer
    torch.cuda.empty_cache()

    return reverse_mapping, param_names_mapping_fn, lora_param_names_mapping_fn


def merge_lora_into_base(
    base_sd: dict,
    adapter: dict,
    *,
    lora_param_names_mapping: ParamMapping | None = None,
    param_names_mapping: ParamMapping | None = None,
    strict: bool = True,
) -> dict:
    LOG.info("Merging LoRA into base weights")

    adapter_layers = group_adapter_keys(adapter)
    merged_sd = dict(base_sd)

    merged_count = 0
    skipped: list[str] = []

    for base_name, parts in adapter_layers.items():
        source_weight_key = base_name if base_name.endswith(".weight") else base_name + ".weight"
        weight_key, merge_index, total = map_adapter_parameter(source_weight_key, lora_param_names_mapping,
                                                               param_names_mapping)
        base_tensor = merged_sd.get(weight_key)
        if base_tensor is None:
            skipped.append(f"{source_weight_key} -> {weight_key} (missing base parameter)")
            continue

        if "A" not in parts or "B" not in parts:
            skipped.append(f"{source_weight_key} (incomplete factor pair)")
            continue

        lora_A = parts["A"].to(torch.float32)
        lora_B = parts["B"].to(torch.float32)
        target_slice = _mapped_slice(base_tensor, merge_index, total)
        if target_slice is None:
            skipped.append(f"{source_weight_key} -> {weight_key} (invalid fused mapping)")
            continue
        index, expected_shape = target_slice
        out_dim, in_dim = expected_shape
        if lora_B.shape[0] != out_dim or lora_A.shape[1] != in_dim or lora_B.shape[1] != lora_A.shape[0]:
            skipped.append(f"{source_weight_key} -> {weight_key} (factor shape mismatch)")
            continue

        delta = lora_B @ lora_A

        rank = int(parts.get("rank", torch.tensor([lora_A.shape[0]])).item()) if "rank" in parts else lora_A.shape[0]
        alpha = float(parts.get("alpha", torch.tensor([rank])).item()) if "alpha" in parts else float(rank)

        if rank != 0 and alpha != rank:
            delta = delta * (alpha / float(rank))

        updated = base_tensor.to(torch.float32).clone()
        updated[index] += delta
        merged_sd[weight_key] = updated.to(base_tensor.dtype)
        merged_count += 1

    additive, replacement, unrecognized = group_dense_keys(adapter)
    dense_merged, dense_skipped = merge_dense_into_base(
        merged_sd,
        additive,
        replacement,
        lora_param_names_mapping,
        param_names_mapping,
    )

    LOG.info("Merged %d LoRA layers, skipped %d", merged_count, len(skipped))
    LOG.info("Merged %d dense tensors (%d additive, %d replacement)", dense_merged, len(additive), len(replacement))
    problems = skipped + dense_skipped + [f"unrecognized adapter key {key}" for key in unrecognized]
    if problems and strict:
        raise ValueError(f"Adapter merge left {len(problems)} keys/layers unapplied: {problems[:5]}")
    if problems:
        LOG.warning("Adapter merge left %d keys/layers unapplied: %s", len(problems), problems[:5])

    return merged_sd


def save_merged_model(merged_sd: dict, base_model_path: str, ft_model_path: str, output_dir: str,
                      reverse_mapping: dict):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transformer_dir = output_path / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Converting to HuggingFace format")
    hf_merged_sd = custom_to_hf_state_dict(merged_sd, reverse_mapping)
    LOG.info(f"Converted {len(hf_merged_sd)} parameters")

    base_path = Path(base_model_path)
    if not base_path.exists() or not base_path.is_dir():
        from huggingface_hub import snapshot_download
        base_path = Path(snapshot_download(repo_id=base_model_path, ignore_patterns=["*.onnx", "*.msgpack"]))

    LOG.info("Copying model components")
    for component in ["scheduler", "text_encoder", "tokenizer", "vae"]:
        src = base_path / component
        if src.exists():
            dst = output_path / component
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    LOG.info("Copying finetuned model config")
    ft_path = Path(ft_model_path)
    if not ft_path.exists() or not ft_path.is_dir():
        from huggingface_hub import snapshot_download
        ft_path = Path(
            snapshot_download(repo_id=ft_model_path,
                              allow_patterns=["model_index.json"],
                              ignore_patterns=["*.onnx", "*.msgpack"]))

    ft_index = ft_path / "model_index.json"
    if ft_index.exists():
        shutil.copy2(ft_index, output_path / "model_index.json")
    else:
        LOG.warning("Finetuned model_index.json not found, using base")
        src_index = base_path / "model_index.json"
        if src_index.exists():
            shutil.copy2(src_index, output_path / "model_index.json")

    weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    LOG.info(f"Saving merged weights to {weight_path}")
    to_save_hf = {k: v.detach().cpu() for k, v in hf_merged_sd.items()}
    save_file(to_save_hf, str(weight_path))

    config_src = base_path / "transformer" / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, transformer_dir / "config.json")

    file_size_mb = weight_path.stat().st_size / (1024 * 1024)
    LOG.info(f"Saved to {output_dir} ({file_size_mb:.0f} MB, {len(hf_merged_sd)} params)")


def merge_lora(
    base: str,
    adapter: str,
    ft: str,
    output: str,
    log_level: str = "INFO",
    allow_unmatched: bool = False,
) -> None:
    """Merge LoRA adapter into base model.
    
    Args:
        base: Base model ID or path
        adapter: LoRA adapter .safetensors file
        ft: Finetuned model ID (for config)
        output: Output directory
        log_level: Logging level
        allow_unmatched: Write output even if recognized adapter payloads cannot be applied
    """
    configure_logging(log_level)

    LOG.info(f"Base: {base}")
    LOG.info(f"Adapter: {adapter}")
    LOG.info(f"Output: {output}")

    reverse_mapping, param_names_mapping_fn, lora_param_names_mapping_fn = get_reverse_param_mapping(base)

    LOG.info(f"Loading base model: {base}")
    base_sd = load_transformer_state_dict_from_model(base)
    LOG.info(f"Loaded { len(base_sd)} parameters")

    adapter_sd = load_adapter(adapter)
    merged_sd = merge_lora_into_base(
        base_sd,
        adapter_sd,
        lora_param_names_mapping=lora_param_names_mapping_fn,
        param_names_mapping=param_names_mapping_fn,
        strict=not allow_unmatched,
    )

    save_merged_model(merged_sd, base, ft, output, reverse_mapping)
    LOG.info("Merge complete")


def main():
    """CLI wrapper for merge_lora."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model ID or path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter .safetensors file")
    parser.add_argument("--ft", required=True, help="Finetuned model ID (for config)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--allow-unmatched", action="store_true",
                        help="Write the merged model even when adapter keys cannot be applied")
    args = parser.parse_args()

    merge_lora(
        base=args.base,
        adapter=args.adapter,
        ft=args.ft,
        output=args.output,
        log_level=args.log_level,
        allow_unmatched=args.allow_unmatched,
    )


if __name__ == "__main__":
    main()
