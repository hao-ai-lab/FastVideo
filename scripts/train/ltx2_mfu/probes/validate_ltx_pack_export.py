#!/usr/bin/env python3
"""GB200 gate for packed LTX projection export and split reload."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open


MODEL_CLASS = "LTX2VideoOnlyTransformer3DModel"
EXPECTED_MERGED_TENSORS = 48 * 4
EXPECTED_SPLIT_TENSORS = 48 * 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/train/configs/overfit_ltx2_t2v.yaml",
    )
    parser.add_argument("--model-path")
    parser.add_argument(
        "--work-dir",
        default="/mnt/pr1630-real-pack-export-gate",
    )
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def _guarded_reset(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to reset symlinked work directory: {path}")
    path = path.resolve()
    if len(path.parts) < 3:
        raise ValueError(f"Refusing to reset unsafe work directory: {path}")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)


def _source_root(model_path: str) -> Path:
    local = Path(os.path.expanduser(model_path))
    if local.exists():
        return local.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_path,
            allow_patterns=["model_index.json", "transformer/*"],
        )
    ).resolve()


def _write_minimal_root(
    destination: Path,
    source: Path,
    *,
    include_weights: bool,
) -> None:
    source_index = json.loads(
        (source / "model_index.json").read_text(encoding="utf-8")
    )
    required = ("_class_name", "_diffusers_version", "transformer")
    missing = [key for key in required if key not in source_index]
    if missing:
        raise ValueError(f"Source model index is missing keys: {missing}")

    transformer = destination / "transformer"
    transformer.mkdir(parents=True)
    minimal_index = {key: source_index[key] for key in required}
    (destination / "model_index.json").write_text(
        json.dumps(minimal_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(source / "transformer" / "config.json", transformer / "config.json")

    if include_weights:
        weights = sorted((source / "transformer").glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"No transformer safetensors under {source}")
        for weight in weights:
            os.symlink(weight.resolve(), transformer / weight.name)


def _training_config(config_path: Path, *, packed: bool) -> Any:
    from fastvideo.train.utils.config import load_run_config

    tc = load_run_config(str(config_path)).training
    tc.distributed.num_gpus = 1
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1
    tc.model.enable_torch_compile = False
    tc.dit_precision = "fp32"
    if tc.pipeline_config is None:
        raise ValueError("LTX export gate requires a resolved pipeline config")
    tc.pipeline_config.dit_config.arch_config.pack_attention_projections = packed
    return tc


def _entries(raw: Any) -> list[tuple[Any, ...]]:
    entries = raw if isinstance(raw, list) else [raw]
    if not entries or not all(isinstance(entry, tuple) for entry in entries):
        raise ValueError(f"Invalid reverse mapping entry: {raw!r}")
    return entries


def _mapping_sources(reverse_mapping: dict[str, Any]) -> set[str]:
    return {
        str(entry[0])
        for raw in reverse_mapping.values()
        for entry in _entries(raw)
    }


def _merged_mapping(reverse_mapping: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    merged: dict[str, list[tuple[Any, ...]]] = {}
    for internal_name, raw in reverse_mapping.items():
        entries = _entries(raw)
        if any(entry[1] is not None for entry in entries):
            if any(entry[1] is None for entry in entries):
                raise ValueError(f"Mixed direct and merged mapping for {internal_name}")
            merged[internal_name] = entries
    return merged


def _safetensor_key_files(paths: list[Path]) -> dict[str, Path]:
    key_files: dict[str, Path] = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in key_files:
                    raise ValueError(f"Duplicate source safetensors key: {key}")
                key_files[key] = path
    return key_files


def _assert_packed_names(names: set[str]) -> None:
    if not any(re.search(r"\.attn1\.to_qkv\.(?:weight|bias)$", name) for name in names):
        raise AssertionError("Packed model has no self-attention QKV parameters")
    if not any(re.search(r"\.attn2\.to_kv\.(?:weight|bias)$", name) for name in names):
        raise AssertionError("Packed model has no cross-attention KV parameters")
    leaked = [
        name
        for name in names
        if re.search(r"\.attn1\.to_[qkv]\.(?:weight|bias)$", name)
        or re.search(r"\.attn2\.to_[kv]\.(?:weight|bias)$", name)
    ]
    if leaked:
        raise AssertionError(f"Packed model retained split parameters: {leaked[:4]}")


def _assert_split_names(names: set[str]) -> None:
    leaked = [
        name
        for name in names
        if re.search(r"\.attn1\.to_qkv\.(?:weight|bias)$", name)
        or re.search(r"\.attn2\.to_kv\.(?:weight|bias)$", name)
    ]
    if leaked:
        raise AssertionError(f"Split model retained packed parameters: {leaked[:4]}")
    for pattern in (
        r"\.attn1\.to_q\.weight$",
        r"\.attn1\.to_k\.weight$",
        r"\.attn1\.to_v\.weight$",
        r"\.attn2\.to_k\.weight$",
        r"\.attn2\.to_v\.weight$",
    ):
        if not any(re.search(pattern, name) for name in names):
            raise AssertionError(f"Split model is missing parameter pattern {pattern}")


def _full_cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = full_tensor()
    else:
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            tensor = to_local()
    return tensor.cpu()


def _load_transformer(model_path: Path, training_config: Any) -> torch.nn.Module:
    from fastvideo.train.utils.moduleloader import load_module_from_path

    return load_module_from_path(
        model_path=str(model_path),
        module_type="transformer",
        training_config=training_config,
        override_transformer_cls_name=MODEL_CLASS,
    )


def main() -> None:
    args = _parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This gate must run as one process with WORLD_SIZE=1")
    os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)

    config_path = Path(args.config).resolve()
    work_dir = Path(args.work_dir).resolve()
    _guarded_reset(work_dir)
    load_base = work_dir / "load_base"
    export_base = work_dir / "export_base"
    export_dir = work_dir / "export"

    packed_model: torch.nn.Module | None = None
    split_model: torch.nn.Module | None = None
    packed_parameters: dict[str, torch.nn.Parameter] | None = None
    split_parameters: dict[str, torch.nn.Parameter] | None = None
    result: dict[str, Any] | None = None
    success = False

    try:
        packed_tc = _training_config(config_path, packed=True)
        source_model_path = str(args.model_path or packed_tc.model_path)
        source = _source_root(source_model_path)
        _write_minimal_root(load_base, source, include_weights=True)
        _write_minimal_root(export_base, source, include_weights=False)
        if list((export_base / "transformer").glob("*.safetensors")):
            raise AssertionError("Export base unexpectedly contains model weights")

        from fastvideo.distributed import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        from fastvideo.train.entrypoint.dcp_to_diffusers import (
            _ensure_distributed,
            _save_role_pretrained,
        )

        _ensure_distributed()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

        packed_model = _load_transformer(load_base, packed_tc)
        packed_parameters = dict(packed_model.named_parameters())
        _assert_packed_names(set(packed_parameters))
        if {parameter.dtype for parameter in packed_parameters.values()} != {torch.float32}:
            raise AssertionError("Packed model does not have FP32 registered parameters")

        reverse_mapping = copy.deepcopy(
            getattr(packed_model, "reverse_param_names_mapping", {})
        )
        merged_mapping = _merged_mapping(reverse_mapping)
        merged_sources = {
            str(entry[0])
            for entries in merged_mapping.values()
            for entry in entries
        }
        if len(merged_mapping) != EXPECTED_MERGED_TENSORS:
            raise AssertionError(
                f"Expected {EXPECTED_MERGED_TENSORS} packed tensors, "
                f"got {len(merged_mapping)}"
            )
        if len(merged_sources) != EXPECTED_SPLIT_TENSORS:
            raise AssertionError(
                f"Expected {EXPECTED_SPLIT_TENSORS} split projection tensors, "
                f"got {len(merged_sources)}"
            )

        packed_parameter_objects = len(packed_parameters)
        packed_numel = sum(parameter.numel() for parameter in packed_parameters.values())
        del packed_parameters

        _save_role_pretrained(
            role="student",
            base_model_path=str(export_base),
            output_dir=str(export_dir),
            overwrite=True,
            model=SimpleNamespace(transformer=packed_model),
        )

        del packed_model
        packed_model = None
        gc.collect()
        torch.cuda.empty_cache()

        exported_files = sorted((export_dir / "transformer").glob("*.safetensors"))
        if [path.name for path in exported_files] != ["model.safetensors"]:
            raise AssertionError(f"Unexpected exported safetensors: {exported_files}")
        exported_file = exported_files[0]

        expected_export_keys = _mapping_sources(reverse_mapping)
        with safe_open(str(exported_file), framework="pt", device="cpu") as exported:
            exported_keys = set(exported.keys())
        if exported_keys != expected_export_keys:
            missing = sorted(expected_export_keys - exported_keys)
            unexpected = sorted(exported_keys - expected_export_keys)
            raise AssertionError(
                f"Export key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        internal_leaks = sorted(
            key
            for key in exported_keys
            if ".to_qkv." in key or ".to_kv." in key
        )
        if internal_leaks:
            raise AssertionError(f"Packed names leaked into export: {internal_leaks[:8]}")

        source_weight_files = sorted((source / "transformer").glob("*.safetensors"))
        source_key_files = _safetensor_key_files(source_weight_files)
        missing_source = sorted(merged_sources - source_key_files.keys())
        if missing_source:
            raise AssertionError(f"Projection keys missing from source: {missing_source[:8]}")

        keys_by_source_file: defaultdict[Path, list[str]] = defaultdict(list)
        for key in sorted(merged_sources):
            keys_by_source_file[source_key_files[key]].append(key)
        with safe_open(str(exported_file), framework="pt", device="cpu") as exported:
            for source_file, keys in keys_by_source_file.items():
                with safe_open(str(source_file), framework="pt", device="cpu") as source_handle:
                    for key in keys:
                        actual = exported.get_tensor(key)
                        expected = source_handle.get_tensor(key).to(actual.dtype)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f"Exported projection differs from source: {key}")

        split_tc = _training_config(config_path, packed=False)
        split_model = _load_transformer(export_dir, split_tc)
        split_parameters = dict(split_model.named_parameters())
        _assert_split_names(set(split_parameters))
        if {parameter.dtype for parameter in split_parameters.values()} != {torch.float32}:
            raise AssertionError("Split model does not have FP32 registered parameters")

        split_reverse = getattr(split_model, "reverse_param_names_mapping", {})
        source_to_internal: dict[str, str] = {}
        for internal_name, raw in split_reverse.items():
            entries = _entries(raw)
            if len(entries) != 1 or entries[0][1] is not None:
                raise AssertionError(f"Split reload retained a merged mapping for {internal_name}")
            source_name = str(entries[0][0])
            if source_name in source_to_internal:
                raise AssertionError(f"Duplicate split source mapping: {source_name}")
            source_to_internal[source_name] = internal_name

        missing_reloaded = sorted(merged_sources - source_to_internal.keys())
        if missing_reloaded:
            raise AssertionError(f"Projection keys missing after split reload: {missing_reloaded[:8]}")

        with safe_open(str(exported_file), framework="pt", device="cpu") as exported:
            for source_name in sorted(merged_sources):
                internal_name = source_to_internal[source_name]
                actual = _full_cpu(split_parameters[internal_name])
                expected = exported.get_tensor(source_name).to(actual.dtype)
                if not torch.equal(actual, expected):
                    raise AssertionError(f"Reloaded projection differs from export: {source_name}")

        result = {
            "dtype": "fp32",
            "export_bytes": exported_file.stat().st_size,
            "export_keys": len(exported_keys),
            "merged_internal_tensors": len(merged_mapping),
            "packed_numel": packed_numel,
            "packed_parameter_objects": packed_parameter_objects,
            "split_parameter_objects": len(split_parameters),
            "split_projection_tensors": len(merged_sources),
            "status": "REAL_PACK_EXPORT_RELOAD_OK",
        }
        success = True
    finally:
        packed_model = None
        split_model = None
        packed_parameters = None
        split_parameters = None
        gc.collect()
        try:
            from fastvideo.distributed import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        finally:
            if success and not args.keep_output:
                shutil.rmtree(work_dir)

    if result is not None:
        result["scratch_retained"] = bool(args.keep_output)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
