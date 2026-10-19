# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory export of model-only DCP state for inference.

The modular trainer stores resumable state with PyTorch Distributed
Checkpoint (DCP), while FastVideo inference consumes a Diffusers-style model
directory.  This module bridges those formats without gathering a full model
in memory: rank 0 reads one bounded shard at a time from an already-complete
role/module-only DCP checkpoint and publishes an immutable inference directory
with an atomic rename.

The manager-facing entry point is :func:`export_inference_checkpoint`; the
lower-level :func:`export_inference_checkpoint_from_dcp` exposes the same
rank-0-only conversion with a run-root destination. The checkpoint manager
owns distributed coordination around the temporary DCP save and local export.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

DEFAULT_MAX_SHARD_SIZE_BYTES = 5 * 1024**3
_FORMAT_VERSION = 1
_ALLOWED_NATIVE_EXTRA_KEYS = (re.compile(r"(?:^|\.)attn\.to_gate_compress\.weight$"), )
_CHECKPOINT_WRAPPER_SEGMENT = "_checkpoint_wrapped_module"
_SAFETENSORS_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})
_TORCH_DTYPE_TO_SAFETENSORS = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}


class InferenceCheckpointExportError(RuntimeError):
    """The model-only checkpoint could not be exported safely."""


class UnsupportedMergedReverseMappingError(InferenceCheckpointExportError):
    """A fused training parameter would need to be split for inference."""


def _is_allowed_native_extra(key: str) -> bool:
    return any(pattern.search(key) for pattern in _ALLOWED_NATIVE_EXTRA_KEYS)


@dataclass(frozen=True, slots=True)
class _TensorPlan:
    checkpoint_key: str
    internal_key: str
    output_key: str
    shape: tuple[int, ...]
    source_dtype: torch.dtype
    output_dtype: torch.dtype
    output_nbytes: int
    lora_a_checkpoint_key: str | None = None
    lora_a_shape: tuple[int, ...] | None = None
    lora_a_dtype: torch.dtype | None = None
    lora_b_checkpoint_key: str | None = None
    lora_b_shape: tuple[int, ...] | None = None
    lora_b_dtype: torch.dtype | None = None
    lora_scale: float | None = None
    lora_wrapper: torch.nn.Module | None = None


@dataclass(frozen=True, slots=True)
class _LoraMergeSpec:
    """One training wrapper that must become a native inference weight."""

    prefix: str
    wrapper: torch.nn.Module
    base_weight_key: str
    lora_a_key: str
    lora_b_key: str
    scale: float


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _normalize_output_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported inference checkpoint dtype: {dtype!r}")
        dtype = resolved
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or torch dtype name")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"Inference checkpoint dtype must be floating point, got {dtype}")
    return dtype


def _resolve_dcp_dir(checkpoint: str | os.PathLike[str]) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if path.name != "dcp" and (path / "dcp").is_dir():
        path = path / "dcp"
    if not path.is_dir():
        raise FileNotFoundError(f"Inference checkpoint DCP directory not found: {path}")
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"Incomplete inference checkpoint DCP (missing .metadata): {path}")
    return path


ExpectedShapeOverride = Callable[[str, tuple[int, ...]], tuple[int, ...]]


def _expected_export_shape(
    key: str,
    base_shape: tuple[int, ...],
    override: ExpectedShapeOverride | None,
) -> tuple[tuple[int, ...], str]:
    """The shape an exported tensor must have, and the name of the contract it comes from."""
    if override is None:
        return tuple(int(dim) for dim in base_shape), "base transformer shape"
    return tuple(int(dim) for dim in override(key, base_shape)), "expected transformer shape"


def module_export_hooks(module: torch.nn.Module | None) -> tuple[ExpectedShapeOverride | None, dict[str, Any]]:
    """Read the optional per-model export contract off a live module.

    ``inference_export_expected_shape(output_key, base_shape)`` widens the base
    checkpoint's shape contract for structurally different exports (PDD's
    grid-widened output heads); ``inference_config_overrides()`` names the
    ``config.json`` keys that let the widened model load again unchanged.
    """
    if module is None:
        return None, {}
    shape_hook = getattr(module, "inference_export_expected_shape", None)
    config_hook = getattr(module, "inference_config_overrides", None)
    config_overrides = config_hook() if callable(config_hook) else {}
    if not isinstance(config_overrides, Mapping):
        raise InferenceCheckpointExportError("module.inference_config_overrides() must return a mapping")
    return (shape_hook if callable(shape_hook) else None), dict(config_overrides)


def validate_complete_inference_checkpoint(
    path: Path,
    *,
    step: int,
    expected_shape_override: ExpectedShapeOverride | None = None,
    expected_dtype: torch.dtype | str | None = None,
    expected_config_overrides: Mapping[str, Any] | None = None,
    expected_role: str | None = None,
    expected_module: str | None = None,
    expected_json_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path | None:
    """Validate a published inference checkpoint without loading its tensors.

    ``expected_shape_override`` maps a base-checkpoint tensor shape to the
    shape the exported model is expected to carry (PDD-widened heads); without
    it every exported tensor must match the base transformer exactly.
    """
    if not (path.exists() or path.is_symlink()):
        return None
    complete_path = path / ".complete"
    metadata_path = path / "metadata.json"
    if not complete_path.is_file() or not metadata_path.is_file():
        raise InferenceCheckpointExportError(f"Refusing to overwrite incomplete inference checkpoint: {path}")
    try:
        complete_marker = complete_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InferenceCheckpointExportError(f"Cannot read inference completion marker: {complete_path}") from exc
    if complete_marker != "complete\n":
        raise InferenceCheckpointExportError(f"Invalid inference completion marker: {complete_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceCheckpointExportError(
            f"Invalid metadata for existing inference checkpoint: {metadata_path}") from exc
    if (metadata.get("format_version") != _FORMAT_VERSION or metadata.get("kind") != "inference"
            or metadata.get("step") != step):
        raise InferenceCheckpointExportError(
            f"Existing inference checkpoint metadata does not match step={step}: {metadata_path}")
    role = metadata.get("role")
    dtype = metadata.get("dtype")
    if expected_role is not None and role != expected_role:
        raise InferenceCheckpointExportError(f"Inference checkpoint role {role!r} != requested {expected_role!r}")
    if not isinstance(role, str) or not role or "." in role or dtype not in {"bfloat16", "float16", "float32"}:
        raise InferenceCheckpointExportError(f"Inference checkpoint role/dtype metadata is invalid: {metadata_path}")
    normalized_expected_dtype = (_normalize_output_dtype(expected_dtype) if expected_dtype is not None else None)
    if normalized_expected_dtype is not None and dtype != str(normalized_expected_dtype).removeprefix("torch."):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint dtype {dtype!r} does not match requested dtype "
            f"{str(normalized_expected_dtype).removeprefix('torch.')!r}: {metadata_path}")
    expected_safetensors_dtype = (_TORCH_DTYPE_TO_SAFETENSORS.get(normalized_expected_dtype)
                                  if normalized_expected_dtype is not None else None)
    module_name = str(metadata.get("module") or "")
    if expected_module is not None and module_name != expected_module:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint module {module_name!r} != requested {expected_module!r}")
    if not module_name or "/" in module_name or "\\" in module_name or module_name in {".", ".."}:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint has an invalid module name {module_name!r}: {metadata_path}")
    module_dir = path / module_name
    index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its module index: {index_path}")
    component_config_path = module_dir / "config.json"
    if not component_config_path.is_file():
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its module config: {module_dir}")
    if expected_config_overrides is not None:
        try:
            component_config = json.loads(component_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceCheckpointExportError(
                f"Invalid inference checkpoint component config: {component_config_path}") from exc
        if not isinstance(component_config, dict):
            raise InferenceCheckpointExportError(
                f"Inference checkpoint component config must be an object: {component_config_path}")
        mismatches = {
            key: (component_config.get(key), value)
            for key, value in expected_config_overrides.items() if component_config.get(key) != value
        }
        if mismatches:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint component config overrides do not match: {mismatches}")
    if not any((path / name).is_file() for name in ("model_index.json", "modular_model_index.json")):
        raise InferenceCheckpointExportError(f"Inference checkpoint is missing its model index: {path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceCheckpointExportError(f"Invalid inference checkpoint index: {index_path}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise InferenceCheckpointExportError(f"Inference checkpoint index has no weight map: {index_path}")
    expected_by_shard: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str) or not filename:
            raise InferenceCheckpointExportError(f"Invalid key/shard entry in {index_path}: {key!r} -> {filename!r}")
        if Path(filename).name != filename:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint index contains an invalid shard path: {filename!r}")
        expected_by_shard.setdefault(filename, set()).add(key)
    actual_shards = {shard.name for shard in module_dir.glob("*.safetensors") if shard.is_file()}
    expected_shards = set(expected_by_shard)
    if actual_shards != expected_shards:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shards differ from its index under {module_dir}: "
            f"missing={sorted(expected_shards - actual_shards)} extra={sorted(actual_shards - expected_shards)}")
    shard_sizes: list[int] = []
    output_shapes: dict[str, tuple[int, ...]] = {}
    for filename in sorted(expected_by_shard):
        expected_keys = expected_by_shard[filename]
        shard_path = module_dir / filename
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                for key in actual_keys:
                    tensor_slice = handle.get_slice(key)
                    output_shapes[key] = tuple(int(dim) for dim in tensor_slice.get_shape())
                    actual_dtype = tensor_slice.get_dtype()
                    if (expected_safetensors_dtype is not None and actual_dtype in _SAFETENSORS_FLOAT_DTYPES
                            and actual_dtype != expected_safetensors_dtype):
                        raise InferenceCheckpointExportError(
                            f"Inference checkpoint tensor {key!r} has dtype {actual_dtype}, "
                            f"expected {expected_safetensors_dtype}: {shard_path}")
        except InferenceCheckpointExportError:
            raise
        except Exception as exc:
            raise InferenceCheckpointExportError(f"Cannot read inference checkpoint shard: {shard_path}") from exc
        if actual_keys != expected_keys:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint shard keys differ from its index for {shard_path}: "
                f"missing={sorted(expected_keys - actual_keys)[:10]} extra={sorted(actual_keys - expected_keys)[:10]}")
        shard_sizes.append(shard_path.stat().st_size)
    if int(metadata.get("tensor_count", -1)) != len(weight_map):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint tensor_count does not match its index: {metadata_path}")
    if int(metadata.get("shard_count", -1)) != len(expected_shards):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shard_count does not match its index: {metadata_path}")
    logical_shard_sizes = metadata.get("shard_sizes")
    total_size = metadata.get("total_size")
    max_shard_size = metadata.get("max_shard_size_bytes")
    index_metadata = index.get("metadata")
    index_total_size = index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    if (not isinstance(logical_shard_sizes, list) or len(logical_shard_sizes) != len(expected_shards)
            or any(not isinstance(size, int) or size < 0
                   for size in logical_shard_sizes) or not isinstance(max_shard_size, int) or max_shard_size <= 0
            or any(size > max_shard_size for size in logical_shard_sizes) or not isinstance(total_size, int)
            or total_size != sum(logical_shard_sizes) or index_total_size != total_size):
        raise InferenceCheckpointExportError(
            f"Inference checkpoint logical shard sizes are inconsistent: {metadata_path}")
    recorded_shard_sizes = metadata.get("shard_file_sizes")
    if not isinstance(recorded_shard_sizes, list) or recorded_shard_sizes != shard_sizes:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint shard_file_sizes do not match files on disk: {metadata_path}")

    base_model_dir = metadata.get("base_model_dir")
    if not isinstance(base_model_dir, str) or not base_model_dir:
        raise InferenceCheckpointExportError(f"Inference checkpoint has no base_model_dir: {metadata_path}")
    base_shapes = _component_tensor_shapes(Path(base_model_dir) / module_name)
    for key, shape in output_shapes.items():
        expected_shape = base_shapes.get(key)
        if expected_shape is None:
            if not _is_allowed_native_extra(key):
                raise InferenceCheckpointExportError(
                    f"Inference checkpoint contains unknown non-native tensor {key!r}: {path}")
            continue
        expected_shape, contract = _expected_export_shape(key, expected_shape, expected_shape_override)
        if shape != expected_shape:
            raise InferenceCheckpointExportError(
                f"Inference checkpoint tensor {key!r} shape {shape} != {contract} {expected_shape}")
    missing_base_keys = set(base_shapes) - set(output_shapes)
    if missing_base_keys:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint is missing {len(missing_base_keys)} base transformer tensors; "
            f"first={sorted(missing_base_keys)[:10]}")
    saved_overrides = metadata.get("json_overrides", {})
    _validate_export_json_overrides(path, saved_overrides)
    if expected_json_overrides is not None:
        if saved_overrides != dict(expected_json_overrides):
            raise InferenceCheckpointExportError("Inference JSON override contract changed")
        _validate_export_json_overrides(path, expected_json_overrides)
    return path


def _component_tensor_shapes(module_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read the base component's exact key/shape contract from safetensors headers."""
    index_candidates = (
        module_dir / "diffusion_pytorch_model.safetensors.index.json",
        module_dir / "model.safetensors.index.json",
    )
    index_path = next((path for path in index_candidates if path.is_file()), None)
    expected_files: set[str] | None = None
    indexed_keys: set[str] | None = None
    if index_path is not None:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceCheckpointExportError(f"Invalid base transformer index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise InferenceCheckpointExportError(f"Base transformer index has no weight map: {index_path}")
        indexed_keys = set(weight_map)
        expected_files = {str(filename) for filename in weight_map.values()}

    files = sorted(module_dir.glob("*.safetensors"))
    if expected_files is not None:
        files = [module_dir / filename for filename in sorted(expected_files)]
    if not files or any(not path.is_file() for path in files):
        raise InferenceCheckpointExportError(f"Base transformer safetensors are incomplete under {module_dir}")

    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                # ``safe_open`` exposes ``keys()`` but is not itself iterable.
                for key in handle.keys():  # noqa: SIM118
                    if key in shapes:
                        raise InferenceCheckpointExportError(f"Duplicate base transformer tensor key {key!r}")
                    shapes[key] = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
        except InferenceCheckpointExportError:
            raise
        except Exception as exc:
            raise InferenceCheckpointExportError(f"Cannot read base transformer safetensors: {path}") from exc
    if indexed_keys is not None and set(shapes) != indexed_keys:
        raise InferenceCheckpointExportError(
            f"Base transformer index/header mismatch under {module_dir}: "
            f"missing={sorted(indexed_keys - set(shapes))[:10]} extra={sorted(set(shapes) - indexed_keys)[:10]}")
    return shapes


def _canonical_state_prefix(module_name: str) -> str:
    """Map module traversal names to checkpoint-wrapper-free state keys."""
    return ".".join(segment for segment in module_name.split(".") if segment != _CHECKPOINT_WRAPPER_SEGMENT)


def _lora_merge_specs(module: torch.nn.Module) -> dict[str, _LoraMergeSpec]:
    """Describe every training LoRA wrapper without materializing its state."""
    from fastvideo.layers.lora.linear import BaseLayerWithLoRA

    specs: dict[str, _LoraMergeSpec] = {}
    for module_name, child in module.named_modules():
        if not module_name or not isinstance(child, BaseLayerWithLoRA):
            continue
        prefix = _canonical_state_prefix(module_name)
        if not prefix:
            raise InferenceCheckpointExportError(f"LoRA module {module_name!r} has an empty state prefix")
        base_weight_key = f"{prefix}.base_layer.weight"
        if base_weight_key in specs:
            raise InferenceCheckpointExportError(f"LoRA modules collide at state prefix {prefix!r}")
        rank = child.lora_rank
        if rank is None or int(rank) <= 0:
            raise InferenceCheckpointExportError(f"LoRA module {prefix!r} has invalid rank {rank!r}")
        if child.lora_A is None or child.lora_B is None:
            raise InferenceCheckpointExportError(f"LoRA module {prefix!r} has no adapter parameters")
        alpha = child.lora_alpha if child.lora_alpha is not None else rank
        specs[base_weight_key] = _LoraMergeSpec(
            prefix=prefix,
            wrapper=child,
            base_weight_key=base_weight_key,
            lora_a_key=f"{prefix}.lora_A",
            lora_b_key=f"{prefix}.lora_B",
            scale=(float(alpha) / float(rank)) * float(child.lora_strength),
        )
    return specs


def _mapping_output_key(
    internal_key: str,
    reverse_mapping: Mapping[str, Any],
    *,
    native_fallback_key: str | None = None,
) -> str:
    entry = reverse_mapping.get(internal_key)
    if entry is None:
        # FastVideo-native additions such as MiniMax-H3's learned
        # ``attn.to_gate_compress`` VSA parameters intentionally have no key in
        # the base checkpoint.  The component loader accepts their native name.
        return native_fallback_key or internal_key
    if not isinstance(entry, tuple | list) or len(entry) != 3:
        raise InferenceCheckpointExportError(f"Invalid reverse mapping for {internal_key!r}: expected "
                                             "(output_key, merge_index, num_params_to_merge)")
    output_key, merge_index, num_params_to_merge = entry
    if merge_index is not None or num_params_to_merge not in (None, 1):
        raise UnsupportedMergedReverseMappingError(f"Cannot stream merged reverse mapping for {internal_key!r}: "
                                                   f"output_key={output_key!r}, merge_index={merge_index!r}, "
                                                   f"num_params_to_merge={num_params_to_merge!r}. "
                                                   "A model-specific split exporter is required.")
    if not isinstance(output_key, str) or not output_key:
        raise InferenceCheckpointExportError(
            f"Invalid output key in reverse mapping for {internal_key!r}: {output_key!r}")
    return output_key


def _tensor_output_dtype(source_dtype: torch.dtype, configured_dtype: torch.dtype) -> torch.dtype:
    if torch.empty((), dtype=source_dtype).is_floating_point():
        return configured_dtype
    return source_dtype


def _build_tensor_plan(
    *,
    dcp_dir: Path,
    state_prefix: str,
    reverse_mapping: Mapping[str, Any],
    base_shapes: Mapping[str, tuple[int, ...]],
    output_dtype: torch.dtype,
    max_shard_size_bytes: int,
    lora_specs: Mapping[str, _LoraMergeSpec],
    expected_shape_override: ExpectedShapeOverride | None = None,
) -> list[_TensorPlan]:
    metadata = FileSystemReader(str(dcp_dir)).read_metadata()
    plans: list[_TensorPlan] = []
    output_keys: set[str] = set()
    state_metadata = metadata.state_dict_metadata
    lora_auxiliary_keys = {key for spec in lora_specs.values() for key in (spec.lora_a_key, spec.lora_b_key)}
    found_lora_weights: set[str] = set()

    for checkpoint_key in sorted(state_metadata):
        if not checkpoint_key.startswith(state_prefix):
            continue
        internal_key = checkpoint_key[len(state_prefix):]
        if internal_key in lora_auxiliary_keys:
            continue

        tensor_metadata = state_metadata[checkpoint_key]
        properties = getattr(tensor_metadata, "properties", None)
        shape = getattr(tensor_metadata, "size", None)
        source_dtype = getattr(properties, "dtype", None)
        if shape is None or not isinstance(source_dtype, torch.dtype):
            raise InferenceCheckpointExportError(f"Inference state contains a non-tensor value at {checkpoint_key!r}; "
                                                 "safetensors exports support tensors only")

        if not internal_key:
            raise InferenceCheckpointExportError(f"Empty module key under DCP prefix {state_prefix!r}")

        native_internal_key = internal_key
        lora_spec = lora_specs.get(internal_key)
        if ".base_layer." in internal_key:
            native_internal_key = internal_key.replace(".base_layer.", ".", 1)
        # The FSDP loader normally aliases the reverse mapping to the
        # transformed ``base_layer`` key. Prefer that production form, while
        # accepting a native-key mapping for older/custom loaders.
        mapping_key = internal_key
        if mapping_key not in reverse_mapping and native_internal_key in reverse_mapping:
            mapping_key = native_internal_key
        if lora_spec is not None:
            found_lora_weights.add(internal_key)
        output_key = _mapping_output_key(
            mapping_key,
            reverse_mapping,
            native_fallback_key=native_internal_key,
        )
        tensor_shape = tuple(int(dim) for dim in shape)
        expected_shape = base_shapes.get(output_key)
        if expected_shape is None:
            if mapping_key in reverse_mapping or not _is_allowed_native_extra(output_key):
                raise InferenceCheckpointExportError(
                    f"Inference tensor {internal_key!r} maps to unknown base key {output_key!r}; "
                    "only MiniMax-H3 attn.to_gate_compress.weight may be exported as a native extra")
        else:
            expected_shape, contract = _expected_export_shape(output_key, expected_shape, expected_shape_override)
            if tensor_shape != expected_shape:
                raise InferenceCheckpointExportError(f"Inference tensor {output_key!r} shape {tensor_shape} != "
                                                     f"{contract} {expected_shape}")
        if output_key in output_keys:
            raise InferenceCheckpointExportError(f"Reverse mapping produces duplicate inference key {output_key!r}")
        output_keys.add(output_key)

        tensor_output_dtype = _tensor_output_dtype(source_dtype, output_dtype)
        numel = 1
        for dim in tensor_shape:
            numel *= dim
        output_nbytes = numel * torch.empty((), dtype=tensor_output_dtype).element_size()
        if output_nbytes > max_shard_size_bytes:
            raise InferenceCheckpointExportError(
                f"Tensor {checkpoint_key!r} requires {output_nbytes} bytes after casting, "
                f"which exceeds max_shard_size_bytes={max_shard_size_bytes}")
        lora_fields: dict[str, Any] = {}
        if lora_spec is not None:
            auxiliary = []
            for name, auxiliary_internal_key in (
                ("lora_a", lora_spec.lora_a_key),
                ("lora_b", lora_spec.lora_b_key),
            ):
                auxiliary_checkpoint_key = f"{state_prefix}{auxiliary_internal_key}"
                auxiliary_metadata = state_metadata.get(auxiliary_checkpoint_key)
                if auxiliary_metadata is None:
                    raise InferenceCheckpointExportError(
                        f"LoRA inference export is missing {auxiliary_internal_key!r} for {internal_key!r}")
                auxiliary_properties = getattr(auxiliary_metadata, "properties", None)
                auxiliary_shape = getattr(auxiliary_metadata, "size", None)
                auxiliary_dtype = getattr(auxiliary_properties, "dtype", None)
                if auxiliary_shape is None or not isinstance(auxiliary_dtype, torch.dtype):
                    raise InferenceCheckpointExportError(
                        f"LoRA inference state contains a non-tensor value at {auxiliary_checkpoint_key!r}")
                auxiliary.append(
                    (name, auxiliary_checkpoint_key, tuple(int(dim) for dim in auxiliary_shape), auxiliary_dtype))
            (_, lora_a_checkpoint_key, lora_a_shape, lora_a_dtype), (_, lora_b_checkpoint_key, lora_b_shape,
                                                                     lora_b_dtype) = auxiliary
            if (len(lora_a_shape) != 2 or len(lora_b_shape) != 2 or lora_a_shape[0] != lora_b_shape[1]
                    or lora_b_shape[0] != tensor_shape[0] or lora_a_shape[1] != tensor_shape[1]):
                raise InferenceCheckpointExportError(
                    f"LoRA shapes for {internal_key!r} cannot merge into {tensor_shape}: "
                    f"A={lora_a_shape}, B={lora_b_shape}")
            lora_fields = {
                "lora_a_checkpoint_key": lora_a_checkpoint_key,
                "lora_a_shape": lora_a_shape,
                "lora_a_dtype": lora_a_dtype,
                "lora_b_checkpoint_key": lora_b_checkpoint_key,
                "lora_b_shape": lora_b_shape,
                "lora_b_dtype": lora_b_dtype,
                "lora_scale": lora_spec.scale,
                "lora_wrapper": lora_spec.wrapper,
            }
        plans.append(
            _TensorPlan(
                checkpoint_key=checkpoint_key,
                internal_key=internal_key,
                output_key=output_key,
                shape=tensor_shape,
                source_dtype=source_dtype,
                output_dtype=tensor_output_dtype,
                output_nbytes=output_nbytes,
                **lora_fields,
            ))

    if not plans:
        raise InferenceCheckpointExportError(f"No tensor keys found under DCP prefix {state_prefix!r} in {dcp_dir}")
    missing_lora_weights = set(lora_specs) - found_lora_weights
    if missing_lora_weights:
        raise InferenceCheckpointExportError("LoRA inference export is missing base weights for wrappers; "
                                             f"first={sorted(missing_lora_weights)[:10]}")
    missing_base_keys = set(base_shapes) - output_keys
    if missing_base_keys:
        raise InferenceCheckpointExportError(
            f"Inference checkpoint is missing {len(missing_base_keys)} base transformer tensors; "
            f"first={sorted(missing_base_keys)[:10]}")
    return plans


def _group_shards(plans: list[_TensorPlan], max_shard_size_bytes: int) -> list[list[_TensorPlan]]:
    shards: list[list[_TensorPlan]] = []
    current: list[_TensorPlan] = []
    current_nbytes = 0
    for plan in plans:
        if current and current_nbytes + plan.output_nbytes > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_nbytes = 0
        current.append(plan)
        current_nbytes += plan.output_nbytes
    if current:
        shards.append(current)
    return shards


def _prepare_model_layout(temp_dir: Path, base_model_dir: Path, module_name: str) -> Path:
    if not any((base_model_dir / name).is_file() for name in ("model_index.json", "modular_model_index.json")):
        raise FileNotFoundError(
            f"Base model directory has no model_index.json or modular_model_index.json: {base_model_dir}")
    base_module_dir = base_model_dir / module_name
    base_config = base_module_dir / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(f"Base model component config not found: {base_config}")

    module_dir = temp_dir / module_name
    module_dir.mkdir(parents=True)
    # Content only: a read-only warm-start source must not make the exported
    # config read-only before its overrides are written.
    shutil.copyfile(base_config, module_dir / "config.json")

    # Deployment receipts describe this export, not its initialization source.
    # Linking them here would let the later write corrupt a warm-start checkpoint.
    reserved = {module_name, "metadata.json", ".complete", "fastvideo_inference.json", "export_manifest.json"}
    for entry in sorted(base_model_dir.iterdir(), key=lambda item: item.name):
        if entry.name in reserved or entry.name == ".cache" or entry.name.startswith(".git"):
            continue
        target = temp_dir / entry.name
        target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    return module_dir


def apply_export_json_overrides(root: Path, overrides: Mapping[str, Mapping[str, Any]]) -> None:
    """Merge small deployment JSONs without writing through linked base components."""
    for name, values in overrides.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) not in (1, 2)
                or relative.suffix != ".json" or not isinstance(values, Mapping)):
            raise InferenceCheckpointExportError(f"Invalid export JSON override: {name!r}")
        target = root / relative
        parent = target.parent
        if parent.is_symlink():
            source = parent.resolve(strict=True)
            parent.unlink()
            parent.mkdir()
            for child in source.iterdir():
                (parent / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
        parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            _apply_config_overrides(target, values)
        else:
            target.write_text(json.dumps(dict(values), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_export_json_overrides(root: Path, overrides: Mapping[str, Mapping[str, Any]]) -> None:
    if not isinstance(overrides, Mapping):
        raise InferenceCheckpointExportError("Inference JSON overrides must be a mapping")
    for name, values in overrides.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) not in (1, 2)
                or relative.suffix != ".json" or not isinstance(values, Mapping)):
            raise InferenceCheckpointExportError(f"Invalid export JSON override: {name!r}")
        try:
            actual = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InferenceCheckpointExportError(f"Missing or invalid inference metadata: {name}") from exc
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in values.items()):
            raise InferenceCheckpointExportError(f"Inference metadata mismatch: {name}")


def _apply_config_overrides(config_path: Path, overrides: Mapping[str, Any]) -> None:
    """Record model-declared architecture keys in the exported component config."""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceCheckpointExportError(f"Invalid exported component config: {config_path}") from exc
    if not isinstance(config, dict):
        raise InferenceCheckpointExportError(f"Exported component config must be a JSON object: {config_path}")
    config.update(overrides)
    # Export layouts may hard-link the base config. Replace this directory
    # entry instead of mutating the initialization checkpoint's inode.
    with tempfile.NamedTemporaryFile(mode="w",
                                     encoding="utf-8",
                                     dir=config_path.parent,
                                     prefix=".config-",
                                     suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_tensor_shards(
    *,
    dcp_dir: Path,
    module_dir: Path,
    shards: list[list[_TensorPlan]],
) -> tuple[dict[str, str], int, list[int], list[int]]:
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_sizes: list[int] = []
    shard_file_sizes: list[int] = []
    shard_count = len(shards)

    for shard_index, shard in enumerate(shards, start=1):
        filename = (f"diffusion_pytorch_model-{shard_index:05d}-of-{shard_count:05d}.safetensors")
        state: dict[str, torch.Tensor] = {}
        for plan in shard:
            state[plan.checkpoint_key] = torch.empty(plan.shape, dtype=plan.source_dtype, device="cpu")
            if plan.lora_a_checkpoint_key is not None:
                if (plan.lora_a_shape is None or plan.lora_a_dtype is None or plan.lora_b_checkpoint_key is None
                        or plan.lora_b_shape is None or plan.lora_b_dtype is None):
                    raise InferenceCheckpointExportError(f"Incomplete LoRA tensor plan for {plan.internal_key!r}")
                state[plan.lora_a_checkpoint_key] = torch.empty(
                    plan.lora_a_shape,
                    dtype=plan.lora_a_dtype,
                    device="cpu",
                )
                state[plan.lora_b_checkpoint_key] = torch.empty(
                    plan.lora_b_shape,
                    dtype=plan.lora_b_dtype,
                    device="cpu",
                )
        # This export is deliberately rank-local.  DCP assembles the full CPU
        # tensors from its storage shards without using the live process group.
        dcp.load(state, checkpoint_id=str(dcp_dir), no_dist=True)

        output_tensors: dict[str, torch.Tensor] = {}
        shard_size = 0
        for plan in shard:
            tensor = state[plan.checkpoint_key]
            if plan.lora_a_checkpoint_key is not None:
                if (plan.lora_b_checkpoint_key is None or plan.lora_scale is None or plan.lora_wrapper is None):
                    raise InferenceCheckpointExportError(f"Incomplete LoRA merge plan for {plan.internal_key!r}")
                lora_a = state[plan.lora_a_checkpoint_key].to(tensor)
                lora_b = state[plan.lora_b_checkpoint_key].to(tensor)
                slice_a = getattr(plan.lora_wrapper, "slice_lora_a_weights", None)
                slice_b = getattr(plan.lora_wrapper, "slice_lora_b_weights", None)
                if not callable(slice_a) or not callable(slice_b):
                    raise InferenceCheckpointExportError(
                        f"LoRA wrapper for {plan.internal_key!r} cannot slice adapter weights")
                # Merge at the master-weight dtype before the configured
                # inference cast. Reuse the staged base tensor in place so a
                # large H3 projection never has two full CPU copies alive.
                tensor.addmm_(
                    slice_b(lora_b),
                    slice_a(lora_a),
                    alpha=float(plan.lora_scale),
                )
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=plan.output_dtype)
            output_tensors[plan.output_key] = tensor.detach().cpu().contiguous()
            weight_map[plan.output_key] = filename
            shard_size += plan.output_nbytes
        save_file(output_tensors, module_dir / filename)
        total_size += shard_size
        shard_sizes.append(shard_size)
        shard_file_sizes.append((module_dir / filename).stat().st_size)
        del output_tensors, state

    index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map,
    }
    index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return weight_map, total_size, shard_sizes, shard_file_sizes


def export_inference_checkpoint_from_dcp(
    *,
    dcp_checkpoint: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    step: int,
    module: torch.nn.Module,
    base_model_dir: str | os.PathLike[str],
    role: str = "student",
    module_name: str = "transformer",
    dtype: torch.dtype | str = torch.bfloat16,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    raw_config: Mapping[str, Any] | None = None,
    json_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Export one role/module-only DCP as an immutable inference checkpoint.

    ``CheckpointManager`` is expected to call this function on rank 0 after a
    collective model-only DCP save has completed. ``dcp_checkpoint`` may name
    that DCP directory directly or its parent containing ``dcp/``. The source
    is never modified or removed.

    Floating tensors are cast into ``dtype`` in independent CPU buffers;
    integer and boolean tensors retain their source dtype. Parameter names are
    converted with ``module.reverse_param_names_mapping``. Unmapped names are
    retained for FastVideo-native inference parameters such as MiniMax-H3 VSA
    gates. Merged mappings fail rather than silently writing incompatible
    weights.

    The completed model is atomically renamed to
    ``<output_dir>/inference/checkpoint-<step>``. A valid existing completed
    export is returned unchanged, making retries idempotent.
    """

    if _rank() != 0:
        raise InferenceCheckpointExportError("export_inference_checkpoint_from_dcp is rank-0-only; "
                                             "the checkpoint manager must coordinate other ranks")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if not role or "." in role:
        raise ValueError(f"role must be a non-empty DCP key segment, got {role!r}")
    if not module_name or "." in module_name:
        raise ValueError(f"module_name must be a non-empty DCP key segment, got {module_name!r}")
    if not 0 < max_shard_size_bytes <= DEFAULT_MAX_SHARD_SIZE_BYTES:
        raise ValueError("max_shard_size_bytes must be in "
                         f"[1, {DEFAULT_MAX_SHARD_SIZE_BYTES}], got {max_shard_size_bytes}")

    output_dtype = _normalize_output_dtype(dtype)
    expected_shape_override, config_overrides = module_export_hooks(module)
    run_output_dir = Path(output_dir).expanduser().resolve()
    inference_root = run_output_dir / "inference"
    final_dir = inference_root / f"checkpoint-{step}"
    existing = validate_complete_inference_checkpoint(
        final_dir,
        step=step,
        expected_shape_override=expected_shape_override,
        expected_dtype=output_dtype,
        expected_config_overrides=config_overrides,
        expected_role=role,
        expected_module=module_name,
        expected_json_overrides=json_overrides,
    )
    if existing is not None:
        return existing

    dcp_dir = _resolve_dcp_dir(dcp_checkpoint)
    base_dir = Path(base_model_dir).expanduser().resolve()
    base_shapes = _component_tensor_shapes(base_dir / module_name)
    reverse_mapping = getattr(module, "reverse_param_names_mapping", {})
    if reverse_mapping is None:
        reverse_mapping = {}
    if not isinstance(reverse_mapping, Mapping):
        raise InferenceCheckpointExportError("module.reverse_param_names_mapping must be a mapping")
    lora_specs = _lora_merge_specs(module)

    state_prefix = f"roles.{role}.{module_name}."
    plans = _build_tensor_plan(
        dcp_dir=dcp_dir,
        state_prefix=state_prefix,
        reverse_mapping=reverse_mapping,
        base_shapes=base_shapes,
        output_dtype=output_dtype,
        max_shard_size_bytes=max_shard_size_bytes,
        lora_specs=lora_specs,
        expected_shape_override=expected_shape_override,
    )
    shards = _group_shards(plans, max_shard_size_bytes)

    inference_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(
        prefix=f".checkpoint-{step}.tmp-",
        dir=str(inference_root),
    ))
    try:
        module_dir = _prepare_model_layout(temp_dir, base_dir, module_name)
        apply_export_json_overrides(temp_dir, json_overrides or {})
        if config_overrides:
            _apply_config_overrides(module_dir / "config.json", config_overrides)
        weight_map, total_size, shard_sizes, shard_file_sizes = _write_tensor_shards(
            dcp_dir=dcp_dir,
            module_dir=module_dir,
            shards=shards,
        )
        metadata = {
            "format_version": _FORMAT_VERSION,
            "kind": "inference",
            "step": step,
            "role": role,
            "module": module_name,
            "dtype": str(output_dtype).removeprefix("torch."),
            "base_model_dir": str(base_dir),
            "tensor_count": len(weight_map),
            "total_size": total_size,
            "shard_count": len(shards),
            "shard_sizes": shard_sizes,
            "shard_file_sizes": shard_file_sizes,
            "max_shard_size_bytes": max_shard_size_bytes,
            "lora": "merged" if lora_specs else "none",
        }
        if json_overrides:
            metadata["json_overrides"] = dict(json_overrides)
        if config_overrides:
            metadata["config_overrides"] = config_overrides
        if raw_config is not None:
            metadata["config"] = raw_config
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Written last inside the private temp directory. The subsequent
        # same-filesystem rename publishes the complete tree in one operation.
        (temp_dir / ".complete").write_text("complete\n", encoding="utf-8")
        validate_complete_inference_checkpoint(
            temp_dir,
            step=step,
            expected_shape_override=expected_shape_override,
            expected_dtype=output_dtype,
            expected_config_overrides=config_overrides,
            expected_role=role,
            expected_module=module_name,
            expected_json_overrides=json_overrides,
        )
        try:
            temp_dir.rename(final_dir)
        except FileExistsError as exc:
            raise InferenceCheckpointExportError(f"Inference checkpoint appeared concurrently: {final_dir}") from exc
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    validated = validate_complete_inference_checkpoint(
        final_dir,
        step=step,
        expected_shape_override=expected_shape_override,
        expected_dtype=output_dtype,
        expected_config_overrides=config_overrides,
        expected_role=role,
        expected_module=module_name,
        expected_json_overrides=json_overrides,
    )
    if validated is None:
        raise InferenceCheckpointExportError(f"Published inference checkpoint disappeared: {final_dir}")
    return validated


def export_inference_checkpoint(
    *,
    dcp_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    base_model_path: str | os.PathLike[str],
    role: str,
    modules: Mapping[str, torch.nn.Module],
    dtype: torch.dtype | str,
    step: int,
    raw_config: Mapping[str, Any] | None = None,
    json_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """CheckpointManager adapter for one deployable role/module checkpoint.

    ``output_dir`` is the manager's fully resolved target,
    ``<run>/inference/checkpoint-<step>``. The temporary DCP may contain only
    the role/module state selected by ``modules``. Multi-component deployment
    is intentionally rejected until its component layout and atomicity
    contract are defined.

    ``raw_config`` is persisted in ``metadata.json`` when supplied, matching
    resumable checkpoint provenance. The ephemeral DCP staging path is not
    persisted because the manager removes it after a successful export.
    """

    if len(modules) != 1:
        raise InferenceCheckpointExportError("Inference checkpoint export currently supports exactly one module; "
                                             f"got {sorted(modules)}")
    module_name, module = next(iter(modules.items()))
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Inference checkpoint module {module_name!r} must be a torch.nn.Module")

    final_dir = Path(output_dir).expanduser().resolve()
    expected_name = f"checkpoint-{step}"
    if final_dir.name != expected_name or final_dir.parent.name != "inference":
        raise ValueError("CheckpointManager output_dir must be "
                         f"<run>/inference/{expected_name}, got {final_dir}")
    run_output_dir = final_dir.parent.parent
    return export_inference_checkpoint_from_dcp(
        dcp_checkpoint=dcp_dir,
        output_dir=run_output_dir,
        step=step,
        module=module,
        base_model_dir=base_model_path,
        role=role,
        module_name=module_name,
        dtype=dtype,
        raw_config=raw_config,
        json_overrides=json_overrides,
    )


__all__ = [
    "DEFAULT_MAX_SHARD_SIZE_BYTES",
    "InferenceCheckpointExportError",
    "UnsupportedMergedReverseMappingError",
    "export_inference_checkpoint",
    "export_inference_checkpoint_from_dcp",
    "validate_complete_inference_checkpoint",
]
