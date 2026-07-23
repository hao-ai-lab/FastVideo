# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

ReverseParamMappingEntry: TypeAlias = (
    tuple[str, int | None, int | None]
    | tuple[str, int | None, int | None, int]
)
ReverseParamNamesMapping: TypeAlias = dict[
    str,
    ReverseParamMappingEntry | list[ReverseParamMappingEntry],
]


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def get_param_names_mapping(
        mapping_dict: dict[str, Any]) -> Callable[[str], tuple[str, Any, Any]]:
    """
    Creates a mapping function that transforms parameter names using regex patterns.
    
    Args:
        mapping_dict (Dict[str, str]): Dictionary mapping regex patterns to replacement patterns
        param_name (str): The parameter name to be transformed
        
    Returns:
        Callable[[str], str]: A function that maps parameter names from source to target format
    """

    def mapping_fn(name: str) -> tuple[str, Any, Any]:
        # Try to match and transform the name using the regex patterns in mapping_dict
        for pattern, replacement in mapping_dict.items():
            match = re.match(pattern, name)
            if match:
                merge_index = None
                total_splitted_params = None
                if isinstance(replacement, tuple):
                    merge_index = replacement[1]
                    total_splitted_params = replacement[2]
                    replacement = replacement[0]
                name = re.sub(pattern, replacement, name)
                return name, merge_index, total_splitted_params

        # If no pattern matches, return the original name
        return name, None, None

    return mapping_fn


def hf_to_custom_state_dict(
    hf_param_sd: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
) -> tuple[dict[str, torch.Tensor], ReverseParamNamesMapping]:
    """
    Converts a Hugging Face parameter state dictionary to a custom parameter state dictionary.
    
    Args:
        hf_param_sd (Dict[str, torch.Tensor]): The Hugging Face parameter state dictionary
        param_names_mapping (Callable[[str], tuple[str, Any, Any]]): A function that maps parameter names from source to target format
        
    Returns:
        custom_param_sd (Dict[str, torch.Tensor]): The custom formatted parameter state dict
        reverse_param_names_mapping: Maps direct targets to a source-name
            3-tuple and merged targets to an ordered list of source-name,
            merge-index, merge-total, and split-size 4-tuples.
    """
    custom_param_sd: dict[str, torch.Tensor] = {}
    to_merge_params: defaultdict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    merge_totals: dict[str, int] = {}
    reverse_param_names_mapping: ReverseParamNamesMapping = {}
    hf_param_items = (
        hf_param_sd.items()
        if isinstance(hf_param_sd, Mapping)
        else hf_param_sd
    )
    for source_param_name, full_tensor in hf_param_items:
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name)
        if merge_index is not None:
            if num_params_to_merge is None:
                raise ValueError(f"Missing merge total for {source_param_name!r}")
            merge_index = int(merge_index)
            num_params_to_merge = int(num_params_to_merge)
            if full_tensor.ndim == 0:
                raise ValueError(f"Cannot merge scalar parameter {source_param_name!r}")
            if num_params_to_merge <= 0 or not 0 <= merge_index < num_params_to_merge:
                raise ValueError(
                    f"Invalid merge metadata for {source_param_name!r}: "
                    f"index={merge_index}, total={num_params_to_merge}"
                )
            previous_total = merge_totals.setdefault(target_param_name, num_params_to_merge)
            if previous_total != num_params_to_merge:
                raise ValueError(
                    f"Inconsistent merge totals for {target_param_name!r}: "
                    f"{previous_total} and {num_params_to_merge}"
                )
            if merge_index in to_merge_params[target_param_name]:
                raise ValueError(
                    f"Duplicate merge index {merge_index} for {target_param_name!r}"
                )
            reverse_entry = (
                source_param_name,
                merge_index,
                num_params_to_merge,
                int(full_tensor.shape[0]),
            )
            reverse_entries = reverse_param_names_mapping.setdefault(
                target_param_name,
                [],
            )
            if not isinstance(reverse_entries, list):
                raise ValueError(f"Mixed direct and merged mappings for {target_param_name!r}")
            reverse_entries.append(reverse_entry)
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                reverse_entries.sort(key=lambda entry: int(entry[1]))
                expected_indices = set(range(num_params_to_merge))
                if set(to_merge_params[target_param_name]) != expected_indices:
                    raise ValueError(
                        f"Incomplete merge indices for {target_param_name!r}: "
                        f"got {sorted(to_merge_params[target_param_name])}, "
                        f"expected {sorted(expected_indices)}"
                    )
                # cat at output dim according to the merge_index order
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
                del merge_totals[target_param_name]
            else:
                continue
        else:
            if target_param_name in reverse_param_names_mapping:
                raise ValueError(f"Duplicate direct mapping for {target_param_name!r}")
            reverse_param_names_mapping[target_param_name] = (
                source_param_name,
                None,
                None,
            )
        if target_param_name in custom_param_sd:
            raise ValueError(f"Duplicate target parameter {target_param_name!r}")
        custom_param_sd[target_param_name] = full_tensor
    if to_merge_params:
        incomplete = {
            name: sorted(parts)
            for name, parts in sorted(to_merge_params.items())
        }
        raise ValueError(f"Incomplete merged parameters: {incomplete}")
    return custom_param_sd, reverse_param_names_mapping


def custom_to_hf_state_dict(
    state_dict: Mapping[str, Any] | Iterable[tuple[str, Any]],
    reverse_param_names_mapping: ReverseParamNamesMapping,
) -> dict[str, Any]:
    """Convert FastVideo parameter names and fused tensors back to HF format."""
    if not reverse_param_names_mapping:
        raise ValueError("reverse_param_names_mapping is empty")
    state = dict(state_dict)

    def _entries(
        raw: ReverseParamMappingEntry | list[ReverseParamMappingEntry],
    ) -> list[ReverseParamMappingEntry]:
        entries = raw if isinstance(raw, list) else [raw]
        if not entries or not all(isinstance(entry, tuple) for entry in entries):
            raise ValueError(f"Invalid reverse parameter mapping: {raw!r}")
        return entries

    def _unpack(
        entry: ReverseParamMappingEntry,
    ) -> tuple[str, int | None, int | None, int | None]:
        if len(entry) == 3:
            source_key, merge_index, total = entry
            return source_key, merge_index, total, None
        if len(entry) == 4:
            source_key, merge_index, total, split_size = entry
            return source_key, merge_index, total, split_size
        raise ValueError(f"Invalid reverse parameter mapping entry: {entry!r}")

    merge_groups: dict[str, list[tuple[str, int, int, int | None]]] = {}
    for training_key, raw_mapping in reverse_param_names_mapping.items():
        for entry in _entries(raw_mapping):
            source_key, merge_index, merge_total, split_size = _unpack(entry)
            if merge_index is None:
                continue
            if merge_total is None:
                raise ValueError(f"Missing merge total for {training_key!r}")
            merge_groups.setdefault(training_key, []).append((
                source_key,
                int(merge_index),
                int(merge_total),
                None if split_size is None else int(split_size),
            ))

    converted: dict[str, Any] = {}
    used_keys: set[str] = set()
    for training_key, splits in merge_groups.items():
        if training_key not in state:
            continue
        tensor = state[training_key]
        splits.sort(key=lambda entry: entry[1])
        total = splits[0][2]
        if any(split[2] != total for split in splits):
            raise ValueError(f"Inconsistent merge totals for {training_key!r}")
        indices = [split[1] for split in splits]
        if len(splits) != total or indices != list(range(total)):
            raise ValueError(
                f"Incomplete reverse merge mapping for {training_key!r}: "
                f"indices={indices}, total={total}"
            )

        recorded_sizes = [split[3] for split in splits]
        if all(size is None for size in recorded_sizes):
            if tensor.shape[0] % total:
                raise ValueError(
                    f"Cannot evenly split legacy merged parameter {training_key!r} "
                    f"with output size {tensor.shape[0]} into {total} parts"
                )
            split_sizes = [tensor.shape[0] // total] * total
        elif any(size is None for size in recorded_sizes):
            raise ValueError(f"Partially specified split sizes for {training_key!r}")
        else:
            split_sizes = [int(size) for size in recorded_sizes if size is not None]
        if sum(split_sizes) != tensor.shape[0]:
            raise ValueError(
                f"Recorded split sizes for {training_key!r} sum to "
                f"{sum(split_sizes)}, expected {tensor.shape[0]}"
            )

        split_tensors = torch.split(tensor, split_sizes, dim=0)
        for (source_key, _, _, _), split_tensor in zip(
            splits,
            split_tensors,
            strict=True,
        ):
            converted[source_key] = split_tensor
        used_keys.add(training_key)

    for training_key, value in state.items():
        if training_key in used_keys:
            continue
        if training_key not in reverse_param_names_mapping:
            converted[training_key] = value
            continue
        entries = _entries(reverse_param_names_mapping[training_key])
        if len(entries) != 1:
            raise ValueError(
                f"Invalid direct reverse mapping for {training_key!r}: {entries!r}"
            )
        source_key, merge_index, _, _ = _unpack(entries[0])
        if merge_index is None:
            converted[source_key] = value

    return converted
