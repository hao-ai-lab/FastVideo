# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def get_param_names_mapping(mapping_dict: dict[str, str]) -> Callable[[str], tuple[str, Any, Any]]:
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


def iter_hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
    reverse_param_names_mapping: dict[str, tuple[str, Any, Any]] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Streaming form of ``hf_to_custom_state_dict``: yields ``(target_param_name, tensor)`` as soon as a parameter is
    complete, buffering only the pieces of a fused parameter until all of them arrived. The source tensor is released
    before the next one is pulled, so the whole checkpoint is never resident at once.

    Args:
        hf_param_sd: The Hugging Face parameter state dictionary or an iterator over its items
        param_names_mapping: A function that maps parameter names from source to target format
        reverse_param_names_mapping: Optional dict filled in place with the custom -> hf mapping; complete once the
            iterator is exhausted
    """
    to_merge_params = defaultdict(dict)  # type: ignore
    if reverse_param_names_mapping is None:
        reverse_param_names_mapping = {}
    if isinstance(hf_param_sd, dict):
        hf_param_sd = hf_param_sd.items()  # type: ignore
    for source_param_name, full_tensor in hf_param_sd:  # type: ignore
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(source_param_name)
        reverse_param_names_mapping[target_param_name] = (source_param_name, merge_index, num_params_to_merge)
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) != num_params_to_merge:
                continue
            # cat at output dim according to the merge_index order
            pieces = to_merge_params.pop(target_param_name)
            full_tensor = torch.cat([pieces[i] for i in range(num_params_to_merge)], dim=0)
            del pieces
        yield target_param_name, full_tensor
        del full_tensor


def hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]]
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, Any, Any]]]:
    """
    Converts a Hugging Face parameter state dictionary to a custom parameter state dictionary.

    Args:
        hf_param_sd (Dict[str, torch.Tensor]): The Hugging Face parameter state dictionary
        param_names_mapping (Callable[[str], tuple[str, Any, Any]]): A function that maps parameter names from source to target format

    Returns:
        custom_param_sd (Dict[str, torch.Tensor]): The custom formatted parameter state dict
        reverse_param_names_mapping (Dict[str, Tuple[str, Any, Any]]): Maps back from custom to hf
    """
    reverse_param_names_mapping: dict[str, tuple[str, Any, Any]] = {}
    custom_param_sd = dict(iter_hf_to_custom_state_dict(hf_param_sd, param_names_mapping, reverse_param_names_mapping))
    return custom_param_sd, reverse_param_names_mapping
