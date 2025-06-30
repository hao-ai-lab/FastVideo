# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, Tuple, Union

import torch

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_param_names_mapping(
        mapping_dict: Dict[str, str]) -> Callable[[str], tuple[str, Any, Any]]:
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


def hf_to_custom_param_sd(
    hf_param_sd: Union[Dict[str, torch.Tensor], Iterator[Tuple[str,
                                                               torch.Tensor]]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]]
) -> Dict[str, torch.Tensor]:
    """
    Converts a Hugging Face parameter state dictionary to a custom parameter state dictionary.
    
    Args:
        hf_param_sd (Dict[str, torch.Tensor]): The Hugging Face parameter state dictionary
        param_names_mapping (Callable[[str], tuple[str, Any, Any]]): A function that maps parameter names from source to target format
        
    Returns:
        Dict[str, torch.Tensor]: The custom parameter state dictionary
    """
    custom_param_sd = {}
    to_merge_params = defaultdict(dict)
    reverse_param_names_mapping = {}
    if isinstance(hf_param_sd, Dict):
        hf_param_sd = hf_param_sd.items()
    for source_param_name, full_tensor in hf_param_sd:
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name)
        reverse_param_names_mapping[target_param_name] = (source_param_name,
                                                          merge_index,
                                                          num_params_to_merge)
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                # cat at output dim according to the merge_index order
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue
        custom_param_sd[target_param_name] = full_tensor
    return custom_param_sd, reverse_param_names_mapping
