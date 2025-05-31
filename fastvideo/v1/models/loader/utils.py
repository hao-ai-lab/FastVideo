# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib
import re
from typing import Callable, Dict

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
        mapping_dict: Dict[str, str]) -> Callable[[str], str]:
    """
    Creates a mapping function that transforms parameter names using regex patterns.
    
    Args:
        mapping_dict (Dict[str, str]): Dictionary mapping regex patterns to replacement patterns
        param_name (str): The parameter name to be transformed
        
    Returns:
        Callable[[str], str]: A function that maps parameter names from source to target format
    """

    def mapping_fn(name: str) -> str:
        # Try to match and transform the name using the regex patterns in mapping_dict
        for pattern, replacement in mapping_dict.items():
            match = re.match(pattern, name)
            if match:
                name = re.sub(pattern, replacement, name)
                return name

        # If no pattern matches, return the original name
        return name

    return mapping_fn
