# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib

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
