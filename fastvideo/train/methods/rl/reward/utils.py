# SPDX-License-Identifier: Apache-2.0
"""Utility functions for reward computation."""

from __future__ import annotations

import numpy as np
import torch


def prepare_images(
    images: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert tensor images to uint8 numpy (NHWC or NFHWC).

    Accepts:
        - (N, C, H, W) or (N, H, W, C) tensors/arrays
        - (N, F, C, H, W) or (N, F, H, W, C) video tensors
    Returns:
        uint8 numpy array in HWC/FHWC layout.
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)

    if images.ndim == 4:
        # Image batch: (N, C, H, W) or (N, H, W, C)
        if images.shape[1] in (1, 3):
            images = images.transpose(0, 2, 3, 1)
    elif images.ndim == 5:
        # Video batch: (N, F, C, H, W) or (N, C, F, H, W)
        if images.shape[2] in (1, 3):
            # (N, F, C, H, W) -> (N, F, H, W, C)
            images = images.transpose(0, 1, 3, 4, 2)
        elif images.shape[1] in (1, 3):
            # (N, C, F, H, W) -> (N, F, H, W, C)
            images = images.transpose(0, 2, 3, 4, 1)

    if images.dtype == np.float32 or images.dtype == np.float64:
        images = np.clip(images * 255, 0, 255).astype(
            np.uint8
        )
    return images
