# SPDX-License-Identifier: Apache-2.0
"""
Minimal image processing utilities for FastVideo.
This module provides lightweight image preprocessing without external dependencies beyond PyTorch/NumPy/PIL.
"""

import numpy as np
import PIL.Image
import torch


class ImageProcessor:
    """
    Minimal image processor for video frame preprocessing.

    This is a lightweight alternative to diffusers.VideoProcessor that handles:
    - PIL image to tensor conversion
    - Resizing to specified dimensions
    - Normalization to [-1, 1] range

    Args:
        vae_scale_factor: The VAE scale factor used to ensure dimensions are multiples of this value.
    """

    def __init__(self, vae_scale_factor: int = 8) -> None:
        self.vae_scale_factor = vae_scale_factor

    def preprocess(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """
        Preprocess an image to a normalized torch tensor.

        Args:
            image: Input image (PIL Image, NumPy array, or torch tensor)
            height: Target height. If None, uses image's original height.
            width: Target width. If None, uses image's original width.

        Returns:
            torch.Tensor: Normalized tensor of shape (1, 3, height, width) or (1, 1, height, width) for grayscale,
                         with values in range [-1, 1].
        """
        # Handle different input types
        if isinstance(image, PIL.Image.Image):
            return self._preprocess_pil(image, height, width)
        elif isinstance(image, np.ndarray):
            return self._preprocess_numpy(image, height, width)
        elif isinstance(image, torch.Tensor):
            return self._preprocess_tensor(image, height, width)
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Supported types: PIL.Image.Image, np.ndarray, torch.Tensor")

    def _preprocess_pil(
        self,
        image: PIL.Image.Image,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Preprocess a PIL image."""
        if height is None:
            height = image.height
        if width is None:
            width = image.width

        height = height - (height % self.vae_scale_factor)
        width = width - (width % self.vae_scale_factor)

        image = image.resize((width, height),
                             resample=PIL.Image.Resampling.LANCZOS)

        image_np = np.array(image, dtype=np.float32) / 255.0

        if image_np.ndim == 2:  # Grayscale
            image_np = np.expand_dims(image_np, axis=-1)

        return self._normalize_to_tensor(image_np)

    def _preprocess_numpy(
        self,
        image: np.ndarray,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Preprocess a numpy array."""
        # Determine target dimensions if not provided
        if image.ndim == 3:
            img_height, img_width = image.shape[:2]
        elif image.ndim == 2:
            img_height, img_width = image.shape
        else:
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

        if height is None:
            height = img_height
        if width is None:
            width = img_width

        height = height - (height % self.vae_scale_factor)
        width = width - (width % self.vae_scale_factor)

        if image.dtype == np.uint8:
            pil_image = PIL.Image.fromarray(image)
        else:
            # Assume normalized [0, 1] or similar
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            pil_image = PIL.Image.fromarray(image_uint8)

        pil_image = pil_image.resize((width, height),
                                     resample=PIL.Image.Resampling.LANCZOS)
        image_np = np.array(pil_image, dtype=np.float32) / 255.0

        # Ensure 3D shape
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=-1)

        return self._normalize_to_tensor(image_np)

    def _preprocess_tensor(
        self,
        image: torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Preprocess a torch tensor."""
        # Determine target dimensions
        if image.ndim == 3:  # (H, W, C) or (C, H, W)
            if image.shape[0] in (1, 3, 4):  # Likely (C, H, W)
                img_height, img_width = image.shape[1], image.shape[2]
            else:  # Likely (H, W, C)
                img_height, img_width = image.shape[0], image.shape[1]
        elif image.ndim == 2:  # (H, W)
            img_height, img_width = image.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.ndim}D")

        if height is None:
            height = img_height
        if width is None:
            width = img_width

        height = height - (height % self.vae_scale_factor)
        width = width - (width % self.vae_scale_factor)

        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.ndim == 3:
            if image.shape[0] in (1, 3, 4):  # (C, H, W)
                image = image.unsqueeze(0)  # (1, C, H, W)
            else:  # (H, W, C) - need to rearrange
                image = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        image = torch.nn.functional.interpolate(image,
                                                size=(height, width),
                                                mode="bilinear",
                                                align_corners=False)

        if image.max() > 1.0:  # Assume [0, 255] range
            image = image / 255.0

        image = 2.0 * image - 1.0

        return image

    def _normalize_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Convert normalized numpy array [0, 1] to torch tensor [-1, 1].

        Args:
            image_np: NumPy array with shape (H, W) or (H, W, C) with values in [0, 1]

        Returns:
            torch.Tensor: Shape (1, C, H, W) or (1, 1, H, W) with values in [-1, 1]
        """
        # Convert to tensor
        if image_np.ndim == 2:  # (H, W) - grayscale
            tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(
                0)  # (1, 1, H, W)
        elif image_np.ndim == 3:  # (H, W, C)
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(
                0)  # (1, C, H, W)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {image_np.ndim}D")

        # Normalize to [-1, 1]
        tensor = 2.0 * tensor - 1.0

        return tensor
