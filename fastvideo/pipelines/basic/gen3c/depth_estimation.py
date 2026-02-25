# SPDX-License-Identifier: Apache-2.0
# Ported from NVIDIA GEN3C: cosmos_predict1/diffusion/inference/gen3c_single_image.py
"""MoGe-based monocular depth estimation for GEN3C 3D cache conditioning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def load_moge_model(
    model_name: str = "Ruicheng/moge-vitl",
    device: str | torch.device = "cuda",
) -> "MoGeModel":
    """Load MoGe depth estimation model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to load model on.

    Returns:
        Loaded MoGe model.
    """
    try:
        from moge.model.v1 import MoGeModel
    except ImportError:
        raise ImportError(
            "MoGe is required for GEN3C 3D cache conditioning. "
            "Install it with: pip install moge-pytorch")

    logger.info("Loading MoGe depth model: %s", model_name)
    model = MoGeModel.from_pretrained(model_name).to(device)
    model.eval()
    logger.info("MoGe model loaded successfully")
    return model


def predict_depth_from_path(
    image_path: str,
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: "MoGeModel",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Predict depth, intrinsics, and mask from an image file path.

    Args:
        image_path: Path to input image (RGB or BGR, any format cv2 supports).
        target_h: Target height for output tensors.
        target_w: Target width for output tensors.
        device: Computation device.
        moge_model: Loaded MoGe model.

    Returns:
        image: (1, 1, 3, target_h, target_w) image tensor in [-1, 1].
        depth: (1, 1, 1, target_h, target_w) depth map.
        mask: (1, 1, 1, target_h, target_w) confidence mask.
        w2c: (1, 1, 4, 4) world-to-camera matrix (identity).
        intrinsics: (1, 1, 3, 3) camera intrinsics.
    """
    import cv2

    input_image_bgr = cv2.imread(image_path)
    if input_image_bgr is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")
    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)

    return _predict_depth_core(input_image_rgb, target_h, target_w, device,
                               moge_model)


def predict_depth_from_tensor(
    image_tensor: torch.Tensor,
    moge_model: "MoGeModel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predict depth and mask from an image tensor (for autoregressive generation).

    Args:
        image_tensor: (C, H, W) image tensor in [0, 1] range.
        moge_model: Loaded MoGe model.

    Returns:
        depth: (1, 1, H, W) depth map.
        mask: (1, 1, H, W) confidence mask.
    """
    moge_output = moge_model.infer(image_tensor)
    depth = moge_output["depth"]
    mask = moge_output["mask"]

    depth = depth.unsqueeze(0).unsqueeze(0)
    depth = torch.nan_to_num(depth, nan=1e4)
    depth = torch.clamp(depth, min=0, max=1e4)

    mask = mask.unsqueeze(0).unsqueeze(0)
    depth = torch.where(mask == 0,
                        torch.tensor(1000.0, device=depth.device), depth)

    return depth, mask


def _predict_depth_core(
    input_image_rgb: np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: "MoGeModel",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Core depth prediction logic shared between path and tensor inputs."""
    import cv2

    # MoGe runs at fixed resolution for best results
    depth_pred_h, depth_pred_w = 720, 1280

    resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    img_tensor = torch.tensor(resized / 255.0,
                              dtype=torch.float32,
                              device=device).permute(2, 0, 1)

    # Run MoGe inference
    moge_output = moge_model.infer(img_tensor)
    depth_hw = moge_output["depth"]
    intrinsics_norm = moge_output["intrinsics"]
    mask_hw = moge_output["mask"]

    # Replace invalid depth with large value
    depth_hw = torch.where(
        mask_hw == 0, torch.tensor(1000.0, device=depth_hw.device), depth_hw)

    # Convert normalized intrinsics to pixel coordinates
    intrinsics_pixel = intrinsics_norm.clone()
    intrinsics_pixel[0, 0] *= depth_pred_w  # fx
    intrinsics_pixel[1, 1] *= depth_pred_h  # fy
    intrinsics_pixel[0, 2] *= depth_pred_w  # cx
    intrinsics_pixel[1, 2] *= depth_pred_h  # cy

    # Scale to target resolution
    h_scale = target_h / depth_pred_h
    w_scale = target_w / depth_pred_w

    depth_target = F.interpolate(depth_hw.unsqueeze(0).unsqueeze(0),
                                 size=(target_h, target_w),
                                 mode='bilinear',
                                 align_corners=False).squeeze(0).squeeze(0)

    mask_target = F.interpolate(mask_hw.unsqueeze(0).unsqueeze(0).to(
        torch.float32),
                                size=(target_h, target_w),
                                mode='nearest').squeeze(0).squeeze(0).to(
                                    torch.bool)

    img_target = F.interpolate(img_tensor.unsqueeze(0),
                               size=(target_h, target_w),
                               mode='bilinear',
                               align_corners=False).squeeze(0)

    # Scale intrinsics for target resolution
    intrinsics_target = intrinsics_pixel.clone()
    intrinsics_target[0, 0] *= w_scale  # fx
    intrinsics_target[0, 2] *= w_scale  # cx
    intrinsics_target[1, 1] *= h_scale  # fy
    intrinsics_target[1, 2] *= h_scale  # cy

    # Format outputs with batch and frame dimensions: (B, F, ...)
    # Image: [-1, 1] range
    image_out = (img_target * 2 - 1).unsqueeze(0).unsqueeze(1)

    depth_out = depth_target.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    depth_out = torch.nan_to_num(depth_out, nan=1e4)
    depth_out = torch.clamp(depth_out, min=0, max=1e4)

    mask_out = mask_target.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    w2c_out = torch.eye(4, dtype=torch.float32,
                        device=device).unsqueeze(0).unsqueeze(0)
    intrinsics_out = intrinsics_target.unsqueeze(0).unsqueeze(0)

    return image_out, depth_out, mask_out, w2c_out, intrinsics_out
