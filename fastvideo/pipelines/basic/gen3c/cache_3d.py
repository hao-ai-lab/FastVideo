# SPDX-License-Identifier: Apache-2.0
"""
This module implements the 3D cache system for GEN3C video generation with camera control.
The cache maintains a point cloud representation of the scene, enabling:
- Unprojecting depth maps to 3D world points
- Forward warping rendered views to new camera poses
- Managing multiple frame buffers for temporal consistency
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def inverse_with_conversion(mtx: torch.Tensor) -> torch.Tensor:
    """Compute matrix inverse with float32 conversion for numerical stability."""
    return torch.linalg.inv(mtx.to(torch.float32)).to(mtx.dtype)


def create_grid(b: int,
                h: int,
                w: int,
                device: str = "cpu",
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create a dense grid of (x, y) coordinates of shape (b, 2, h, w).
    
    Args:
        b: Batch size
        h: Height
        w: Width
        device: Device for tensor creation
        dtype: Data type for tensor
        
    Returns:
        Grid tensor of shape (b, 2, h, w)
    """
    x = torch.arange(0, w, device=device,
                     dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device,
                     dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, y], dim=1)


def unproject_points(
    depth: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    is_depth: bool = True,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Unproject depth map to 3D world points.
    
    Args:
        depth: (b, 1, h, w) depth map
        w2c: (b, 4, 4) world-to-camera transformation matrix
        intrinsic: (b, 3, 3) camera intrinsic matrix
        is_depth: If True, depth is z-depth; if False, depth is distance to camera
        mask: Optional (b, h, w) or (b, 1, h, w) mask for valid pixels
        
    Returns:
        world_points: (b, h, w, 3) 3D world coordinates
    """
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    if mask is None:
        mask = depth > 0
    if mask.dim() == depth.dim() and mask.shape[1] == 1:
        mask = mask[:, 0]

    idx = torch.nonzero(mask)
    if idx.numel() == 0:
        return torch.zeros((b, h, w, 3), device=device, dtype=dtype)

    b_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]

    intrinsic_inv = inverse_with_conversion(intrinsic)  # (b, 3, 3)

    x_valid = x_idx.to(dtype)
    y_valid = y_idx.to(dtype)
    ones = torch.ones_like(x_valid)
    pos = torch.stack([x_valid, y_valid, ones],
                      dim=1).unsqueeze(-1)  # (N, 3, 1)

    intrinsic_inv_valid = intrinsic_inv[b_idx]  # (N, 3, 3)
    unnormalized_pos = torch.matmul(intrinsic_inv_valid, pos)  # (N, 3, 1)

    depth_valid = depth[b_idx, 0, y_idx, x_idx].view(-1, 1, 1)
    if is_depth:
        world_points_cam = depth_valid * unnormalized_pos
    else:
        norm_val = torch.norm(unnormalized_pos, dim=1, keepdim=True)
        direction = unnormalized_pos / (norm_val + 1e-8)
        world_points_cam = depth_valid * direction

    ones_h = torch.ones((world_points_cam.shape[0], 1, 1),
                        device=device,
                        dtype=dtype)
    world_points_homo = torch.cat([world_points_cam, ones_h],
                                  dim=1)  # (N, 4, 1)

    trans = inverse_with_conversion(w2c)  # (b, 4, 4)
    trans_valid = trans[b_idx]  # (N, 4, 4)
    world_points_transformed = torch.matmul(trans_valid,
                                            world_points_homo)  # (N, 4, 1)
    sparse_points = world_points_transformed[:, :3, 0]  # (N, 3)

    out_points = torch.zeros((b, h, w, 3), device=device, dtype=dtype)
    out_points[b_idx, y_idx, x_idx, :] = sparse_points
    return out_points


def project_points(
    world_points: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """
    Project 3D world points to 2D pixel coordinates.
    
    Args:
        world_points: (b, h, w, 3) 3D world coordinates
        w2c: (b, 4, 4) world-to-camera transformation matrix
        intrinsic: (b, 3, 3) camera intrinsic matrix
        
    Returns:
        projected_points: (b, h, w, 3, 1) projected 2D coordinates (x, y, z)
    """
    world_points = world_points.unsqueeze(-1)  # (b, h, w, 3, 1)
    b, h, w, _, _ = world_points.shape

    ones_4d = torch.ones((b, h, w, 1, 1),
                         device=world_points.device,
                         dtype=world_points.dtype)
    world_points_homo = torch.cat([world_points, ones_4d],
                                  dim=3)  # (b, h, w, 4, 1)

    trans_4d = w2c[:, None, None]  # (b, 1, 1, 4, 4)
    camera_points_homo = torch.matmul(trans_4d,
                                      world_points_homo)  # (b, h, w, 4, 1)

    camera_points = camera_points_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    intrinsic_4d = intrinsic[:, None, None]  # (b, 1, 1, 3, 3)
    projected_points = torch.matmul(intrinsic_4d,
                                    camera_points)  # (b, h, w, 3, 1)

    return projected_points


def bilinear_splatting(
    frame1: torch.Tensor,
    mask1: torch.Tensor | None,
    depth1: torch.Tensor,
    flow12: torch.Tensor,
    flow12_mask: torch.Tensor | None = None,
    is_image: bool = False,
    depth_weight_scale: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear splatting for forward warping.
    
    Args:
        frame1: (b, c, h, w) source frame
        mask1: (b, 1, h, w) valid pixel mask (1 for known, 0 for unknown)
        depth1: (b, 1, h, w) depth map
        flow12: (b, 2, h, w) optical flow from frame1 to frame2
        flow12_mask: (b, 1, h, w) flow validity mask
        is_image: If True, output will be clipped to (-1, 1) range
        depth_weight_scale: Scale factor for depth weighting
        
    Returns:
        warped_frame2: (b, c, h, w) warped frame
        mask2: (b, 1, h, w) validity mask for warped frame
    """
    b, c, h, w = frame1.shape
    device = frame1.device
    dtype = frame1.dtype

    if mask1 is None:
        mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)
    if flow12_mask is None:
        flow12_mask = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)

    grid = create_grid(b, h, w, device=device, dtype=dtype)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()

    trans_pos_offset = torch.stack([
        torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)
    ],
                                   dim=1)
    trans_pos_floor = torch.stack([
        torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)
    ],
                                  dim=1)
    trans_pos_ceil = torch.stack([
        torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
        torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)
    ],
                                 dim=1)

    # Bilinear weights
    prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                     (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                     (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                     (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
    prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                     (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

    # Depth weighting for occlusion handling
    clamped_depth1 = torch.clamp(depth1, min=0)
    log_depth1 = torch.log1p(clamped_depth1)
    exponent = log_depth1 / (log_depth1.max() + 1e-7) * depth_weight_scale
    max_exponent = 80.0 if dtype in [torch.float32, torch.bfloat16] else 10.0
    clamped_exponent = torch.clamp(exponent, max=max_exponent)
    depth_weights = torch.exp(clamped_exponent) + 1e-7

    weight_nw = torch.moveaxis(
        prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3],
        [0, 3, 1, 2])
    weight_sw = torch.moveaxis(
        prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3],
        [0, 3, 1, 2])
    weight_ne = torch.moveaxis(
        prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3],
        [0, 3, 1, 2])
    weight_se = torch.moveaxis(
        prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3],
        [0, 3, 1, 2])

    warped_frame = torch.zeros(size=(b, h + 2, w + 2, c),
                               dtype=dtype,
                               device=device)
    warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1),
                                 dtype=dtype,
                                 device=device)

    frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
    batch_indices = torch.arange(b, device=device, dtype=torch.long)[:, None,
                                                                     None]

    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
        frame1_cl * weight_nw,
        accumulate=True)
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
        frame1_cl * weight_sw,
        accumulate=True)
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
        frame1_cl * weight_ne,
        accumulate=True)
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
        frame1_cl * weight_se,
        accumulate=True)

    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
        weight_nw,
        accumulate=True)
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
        weight_sw,
        accumulate=True)
    warped_weights.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
        weight_ne,
        accumulate=True)
    warped_weights.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
        weight_se,
        accumulate=True)

    warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3],
                                       [0, 2, 3, 1])
    cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
    cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]
    cropped_weights = torch.nan_to_num(cropped_weights, nan=1000.0)

    mask = cropped_weights > 0
    zero_value = -1 if is_image else 0
    zero_tensor = torch.tensor(zero_value,
                               dtype=frame1.dtype,
                               device=frame1.device)
    warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights,
                                zero_tensor)
    mask2 = mask.to(frame1)

    if is_image:
        warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)

    return warped_frame2, mask2


def forward_warp(
    frame1: torch.Tensor,
    mask1: torch.Tensor | None,
    depth1: torch.Tensor | None,
    transformation1: torch.Tensor | None,
    transformation2: torch.Tensor,
    intrinsic1: torch.Tensor | None,
    intrinsic2: torch.Tensor | None,
    is_image: bool = True,
    is_depth: bool = True,
    render_depth: bool = False,
    world_points1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Forward warp frame1 to a new view defined by transformation2.
    
    Args:
        frame1: (b, c, h, w) source frame in range [-1, 1] for images
        mask1: (b, 1, h, w) valid pixel mask
        depth1: (b, 1, h, w) depth map (required if world_points1 is None)
        transformation1: (b, 4, 4) source camera w2c (required if depth1 is provided)
        transformation2: (b, 4, 4) target camera w2c
        intrinsic1: (b, 3, 3) source camera intrinsics
        intrinsic2: (b, 3, 3) target camera intrinsics
        is_image: If True, output will be clipped to (-1, 1)
        is_depth: If True, depth1 is z-depth; if False, it's distance
        render_depth: If True, also return the warped depth map
        world_points1: (b, h, w, 3) pre-computed world points (alternative to depth1)
        
    Returns:
        warped_frame2: (b, c, h, w) warped frame
        mask2: (b, 1, h, w) validity mask
        warped_depth2: (b, h, w) warped depth (if render_depth=True)
        flow12: (b, 2, h, w) optical flow
    """
    device = frame1.device
    b, c, h, w = frame1.shape
    dtype = frame1.dtype

    if mask1 is None:
        mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)
    if intrinsic2 is None:
        assert intrinsic1 is not None
        intrinsic2 = intrinsic1.clone()

    if world_points1 is not None:
        # Use pre-computed world points
        assert world_points1.shape == (b, h, w, 3)
        trans_points1 = project_points(world_points1, transformation2,
                                       intrinsic2)
    else:
        # Compute from depth
        assert depth1 is not None and transformation1 is not None
        assert depth1.shape == (b, 1, h, w)

        depth1 = torch.nan_to_num(depth1, nan=1e4)
        depth1 = torch.clamp(depth1, min=0, max=1e4)

        # Unproject to world, then project to target view
        world_points1 = unproject_points(depth1,
                                         transformation1,
                                         intrinsic1,
                                         is_depth=is_depth)
        trans_points1 = project_points(world_points1, transformation2,
                                       intrinsic2)

    # Filter points behind camera
    mask1 = mask1 * (trans_points1[:, :, :, 2, 0].unsqueeze(1) > 0)
    trans_coordinates = trans_points1[:, :, :, :2, 0] / (
        trans_points1[:, :, :, 2:3, 0] + 1e-7)
    trans_coordinates = trans_coordinates.permute(0, 3, 1, 2)  # b, 2, h, w
    trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)

    grid = create_grid(b, h, w, device=device, dtype=dtype)
    flow12 = trans_coordinates - grid

    warped_frame2, mask2 = bilinear_splatting(frame1,
                                              mask1,
                                              trans_depth1,
                                              flow12,
                                              None,
                                              is_image=is_image)

    warped_depth2 = None
    if render_depth:
        warped_depth2 = bilinear_splatting(trans_depth1,
                                           mask1,
                                           trans_depth1,
                                           flow12,
                                           None,
                                           is_image=False)[0][:, 0]

    return warped_frame2, mask2, warped_depth2, flow12


def reliable_depth_mask_range_batch(
    depth: torch.Tensor,
    window_size: int = 5,
    ratio_thresh: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a mask for reliable depth values based on local variation.
    
    Args:
        depth: (b, h, w) or (b, 1, h, w) depth map
        window_size: Size of the local window (must be odd)
        ratio_thresh: Threshold for depth variation ratio
        eps: Small epsilon for numerical stability
        
    Returns:
        reliable_mask: Boolean mask where True indicates reliable depth
    """
    assert window_size % 2 == 1, "Window size must be odd."

    if depth.dim() == 3:
        depth_unsq = depth.unsqueeze(1)
    elif depth.dim() == 4:
        depth_unsq = depth
    else:
        raise ValueError(
            "depth tensor must be of shape (b, h, w) or (b, 1, h, w)")

    local_max = F.max_pool2d(depth_unsq,
                             kernel_size=window_size,
                             stride=1,
                             padding=window_size // 2)
    local_min = -F.max_pool2d(-depth_unsq,
                              kernel_size=window_size,
                              stride=1,
                              padding=window_size // 2)
    local_mean = F.avg_pool2d(depth_unsq,
                              kernel_size=window_size,
                              stride=1,
                              padding=window_size // 2)

    ratio = (local_max - local_min) / (local_mean + eps)
    reliable_mask = (ratio < ratio_thresh) & (depth_unsq > 0)

    return reliable_mask


class Cache3DBase:
    """
    Base class for 3D cache management.
    
    The cache maintains:
    - input_image: RGB images stored in the cache
    - input_points: 3D world coordinates for each pixel
    - input_mask: Validity mask for each pixel
    """

    def __init__(
        self,
        input_image: torch.Tensor,
        input_depth: torch.Tensor,
        input_w2c: torch.Tensor,
        input_intrinsics: torch.Tensor,
        input_mask: torch.Tensor | None = None,
        input_format: list[str] | None = None,
        input_points: torch.Tensor | None = None,
        weight_dtype: torch.dtype = torch.float32,
        is_depth: bool = True,
        device: str = "cuda",
        filter_points_threshold: float = 1.0,
    ):
        """
        Initialize the 3D cache.
        
        Args:
            input_image: Input image tensor with varying dimensions
            input_depth: Depth map tensor
            input_w2c: World-to-camera transformation matrix
            input_intrinsics: Camera intrinsic matrix
            input_mask: Optional validity mask
            input_format: Dimension labels for input_image (e.g., ['B', 'C', 'H', 'W'])
            input_points: Pre-computed 3D world points (alternative to depth)
            weight_dtype: Data type for computations
            is_depth: If True, input_depth is z-depth; if False, it's distance
            device: Computation device
            filter_points_threshold: Threshold for filtering unreliable depth
        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold

        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        # Map dimension names to indices
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape

        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask],
                                    dim=format_to_indices.get("C"))

        # Extract dimensions
        B = input_shape[format_to_indices.get(
            "B", 0)] if "B" in format_to_indices else 1
        F = input_shape[format_to_indices.get(
            "F", 0)] if "F" in format_to_indices else 1
        N = input_shape[format_to_indices.get(
            "N", 0)] if "N" in format_to_indices else 1
        V = input_shape[format_to_indices.get(
            "V", 0)] if "V" in format_to_indices else 1
        H = input_shape[format_to_indices.get(
            "H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get(
            "W", 0)] if "W" in format_to_indices else None

        # Reorder dimensions to B x F x N x V x C x H x W
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]
        permute_order: list[int | None] = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            permute_order.append(idx)

        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :
                                                            3], input_image[:, :, :, :,
                                                                            3:]
            self.input_mask = self.input_mask.to("cpu")
        else:
            self.input_mask = None
            self.input_image = input_image
        self.input_image = self.input_image.to(weight_dtype).to("cpu")

        # Compute 3D world points
        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W,
                                                     3).to("cpu")
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)

            self.input_points = (unproject_points(
                input_depth.reshape(-1, 1, H, W),
                input_w2c.reshape(-1, 4, 4),
                input_intrinsics.reshape(-1, 3, 3),
                is_depth=self.is_depth,
            ).to(weight_dtype).reshape(B, F, N, V, H, W, 3).to("cpu"))
            self.input_depth = input_depth

        # Filter unreliable depth
        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(
                input_depth, ratio_thresh=self.filter_points_threshold).reshape(
                    B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask.to("cpu")
            else:
                self.input_mask = self.input_mask * depth_mask.to(
                    self.input_mask.device)

    def update_cache(self, **kwargs):
        """Update the cache with new frames. To be implemented by subclasses."""
        raise NotImplementedError

    def input_frame_count(self) -> int:
        """Return the number of frames in the cache."""
        return self.input_image.shape[1]

    def render_cache(
        self,
        target_w2cs: torch.Tensor,
        target_intrinsics: torch.Tensor,
        render_depth: bool = False,
        start_frame_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render the cached 3D points from new camera viewpoints.
        
        Args:
            target_w2cs: (b, F_target, 4, 4) target camera transformations
            target_intrinsics: (b, F_target, 3, 3) target camera intrinsics
            render_depth: If True, return depth instead of RGB
            start_frame_idx: Starting frame index in the cache
            
        Returns:
            pixels: (b, F_target, N, c, h, w) rendered images or depth
            masks: (b, F_target, N, 1, h, w) validity masks
        """
        bs, F_target, _, _ = target_w2cs.shape
        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B

        target_w2cs = target_w2cs.reshape(B, F_target, 1, 4,
                                          4).expand(B, F_target, N, 4,
                                                    4).reshape(-1, 4, 4)
        target_intrinsics = target_intrinsics.reshape(B, F_target, 1, 3,
                                                      3).expand(
                                                          B, F_target, N, 3,
                                                          3).reshape(-1, 3, 3)

        # Prepare inputs
        first_images = rearrange(
            self.input_image[:,
                             start_frame_idx:start_frame_idx + F_target].expand(
                                 B, F_target, N, V, C, H, W),
            "B F N V C H W -> (B F N) V C H W")
        first_points = rearrange(
            self.input_points[:, start_frame_idx:start_frame_idx +
                              F_target].expand(B, F_target, N, V, H, W, 3),
            "B F N V H W C -> (B F N) V H W C")
        first_masks = rearrange(
            self.input_mask[:, start_frame_idx:start_frame_idx +
                            F_target].expand(B, F_target, N, V, 1, H, W),
            "B F N V C H W -> (B F N) V C H W"
        ) if self.input_mask is not None else None

        # Process in chunks for memory efficiency
        if first_images.shape[1] == 1:
            warp_chunk_size = 2
            rendered_warp_images = []
            rendered_warp_masks = []
            rendered_warp_depth = []

            first_images = first_images.squeeze(1)
            first_points = first_points.squeeze(1)
            first_masks = first_masks.squeeze(
                1) if first_masks is not None else None

            for i in range(0, first_images.shape[0], warp_chunk_size):
                with torch.no_grad():
                    imgs_chunk = first_images[i:i + warp_chunk_size].to(
                        self.device, non_blocking=True)
                    pts_chunk = first_points[i:i + warp_chunk_size].to(
                        self.device, non_blocking=True)
                    masks_chunk = (first_masks[i:i + warp_chunk_size].to(
                        self.device, non_blocking=True)
                                   if first_masks is not None else None)

                    (
                        rendered_warp_images_chunk,
                        rendered_warp_masks_chunk,
                        rendered_warp_depth_chunk,
                        _,
                    ) = forward_warp(
                        imgs_chunk,
                        mask1=masks_chunk,
                        depth1=None,
                        transformation1=None,
                        transformation2=target_w2cs[i:i + warp_chunk_size],
                        intrinsic1=target_intrinsics[i:i + warp_chunk_size],
                        intrinsic2=target_intrinsics[i:i + warp_chunk_size],
                        render_depth=render_depth,
                        world_points1=pts_chunk,
                    )

                    rendered_warp_images.append(
                        rendered_warp_images_chunk.to("cpu"))
                    rendered_warp_masks.append(
                        rendered_warp_masks_chunk.to("cpu"))
                    if render_depth:
                        rendered_warp_depth.append(
                            rendered_warp_depth_chunk.to("cpu"))

                    del imgs_chunk, pts_chunk, masks_chunk
                    torch.cuda.empty_cache()

            rendered_warp_images = torch.cat(rendered_warp_images, dim=0)
            rendered_warp_masks = torch.cat(rendered_warp_masks, dim=0)
            if render_depth:
                rendered_warp_depth = torch.cat(rendered_warp_depth, dim=0)
        else:
            raise NotImplementedError("Multi-view rendering not yet supported")

        pixels = rearrange(rendered_warp_images,
                           "(b f n) c h w -> b f n c h w",
                           b=bs,
                           f=F_target,
                           n=N)
        masks = rearrange(rendered_warp_masks,
                          "(b f n) c h w -> b f n c h w",
                          b=bs,
                          f=F_target,
                          n=N)

        if render_depth:
            pixels = rearrange(rendered_warp_depth,
                               "(b f n) h w -> b f n h w",
                               b=bs,
                               f=F_target,
                               n=N)

        return pixels.to(self.device), masks.to(self.device)


class Cache3DBuffer(Cache3DBase):
    """
    3D cache with frame buffer support.
    
    This class manages multiple frame buffers for temporal consistency
    and supports noise augmentation for training stability.
    """

    def __init__(
        self,
        frame_buffer_max: int = 2,
        noise_aug_strength: float = 0.0,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        """
        Initialize the buffered 3D cache.
        
        Args:
            frame_buffer_max: Maximum number of frames to buffer
            noise_aug_strength: Strength of noise augmentation per buffer
            generator: Random generator for reproducibility
            **kwargs: Arguments passed to Cache3DBase
        """
        super().__init__(**kwargs)
        self.frame_buffer_max = frame_buffer_max
        self.noise_aug_strength = noise_aug_strength
        self.generator = generator

    def update_cache(
        self,
        new_image: torch.Tensor,
        new_depth: torch.Tensor,
        new_w2c: torch.Tensor,
        new_mask: torch.Tensor | None = None,
        new_intrinsics: torch.Tensor | None = None,
    ):
        """
        Update the cache with a new frame.
        
        Args:
            new_image: (B, C, H, W) new RGB image
            new_depth: (B, 1, H, W) new depth map
            new_w2c: (B, 4, 4) new world-to-camera transformation
            new_mask: Optional (B, 1, H, W) validity mask
            new_intrinsics: (B, 3, 3) camera intrinsics (optional)
        """
        new_image = new_image.to(self.weight_dtype).to(self.device)
        new_depth = new_depth.to(self.weight_dtype).to(self.device)
        new_w2c = new_w2c.to(self.weight_dtype).to(self.device)
        if new_intrinsics is not None:
            new_intrinsics = new_intrinsics.to(self.weight_dtype).to(
                self.device)

        new_depth = torch.nan_to_num(new_depth, nan=1e4)
        new_depth = torch.clamp(new_depth, min=0, max=1e4)

        B, F, N, V, C, H, W = self.input_image.shape

        # Compute new 3D points
        new_points = unproject_points(new_depth,
                                      new_w2c,
                                      new_intrinsics,
                                      is_depth=self.is_depth).cpu()
        new_image = new_image.cpu()

        if self.filter_points_threshold < 1.0:
            new_depth = new_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(
                new_depth,
                ratio_thresh=self.filter_points_threshold).reshape(B, 1, H, W)
            if new_mask is None:
                new_mask = depth_mask.to("cpu")
            else:
                new_mask = new_mask * depth_mask.to(new_mask.device)
        if new_mask is not None:
            new_mask = new_mask.cpu()

        # Update buffer (newest frame first)
        if self.frame_buffer_max > 1:
            if self.input_image.shape[2] < self.frame_buffer_max:
                self.input_image = torch.cat(
                    [new_image[:, None, None, None], self.input_image], 2)
                self.input_points = torch.cat(
                    [new_points[:, None, None, None], self.input_points], 2)
                if self.input_mask is not None:
                    self.input_mask = torch.cat(
                        [new_mask[:, None, None, None], self.input_mask], 2)
            else:
                self.input_image[:, :, 0] = new_image[:, None, None]
                self.input_points[:, :, 0] = new_points[:, None, None]
                if self.input_mask is not None:
                    self.input_mask[:, :, 0] = new_mask[:, None, None]
        else:
            self.input_image = new_image[:, None, None, None]
            self.input_points = new_points[:, None, None, None]

    def render_cache(
        self,
        target_w2cs: torch.Tensor,
        target_intrinsics: torch.Tensor,
        render_depth: bool = False,
        start_frame_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render the cache with optional noise augmentation.
        
        Args:
            target_w2cs: (b, F_target, 4, 4) target camera transformations
            target_intrinsics: (b, F_target, 3, 3) target camera intrinsics
            render_depth: If True, return depth instead of RGB
            start_frame_idx: Starting frame index (must be 0 for this class)
            
        Returns:
            pixels: (b, F_target, N, c, h, w) rendered images
            masks: (b, F_target, N, 1, h, w) validity masks
        """
        assert start_frame_idx == 0, "start_frame_idx must be 0 for Cache3DBuffer"

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(
            self.device)

        pixels, masks = super().render_cache(target_w2cs, target_intrinsics,
                                             render_depth)

        pixels = pixels.to(output_device)
        masks = masks.to(output_device)

        # Apply noise augmentation (stronger for older buffers)
        if not render_depth and self.noise_aug_strength > 0:
            noise = torch.randn(pixels.shape,
                                generator=self.generator,
                                device=pixels.device,
                                dtype=pixels.dtype)
            per_buffer_noise = (torch.arange(start=pixels.shape[2] - 1,
                                             end=-1,
                                             step=-1,
                                             device=pixels.device) *
                                self.noise_aug_strength)
            pixels = pixels + noise * per_buffer_noise.reshape(
                1, 1, -1, 1, 1, 1)

        return pixels, masks
