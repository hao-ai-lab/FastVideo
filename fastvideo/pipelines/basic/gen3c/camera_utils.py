# SPDX-License-Identifier: Apache-2.0
# Ported from NVIDIA GEN3C: cosmos_predict1/diffusion/inference/camera_utils.py
"""Camera trajectory generation utilities for GEN3C 3D cache conditioning."""

import math

import torch


def apply_transformation(Bx4x4: torch.Tensor,
                         another_matrix: torch.Tensor) -> torch.Tensor:
    """Apply batch transformation to a matrix."""
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)
    return torch.bmm(Bx4x4, another_matrix)


def look_at_matrix(camera_pos: torch.Tensor,
                   target: torch.Tensor,
                   invert_pos: bool = True) -> torch.Tensor:
    """Create a 4x4 look-at view matrix pointing camera toward target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    look_at[:3, 3] = (-camera_pos) if invert_pos else camera_pos

    return look_at


def create_horizontal_trajectory(
    world_to_camera_matrix: torch.Tensor,
    center_depth: float,
    positive: bool = True,
    n_steps: int = 13,
    distance: float = 0.1,
    device: str = "cuda",
    axis: str = "x",
    camera_rotation: str = "center_facing",
) -> torch.Tensor:
    """Create a linear camera trajectory along a specified axis."""
    look_at_target = torch.tensor([0.0, 0.0, center_depth]).to(device)
    trajectory = []
    initial_camera_pos = torch.tensor([0, 0, 0],
                                      device=device,
                                      dtype=torch.float32)

    translation_positions = []
    for i in range(n_steps):
        offset = i * distance * center_depth / n_steps * (1 if positive else -1)
        if axis == "x":
            pos = torch.tensor([offset, 0, 0], device=device)
        elif axis == "y":
            pos = torch.tensor([0, offset, 0], device=device)
        elif axis == "z":
            pos = torch.tensor([0, 0, offset], device=device)
        else:
            raise ValueError(f"Axis should be x, y or z, got {axis}")
        translation_positions.append(pos)

    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if camera_rotation == "trajectory_aligned":
            _look_at = look_at_target + pos * 2
        elif camera_rotation == "center_facing":
            _look_at = look_at_target
        elif camera_rotation == "no_rotation":
            _look_at = look_at_target + pos
        else:
            raise ValueError(
                f"camera_rotation should be center_facing, trajectory_aligned, "
                f"or no_rotation, got {camera_rotation}")
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)

    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_spiral_trajectory(
    world_to_camera_matrix: torch.Tensor,
    center_depth: float,
    radius_x: float = 0.03,
    radius_y: float = 0.02,
    radius_z: float = 0.0,
    positive: bool = True,
    camera_rotation: str = "center_facing",
    n_steps: int = 13,
    device: str = "cuda",
    start_from_zero: bool = True,
    num_circles: int = 1,
) -> torch.Tensor:
    """Create a spiral/circular camera trajectory."""
    look_at_target = torch.tensor([0.0, 0.0, center_depth]).to(device)
    trajectory = []
    initial_camera_pos = torch.tensor([0, 0, 0],
                                      device=device,
                                      dtype=torch.float32)

    theta_max = 2 * math.pi * num_circles
    spiral_positions = []

    for i in range(n_steps):
        theta = theta_max * i / (n_steps - 1)
        if start_from_zero:
            x = radius_x * (math.cos(theta) -
                            1) * (1 if positive else -1) * center_depth
        else:
            x = radius_x * math.cos(theta) * center_depth
        y = radius_y * math.sin(theta) * center_depth
        z = radius_z * math.sin(theta) * center_depth
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        camera_pos = initial_camera_pos + pos
        if camera_rotation == "center_facing":
            view_matrix = look_at_matrix(camera_pos, look_at_target)
        elif camera_rotation == "trajectory_aligned":
            view_matrix = look_at_matrix(camera_pos, look_at_target + pos * 2)
        elif camera_rotation == "no_rotation":
            view_matrix = look_at_matrix(camera_pos, look_at_target + pos)
        else:
            raise ValueError(
                f"camera_rotation should be center_facing, trajectory_aligned, "
                f"or no_rotation, got {camera_rotation}")
        trajectory.append(view_matrix)

    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def generate_camera_trajectory(
    trajectory_type: str,
    initial_w2c: torch.Tensor,
    initial_intrinsics: torch.Tensor,
    num_frames: int,
    movement_distance: float,
    camera_rotation: str = "center_facing",
    center_depth: float = 1.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera trajectory for GEN3C video generation.

    Args:
        trajectory_type: One of "left", "right", "up", "down", "zoom_in",
            "zoom_out", "clockwise", "counterclockwise".
        initial_w2c: Initial world-to-camera matrix (4, 4).
        initial_intrinsics: Camera intrinsics matrix (3, 3).
        num_frames: Number of frames in the trajectory.
        movement_distance: Distance factor for camera movement.
        camera_rotation: "center_facing", "no_rotation", or "trajectory_aligned".
        center_depth: Depth of the scene center point.
        device: Computation device.

    Returns:
        generated_w2cs: (1, num_frames, 4, 4) world-to-camera matrices.
        generated_intrinsics: (1, num_frames, 3, 3) camera intrinsics.
    """
    if trajectory_type in ["clockwise", "counterclockwise"]:
        new_w2cs_seq = create_spiral_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=trajectory_type == "clockwise",
            device=device,
            camera_rotation=camera_rotation,
            radius_x=movement_distance,
            radius_y=movement_distance,
        )
    elif trajectory_type == "none":
        # Static camera - repeat identity
        new_w2cs_seq = initial_w2c.unsqueeze(0).expand(num_frames, -1, -1)
    else:
        axis_map = {
            "left": (False, "x"),
            "right": (True, "x"),
            "up": (False, "y"),
            "down": (True, "y"),
            "zoom_in": (True, "z"),
            "zoom_out": (False, "z"),
        }
        if trajectory_type not in axis_map:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")
        positive, axis = axis_map[trajectory_type]

        new_w2cs_seq = create_horizontal_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=positive,
            axis=axis,
            distance=movement_distance,
            device=device,
            camera_rotation=camera_rotation,
        )

    generated_w2cs = new_w2cs_seq.unsqueeze(0)  # (1, num_frames, 4, 4)
    if initial_intrinsics.dim() == 2:
        generated_intrinsics = initial_intrinsics.unsqueeze(0).unsqueeze(
            0).repeat(1, num_frames, 1, 1)
    else:
        generated_intrinsics = initial_intrinsics.unsqueeze(0)

    return generated_w2cs, generated_intrinsics
