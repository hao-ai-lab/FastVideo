import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray, 
    src_rot_mat: np.ndarray, 
    src_trans_vec: np.ndarray, 
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # interpolate translation
    interp_func_trans = interp1d(
        src_indices, 
        src_trans_vec, 
        axis=0, 
        kind='linear', 
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # ensure there is no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3] # [B,3,3]
    trans = T[:, :3, 3:] # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor, 
    framewise: bool = False, 
    normalize_trans: bool = True, 
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans: # note refer to camctrl2: "we scale the coordinate inputs to roughly 1 standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3] # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normlaize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(n_frames: int, height: int, width: int, bias: float = 0.5, device='cuda', dtype=torch.float32) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1) # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype) # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1) # [f, 1]

    i = grid_xy[..., 0] # [f, h*w]
    j = grid_xy[..., 1] # [f, h*w]
    zs = torch.ones_like(i) # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1) # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True) # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2) # [f, h*w, 3]
    rays_o = c2ws_mat[:, :3, 3] # [f, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d) # [f, h*w, 3]
    # rays_dxo = torch.cross(rays_o, rays_d, dim=-1) # [f, h*w, 3]
    # note refer to: apt2
    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1) # [f, h*w, 6]
    plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6]) # [f*h*w, 6]
    return plucker_embeddings


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = Ks.chunk(4, dim=-1) # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y
    
    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


def _reformat_keyboard_and_mouse_cond(
    num_frames: int,
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    compression_ratio: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert frame-level controls to latent-level controls."""

    # keyboard / mouse shape: [T, C]
    assert keyboard_cond.shape[0] == num_frames and mouse_cond.shape[0] == num_frames, (
        "keyboard_cond and mouse_cond must have the same number of frames as num_frames, "
        f"num_frames: {num_frames} "
        f"got keyboard_cond shape: {keyboard_cond.shape}, mouse_cond shape: {mouse_cond.shape}"
    )
    assert (num_frames - 1) % compression_ratio == 0, (
        f"num_frames must satisfy (num_frames - 1) % {compression_ratio} == 0, "
        f"got {num_frames}"
    )
    keyboard_cond = keyboard_cond[1:, :]
    mouse_cond = mouse_cond[1:, :]

    groups = keyboard_cond.view(-1, compression_ratio, keyboard_cond.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), (
        "keyboard_tensor must be constant within each compression group"
    )
    groups = mouse_cond.view(-1, compression_ratio, mouse_cond.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), (
        "mouse_tensor must be constant within each compression group"
    )
    return keyboard_cond[::compression_ratio], mouse_cond[::compression_ratio]


def _motions_to_c2ws(
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    forward_speed: float = 0.08,  # from HYW
) -> torch.Tensor:
    n_steps = keyboard_cond.shape[0]
    poses = []
    pose = torch.eye(4, dtype=torch.float32)
    poses.append(pose.clone())

    for t in range(n_steps):
        forward = 0.0
        right = 0.0
        if keyboard_cond[t, 0] > 0.9:  # W
            forward += forward_speed
        if keyboard_cond[t, 1] > 0.9:  # S
            forward -= forward_speed
        if keyboard_cond[t, 2] > 0.9:  # A
            right -= forward_speed
        if keyboard_cond[t, 3] > 0.9:  # D
            right += forward_speed

        pitch = float(mouse_cond[t, 0].item())
        yaw = float(mouse_cond[t, 1].item())

        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        r_pitch = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=torch.float32
        )
        r_yaw = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float32
        )
        r_delta = r_yaw @ r_pitch

        delta = torch.eye(4, dtype=torch.float32)
        delta[:3, :3] = r_delta
        delta[:3, 3] = torch.tensor([right, 0.0, forward], dtype=torch.float32)
        pose = pose @ delta
        poses.append(pose.clone())

    return torch.stack(poses, dim=0)


def process_custom_actions(
    num_frames: int,
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    height_org: int = 480,
    width_org: int = 832,
    spatial_scale: int = 8,
    intrinsics: torch.Tensor | None = None,
    forward_speed: float = 0.08,  # from HYW
    normalize_trans: bool = True,
) -> torch.Tensor:
    """
    Convert frame-level keyboard/mouse controls to camera condition.

    Returns:
        c2ws_plucker_emb with shape [B=1, C=6*spatial_scale^2, F_lat, H_lat, W_lat]
    """
    if keyboard_cond.ndim == 3:
        keyboard_cond = keyboard_cond.squeeze(0)
    if mouse_cond.ndim == 3:
        mouse_cond = mouse_cond.squeeze(0)
    assert keyboard_cond.ndim == 2 and mouse_cond.ndim == 2, (
        "keyboard_cond and mouse_cond must be [T, C]"
    )

    keyboard_cond, mouse_cond = _reformat_keyboard_and_mouse_cond(
        num_frames, keyboard_cond.float(), mouse_cond.float()
    )
    c2ws = _motions_to_c2ws(
        keyboard_cond=keyboard_cond,
        mouse_cond=mouse_cond,
        forward_speed=forward_speed,
    )
    c2ws = compute_relative_poses(c2ws, framewise=True, normalize_trans=normalize_trans)

    lat_f = c2ws.shape[0]
    h = latent_height * spatial_scale
    w = latent_width * spatial_scale

    if intrinsics is None:
        # Fallback pinhole intrinsics centered at image center.
        fx = float(width_org)
        fy = float(height_org)
        cx = float(width_org) / 2.0
        cy = float(height_org) / 2.0
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    intrinsics = intrinsics.float()
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0).repeat(lat_f, 1)
    assert intrinsics.shape == (lat_f, 4), (
        f"intrinsics must be [F_lat, 4], got {tuple(intrinsics.shape)}"
    )
    Ks = get_Ks_transformed(
        intrinsics,
        height_org=height_org,
        width_org=width_org,
        height_resize=h,
        width_resize=w,
        height_final=h,
        width_final=w,
    )
    plucker = get_plucker_embeddings(c2ws, Ks, h, w)  # [F, H, W, 6]

    c1 = h // latent_height
    c2 = w // latent_width
    assert c1 == spatial_scale and c2 == spatial_scale
    plucker = plucker.view(lat_f, latent_height, c1, latent_width, c2, 6)
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).contiguous()
    plucker = plucker.view(lat_f, latent_height, latent_width, 6 * c1 * c2)
    c2ws_plucker_emb = plucker.permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    return c2ws_plucker_emb