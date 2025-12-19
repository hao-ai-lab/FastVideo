import math
import time
import torch
import random
from loguru import logger
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hymm_sp.diffusion import load_diffusion_pipeline
from hymm_sp.helpers import get_nd_rotary_pos_embed_new
from hymm_sp.inference import Inference
from hymm_sp.diffusion.schedulers import FlowMatchDiscreteScheduler
from packaging import version as pver

ACTION_DICT = {"w": "forward", "a": "left", "d": "right", "s": "backward", "left_rot":"left_rot", "right_rot":"right_rot", "up_rot":"up_rot", "down_rot":"down_rot",}
            
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')
    
def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= 10
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def get_c2w(w2cs, transform_matrix, relative_c2w):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
        for pose in ret_poses:
            pose[:3, -1:] *= 2
        # ret_poses = [poses[:, :3]*2 for poses in ret_poses]
        # ret_poses[:, :, :3] *= 2
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)

def generate_motion_segment(current_pose,
                            motion_type: str, 
                            value: float, 
                            duration: int = 30):
    """
    Parameters:
        motion_type: ('forward', 'backward', 'left', 'right', 
                            'rotate_left', 'rotate_right', 'rotate_up', 'rotate_down')
        value: Translation(m) or Rotation(degree)
        duration: frames
        
    Return:
        positions: [np.array(x,y,z), ...]
        rotations: [np.array(pitch,yaw,roll), ...]
    """
    positions = []
    rotations = []

    if motion_type in ['forward', 'backward']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        pitch_rad = np.radians(current_pose['rotation'][0])
        
        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        
        direction = 1 if motion_type == 'forward' else -1
        total_move = forward_vec * value * direction
        step = total_move / duration
        
        for i in range(1, duration+1):
            new_pos = current_pose['position'] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose['rotation'].copy())
            
        current_pose['position'] = positions[-1]
        
    elif motion_type in ['left', 'right']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        right_vec = np.array([math.cos(yaw_rad), 0, -math.sin(yaw_rad)])
        
        direction = -1 if motion_type == 'right' else 1
        total_move = right_vec * value * direction
        step = total_move / duration
        
        for i in range(1, duration+1):
            new_pos = current_pose['position'] + step * i
            positions.append(new_pos.copy())
            rotations.append(current_pose['rotation'].copy())
            
        current_pose['position'] = positions[-1]
        
    elif motion_type.endswith('rot'):
        axis = motion_type.split('_')[0]
        total_rotation = np.zeros(3)
        
        if axis == 'left':
            total_rotation[0] = value  
        elif axis == 'right':
            total_rotation[0] = -value   
        elif axis == 'up':
            total_rotation[2] = -value  
        elif axis == 'down':
            total_rotation[2] = value   
            
        step = total_rotation / duration
        
        for i in range(1, duration+1):
            positions.append(current_pose['position'].copy())
            new_rot = current_pose['rotation'] + step * i
            rotations.append(new_rot.copy())
            
        current_pose['rotation'] = rotations[-1]
    
    return positions, rotations, current_pose

def euler_to_quaternion(angles):
    pitch, yaw, roll = np.radians(angles)
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    
    return [qw, qx, qy, qz]

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
def ActionToPoseFromID(action_id, value=0.2, duration=33):
    
    all_positions = []
    all_rotations = []
    current_pose = {
        'position': np.array([0.0, 0.0, 0.0]),  # XYZ
        'rotation': np.array([0.0, 0.0, 0.0])   # (pitch, yaw, roll)
    }
    intrinsic = [0.50505, 0.8979, 0.5, 0.5]  
    motion_type = ACTION_DICT[action_id]
    positions, rotations, current_pose = generate_motion_segment(current_pose, motion_type, value, duration)
    all_positions.extend(positions)
    all_rotations.extend(rotations)
    
    pose_list = []

    row = [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    first_row = " ".join(map(str, row))
    pose_list.append(first_row)
    for i, (pos, rot) in enumerate(zip(all_positions, all_rotations)):
        quat = euler_to_quaternion(rot)
        R = quaternion_to_rotation_matrix(quat)
        extrinsic = np.hstack([R, pos.reshape(3, 1)])
        
        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))

    return pose_list

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        # self.ax.view_init(elev=25, azim=-90)
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], 
                             vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                          ax=self.ax, orientation='vertical', label='Frame Number')

    def show(self, file_name):
        plt.title('Extrinsic Parameters')
        # plt.show()
        plt.savefig(file_name)


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def GetPoseEmbedsFromPoses(poses, h, w, target_length, flip=False, start_index=None):

    poses = [pose.split(' ') for pose in poses]

    start_idx = start_index
    sample_id = [start_idx + i for i in range(target_length)]
    
    poses = [poses[i] for i in sample_id]

    frame = len(poses)
    w2cs = [np.asarray([float(p) for p in pose[7:]]).reshape(3, 4) for pose in poses]
    transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
    last_row = np.zeros((1, 4))
    last_row[0, -1] = 1.0
    w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
    c2ws = get_c2w(w2cs, transform_matrix, relative_c2w=True)
    
    cam_params = [[float(x) for x in pose] for pose in poses]
    assert len(cam_params) == target_length
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    monst3r_w = cam_params[0].cx * 2
    monst3r_h = cam_params[0].cy * 2
    ratio_w, ratio_h = w/monst3r_w, h/monst3r_h
    intrinsics = np.asarray([[cam_param.fx * ratio_w,
                                cam_param.fy * ratio_h,
                                cam_param.cx * ratio_w,
                                cam_param.cy * ratio_h]
                                for cam_param in cam_params], dtype=np.float32)
    intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
    relative_pose = True
    if relative_pose:
        c2w_poses = get_relative_pose(cam_params)
    else:
        c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
    uncond_c2w = torch.zeros_like(c2w)
    uncond_c2w[:, :] = torch.eye(4, device=c2w.device)
    flip_flag = torch.zeros(target_length, dtype=torch.bool, device=c2w.device)
    plucker_embedding = ray_condition(intrinsics, c2w, h, w, device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
    uncond_plucker_embedding = ray_condition(intrinsics, uncond_c2w, h, w, device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()

    return plucker_embedding, uncond_plucker_embedding, poses

def GetPoseEmbedsFromTxt(pose_dir, h, w, target_length, flip=False, start_index=None, step=1):
    # get camera pose
    with open(pose_dir, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    start_idx = start_index
    sample_id = [start_idx + i*step for i in range(target_length)]
    poses = [poses[i] for i in sample_id]
    
    cam_params = [[float(x) for x in pose] for pose in poses]
    assert len(cam_params) == target_length
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    monst3r_w = cam_params[0].cx * 2
    monst3r_h = cam_params[0].cy * 2
    ratio_w, ratio_h = w/monst3r_w, h/monst3r_h
    intrinsics = np.asarray([[cam_param.fx * ratio_w,
                                cam_param.fy * ratio_h,
                                cam_param.cx * ratio_w,
                                cam_param.cy * ratio_h]
                                for cam_param in cam_params], dtype=np.float32)
    intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
    relative_pose = True
    if relative_pose:
        c2w_poses = get_relative_pose(cam_params)
    else:
        c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
    uncond_c2w = torch.zeros_like(c2w)
    uncond_c2w[:, :] = torch.eye(4, device=c2w.device)
    if flip:
        flip_flag = torch.ones(target_length, dtype=torch.bool, device=c2w.device)
    else:
        flip_flag = torch.zeros(target_length, dtype=torch.bool, device=c2w.device)
    plucker_embedding = ray_condition(intrinsics, c2w, h, w, device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
    uncond_plucker_embedding = ray_condition(intrinsics, uncond_c2w, h, w, device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()

    return plucker_embedding, uncond_plucker_embedding, poses


class HunyuanVideoSampler(Inference):
    def __init__(self, args, vae, vae_kwargs, text_encoder, model, text_encoder_2=None, pipeline=None,
                 device=0, logger=None):
        super().__init__(args, vae, vae_kwargs, text_encoder, model, text_encoder_2=text_encoder_2,
                         pipeline=pipeline,  device=device, logger=logger)
        
        self.args = args
        self.pipeline = load_diffusion_pipeline(
            args, 0, self.vae, self.text_encoder, self.text_encoder_2, self.model,
            device=self.device)
        print('load hunyuan model successful... ')

    def get_rotary_pos_embed(self, video_length, height, width, concat_dict={}):
        target_ndim = 3
        ndim = 5 - 2
        if '884' in self.args.vae:
            latents_size = [(video_length-1)//4+1 , height//8, width//8]
        else:
            latents_size = [video_length , height//8, width//8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.num_heads
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(rope_dim_list, 
                                                    rope_sizes, 
                                                    theta=self.args.rope_theta, 
                                                    use_real=True,
                                                    theta_rescale_factor=1,
                                                    concat_dict=concat_dict)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(self, 
                prompt, 
                is_image=True,
                size=(720, 1280),
                video_length=129,
                seed=None,
                negative_prompt=None,
                infer_steps=50,
                guidance_scale=6.0,
                flow_shift=5.0,
                batch_size=1,
                num_videos_per_prompt=1,
                verbose=1,
                output_type="pil",
                **kwargs):
        """
        Predict the image from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                size (int): The (height, width) of the output image/video. Default is (256, 256).
                video_length (int): The frame number of the output video. Default is 1.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                infer_steps (int): The number of inference steps. Default is 100.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_videos_per_prompt (int): The number of videos per prompt. Default is 1.    
                verbose (int): 0 for no log, 1 for all log, 2 for fewer log. Default is 1.
                output_type (str): The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
                    Default is 'pil'.
        """
        
        out_dict = dict()

        # ---------------------------------
        # Prompt
        # ---------------------------------
        prompt_embeds = kwargs.get("prompt_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        negative_attention_mask = kwargs.get("negative_attention_mask", None)
        ref_latents = kwargs.get("ref_latents", None)
        uncond_ref_latents = kwargs.get("uncond_ref_latents", None)
        return_latents = kwargs.get("return_latents", False)
        negative_prompt = kwargs.get("negative_prompt", None)
        
        action_id = kwargs.get("action_id", None)
        action_speed = kwargs.get("action_speed", None)
        start_index = kwargs.get("start_index", None)
        last_latents = kwargs.get("last_latents", None)
        ref_latents = kwargs.get("ref_latents", None)
        input_pose = kwargs.get("input_pose", None)
        step = kwargs.get("step", 1)
        use_sage = kwargs.get("use_sage", False)
        
        size = self.parse_size(size)
        target_height = align_to(size[0], 16)
        target_width = align_to(size[1], 16)
        # target_video_length = video_length
        
        if input_pose is not None:
            pose_embeds, uncond_pose_embeds, poses = GetPoseEmbedsFromTxt(input_pose, 
                                                                          target_height, 
                                                                          target_width, 
                                                                          33, 
                                                                          kwargs.get("flip", False), 
                                                                          start_index, 
                                                                          step)
        else:
            pose = ActionToPoseFromID(action_id, value=action_speed)
            pose_embeds, uncond_pose_embeds, poses = GetPoseEmbedsFromPoses(pose, 
                                                                            target_height, 
                                                                            target_width, 
                                                                            33, 
                                                                            kwargs.get("flip", False), 
                                                                            0)

        if is_image:
            target_length = 34
        else:
            target_length = 66
            
        out_dict['frame'] = target_length
        # print("pose embeds: ", pose_embeds.shape, uncond_pose_embeds.shape)

        pose_embeds = pose_embeds.unsqueeze(0).to(torch.bfloat16).to('cuda')
        uncond_pose_embeds = uncond_pose_embeds.unsqueeze(0).to(torch.bfloat16).to('cuda')

        
        
        cpu_offload = kwargs.get("cpu_offload", 0)
        use_deepcache = kwargs.get("use_deepcache", 1)
        denoise_strength = kwargs.get("denoise_strength", 1.0)
        init_latents = kwargs.get("init_latents", None)
        mask = kwargs.get("mask", None)
        if prompt is None:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should not be None
            # pipeline will help to check this
            prompt = None
            negative_prompt = None
            batch_size = prompt_embeds.shape[0]
            assert prompt_embeds is not None
        else:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should be None
            # pipeline will help to check this
            if isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif isinstance(prompt, (list, tuple)):
                batch_size = len(prompt)
            else:
                raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            
        # ---------------------------------
        # Other arguments
        # ---------------------------------
        scheduler = FlowMatchDiscreteScheduler(shift=flow_shift,
                                                reverse=self.args.flow_reverse,
                                                solver=self.args.flow_solver,
                                                )
        self.pipeline.scheduler = scheduler

        # ---------------------------------
        # Random seed
        # ---------------------------------
        
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        
        # ---------------------------------
        # Image/Video size and frame
        # ---------------------------------
        

        out_dict['size'] = (target_height, target_width)
        out_dict['video_length'] = target_length
        out_dict['seeds'] = seeds
        out_dict['negative_prompt'] = negative_prompt
        # ---------------------------------
        # Build RoPE
        # ---------------------------------

        concat_dict = {'mode': 'timecat', 'bias': -1} 
        if is_image:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(37, target_height, target_width)
        else:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(69, target_height, target_width)
        
        n_tokens = freqs_cos.shape[0]
        
        # ---------------------------------
        # Inference
        # ---------------------------------
        output_dir = kwargs.get("output_dir", None)
        
        if verbose == 1:
            debug_str = f"""
                  size: {out_dict['size']}
          video_length: {target_length}
                prompt: {prompt}
            neg_prompt: {negative_prompt}
                  seed: {seed}
           infer_steps: {infer_steps}
      denoise_strength: {denoise_strength}
         use_deepcache: {use_deepcache}
              use_sage: {use_sage}
           cpu_offload: {cpu_offload}
 num_images_per_prompt: {num_videos_per_prompt}
        guidance_scale: {guidance_scale}
              n_tokens: {n_tokens}
            flow_shift: {flow_shift}
                output: {output_dir}"""
            self.logger.info(debug_str)

        start_time = time.time()
        samples = self.pipeline(prompt=prompt,   
                                last_latents=last_latents,
                                cam_latents=pose_embeds,
                                uncond_cam_latents=uncond_pose_embeds,
                                height=target_height,
                                width=target_width,
                                video_length=target_length,
                                gt_latents = ref_latents,
                                num_inference_steps=infer_steps,
                                guidance_scale=guidance_scale,
                                negative_prompt=negative_prompt,
                                num_videos_per_prompt=num_videos_per_prompt,
                                generator=generator,
                                prompt_embeds=prompt_embeds,
                                ref_latents=ref_latents,
                                latents=init_latents,
                                denoise_strength=denoise_strength,
                                mask=mask,
                                uncond_ref_latents=uncond_ref_latents,
                                ip_cfg_scale=self.args.ip_cfg_scale, 
                                use_deepcache=use_deepcache,
                                attention_mask=attention_mask,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_attention_mask=negative_attention_mask,
                                output_type=output_type,
                                freqs_cis=(freqs_cos, freqs_sin),
                                n_tokens=n_tokens,
                                data_type='video' if target_length > 1 else 'image',
                                is_progress_bar=True,
                                vae_ver=self.args.vae,
                                enable_tiling=self.args.vae_tiling,
                                cpu_offload=cpu_offload, 
                                return_latents=return_latents,
                                use_sage=use_sage,
                                )
        if samples is None:
            return None
        out_dict['samples'] = []
        out_dict["prompts"] = prompt
        out_dict['pose'] = poses

        if return_latents:
            print("return_latents | TRUE")
            latents, timesteps, last_latents, ref_latents = samples[1], samples[2], samples[3], samples[4]
            # samples = samples[0][0]
            if samples[0] is not None and len(samples[0]) > 0:
                samples = samples[0][0]
            else:
                samples = None
            out_dict["denoised_lantents"] = latents
            out_dict["timesteps"] = timesteps
            out_dict["ref_latents"] = ref_latents
            out_dict["last_latents"] = last_latents
        
        else:
            samples = samples[0]

        if samples is not None:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                sub_samples = []
                sub_samples.append(sample)
                sample_num = len(sub_samples)
                sub_samples = torch.concat(sub_samples)
                # only save in tp rank 0
                out_dict['samples'].append(sub_samples)

                # visualize pose
        
        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")
        return out_dict
    
