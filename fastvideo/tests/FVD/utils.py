# Original code from https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/videogpt/fvd.py
import math
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

from .pytorch_i3d import InceptionI3d


def load_i3d_pretrained(device=torch.device('cpu')):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_pretrained_400.pt')
    i3d = InceptionI3d(400, in_channels=3).eval().to(device)
    if os.path.exists(filepath):
        i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d = torch.nn.DataParallel(i3d)
    return i3d

def preprocess_single(video, resolution, sequence_length=None):
    video = video.permute(0, 3, 1, 2).float() / 255. 
    t, c, h, w = video.shape
    if sequence_length is not None:
        video = video[:sequence_length]
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() 
    video -= 0.5
    return video

def preprocess(videos, target_resolution=224):
    videos = einops.rearrange(videos, 'b c t h w -> b t h w c')
    videos = (videos*255).numpy().astype(np.uint8)
    videos = torch.from_numpy(videos)
    videos = torch.stack([preprocess_single(video, target_resolution) for video in videos])
    return videos * 2 

def get_logits(i3d, videos, device, bs=10):
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], bs):
            batch = videos[i:i + bs].to(device)
            logits.append(i3d(batch))
        logits = torch.cat(logits, dim=0)
        return logits

def get_fvd_logits(videos, i3d, device, bs=10):
    videos = preprocess(videos)
    embeddings = get_logits(i3d, videos, device, bs=bs)
    return embeddings

def cov(m, rowvar=False):
    if m.dim() > 2: raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2: m = m.view(1, -1)
    if not rowvar and m.size(0) != 1: m = m.t()
    fact = 1.0 / (m.size(1) - 1) 
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t() 
    return fact * m.matmul(mt).squeeze()

def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
    mean = torch.sum((m - m_w) ** 2)
    if x1.shape[0] > 1:
        sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
        trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
        fd = trace + mean
    else:
        fd = np.real(mean)
    return float(fd)

def calculate_single_fvd(reference_path, generated_path, device):
    """
    Helper wrapper to calculate FVD between two single video files.
    """
    i3d = load_i3d_pretrained(device=device)
    
    # Load videos
    ref_video, _, _ = read_video(reference_path, pts_unit='sec', output_format="TCHW")
    gen_video, _, _ = read_video(generated_path, pts_unit='sec', output_format="TCHW")

    # Add Batch Dimension (1, T, C, H, W)
    ref_video = ref_video.unsqueeze(0) 
    gen_video = gen_video.unsqueeze(0)
    
    # Permute to (B, C, T, H, W) as needed for the preprocess function logic
    ref_video = ref_video.permute(0, 2, 1, 3, 4)
    gen_video = gen_video.permute(0, 2, 1, 3, 4)

    # Get Features
    # Note: Using batch size 1 since we only have 1 video
    ref_feats = get_fvd_logits(ref_video, i3d, device, bs=1)
    gen_feats = get_fvd_logits(gen_video, i3d, device, bs=1)

    # Calculate Distance
    return frechet_distance(ref_feats, gen_feats)
