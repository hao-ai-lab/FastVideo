# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange, repeat

from fastvideo.layers.rotary_embedding import _apply_rotary_emb as fastvideo_apply_rotary_emb
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

from . import log
from .a2a_cp import MinimalA2AAttnOp
from .selective_activation_checkpoint import CheckpointMode, SACConfig
from .context_parallel import split_inputs_cp

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

from collections import namedtuple

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None

    @property
    def seq_dim(self):
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size):
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self._is_initialized = False

    def cache_parameters(self) -> None:
        if self._is_initialized:
            return

        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().cuda()
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t
        self._is_initialized = True

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        self.cache_parameters()

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        freqs_h = torch.outer(self.seq[:H], h_spatial_freqs)
        freqs_w = torch.outer(self.seq[:W], w_spatial_freqs)

        freqs_t = torch.outer(self.seq[:T], temporal_freqs)

        freqs_T_H_W_D = torch.cat(
            [
                repeat(freqs_t, "t d -> t h w d", h=H, w=W),
                repeat(freqs_h, "h d -> t h w d", t=T, w=W),
                repeat(freqs_w, "w d -> t h w d", t=T, h=H),
            ],
            dim=-1,
        )

        return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_apply(x, video_size: VideoSize, freqs):
    """
    Optimized version of rope_apply using flash_attention's rotary embedding implementation.
    This version processes the entire batch at once for efficiency.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads, head_dim]
        video_size (VideoSize): Video dimensions with shape [T, H, W]
        freqs (Tensor): Complex frequencies with shape [max_seq_len, head_dim // 2]

    Returns:
        Tensor: Rotary-embedded tensor with same shape as input
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    # Since all items in the batch share the same grid dimensions, we can use the first item
    T, H, W = video_size
    curr_seq_len = T * H * W

    # Make sure the sequence length matches the grid size
    assert seq_len == curr_seq_len, "Sequence length must be equal to T*H*W"

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    # Apply the rotation
    # Apply the rotation
    rotated = fastvideo_apply_rotary_emb(x.to(torch.float32), cos, sin, is_neox_style=False)

    return rotated.to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.qk_norm = qk_norm

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op = MinimalA2AAttnOp()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.q.weight, std=std)
        torch.nn.init.trunc_normal_(self.k.weight, std=std)
        torch.nn.init.trunc_normal_(self.v.weight, std=std)
        torch.nn.init.trunc_normal_(self.o.weight, std=std)
        # zero out bias
        self.q.bias.data.zero_()
        self.k.bias.data.zero_()
        self.v.bias.data.zero_()
        self.o.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_q.reset_parameters()
            self.norm_k.reset_parameters()

    def forward(self, x, seq_lens, video_size: VideoSize, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            video_size(VideoSize): Shape [T, H, W]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = self.attn_op(rope_apply(q, video_size, freqs), rope_apply(k, video_size, freqs), v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = self.attn_op(q, k, v)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op_image = MinimalA2AAttnOp()

    def init_weights(self):
        super().init_weights()
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.k_img.weight, std=std)
        torch.nn.init.trunc_normal_(self.v_img.weight, std=std)
        # zero out bias
        self.k_img.bias.data.zero_()
        self.v_img.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_k_img.reset_parameters()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = self.attn_op_image(q, k_img, v_img)
        # compute attention
        x = self.attn_op(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {"t2v_cross_attn": WanT2VCrossAttention, "i2v_cross_attn": WanI2VCrossAttention}


class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)

    def forward(self, x, e, seq_lens, video_size: VideoSize, freqs, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            video_size(VideoSize): Shape [T, H, W]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, video_size, freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].type_as(x)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].type_as(x)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def init_weights(self):
        self.norm.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)
        torch.nn.init.trunc_normal_(self.head.weight, std=std)
        self.head.bias.data.zero_()

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        input_dtype = x.dtype
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x.to(dtype=input_dtype)


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def init_weights(self):
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()
        self.proj[3].reset_parameters()
        self.proj[4].reset_parameters()

        if hasattr(self, "emb_pos"):
            self.emb_pos.data.zero_()

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        sac_config: SACConfig = SACConfig(),
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_crossattn_projection = False

        # embeddings
        self.patch_embedding = nn.Linear(in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        d = dim // num_heads

        self.rope_position_embedding = VideoRopePosition3DEmb(head_dim=d, len_h=128, len_w=128, len_t=32)

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        self.enable_selective_checkpoint(sac_config)

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x_B_C_T_H_W (Tensor):
                Input video tensor with shape [B, C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            frame_cond_crossattn_emb_B_L_D (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y_B_C_T_H_W (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, shape [B, C_in, T, H, W]

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, T, H / 8, W / 8]
        """

        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]
        del kwargs
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        # embeddings
        x_B_T_H_W_D = rearrange(
            x_B_C_T_H_W,
            "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )

        x_B_T_H_W_D = self.patch_embedding(x_B_T_H_W_D)

        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))
            assert e_B_D.dtype == torch.float32 and e0_B_6_D.dtype == torch.float32

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)

        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)  # bs x 257 (x2) x dim
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        # arguments
        kwargs = dict(
            e=e0_B_6_D,
            seq_lens=seq_lens,
            video_size=video_size,
            freqs=self.rope_position_embedding(x_B_T_H_W_D),
            context=context_B_L_D,
            context_lens=context_lens,
        )

        for block_idx, block in enumerate(self.blocks):
            x_B_L_D = block(x_B_L_D, **kwargs)

        # head
        x_B_L_D = self.head(x_B_L_D, e_B_D)

        # unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            t=t,
            h=h,
            w=w,
            d=self.out_dim,
        )
        return x_B_C_T_H_W

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for block in self.blocks:
            block.init_weights()
        self.head.init_weights()

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    def fully_shard(self, mesh, mp_policy):
        for i, block in enumerate(self.blocks):
            fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.head, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
        fully_shard(self.text_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.time_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self.patch_embedding, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)

    def disable_context_parallel(self):
        # pos_embedder
        self.rope_position_embedding.disable_context_parallel()
        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        self.rope_position_embedding.enable_context_parallel(process_group=process_group)
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(process_group=process_group, ranks=cp_ranks, stream=torch.cuda.Stream())

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(f"Enable selective checkpoint with mm_only, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}")
        _context_fn = sac_config.get_context_fn()
        for block_id, block in self.blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                block = ptd_checkpoint_wrapper(block, context_fn=_context_fn, preserve_rng_state=False)
                self.blocks.register_module(block_id, block)
        self.register_module("head", ptd_checkpoint_wrapper(self.head, context_fn=_context_fn, preserve_rng_state=False))

        return self
