# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/SkyworkAI/Matrix-Game

from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
import torch
import torch.nn as nn
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
import math
from torch.nn.attention.flex_attention import flex_attention

from fastvideo.attention import LocalAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import FP32LayerNorm
from fastvideo.platforms import AttentionBackendEnum


DISABLE_COMPILE = False  # get os env
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class ActionModule(nn.Module):
    """
    action module from https://arxiv.org/pdf/2501.08325
    鼠标控制信号的输入是一个 L*D 的向量
    键盘同样
    """

    def __init__(
        self, 
        mouse_dim_in: int = 2,
        keyboard_dim_in: int = 6,
        hidden_size: int = 128,
        img_hidden_size: int = 1536,
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4, 
        windows_size: int = 3,
        heads_num: int = 16,
        patch_size: list = [1, 2, 2],
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28],
        rope_theta = 256,
        mouse_qk_dim_list = [8, 28, 28],
        enable_mouse = True,
        enable_keyboard = True,
        local_attn_size = 6,
        blocks = [],
    ):
        device = None
        
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = nn.Sequential(nn.Linear(keyboard_dim_in, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = torch.nn.Sequential(
                torch.nn.Linear(mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True),
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(c, c),
                FP32LayerNorm(c, elementwise_affine=True),
            )
            
            head_dim = c // heads_num
            self.t_qkv = ReplicatedLinear(c, c*3, bias=qkv_bias)
            self.img_attn_q_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.img_attn_k_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.proj_mouse = ReplicatedLinear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.key_attn_k_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            
            self.mouse_attn_q = ReplicatedLinear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = ReplicatedLinear(hidden_size * windows_size * vae_time_compression_ratio, keyboard_hidden_dim * 2, bias=qkv_bias)
            self.proj_keyboard = ReplicatedLinear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

        self.mouse_attn_layer = LocalAttention(
            num_heads=heads_num,
            head_size=mouse_hidden_dim // heads_num,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA)
        ) if self.enable_mouse else None

        self.keyboard_attn_layer = LocalAttention(
            num_heads=heads_num,
            head_size=keyboard_hidden_dim // heads_num,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA)
        ) if self.enable_keyboard else None

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(7500, self.patch_size[1], self.patch_size[2], 64, self.mouse_qk_dim_list, start_offset=0)

    def patchify(self, x, patch_size):
        """
        x : (N C T H W)
        """
        pt, ph, pw = self.patch_size
        t, h, w = x.shape[2] //  pt, x.shape[3] // ph, x.shape[4] // pw
        c = x.shape[1]
        x = x.reshape(shape=(x.shape[0], c, t , pt, h , ph, w , pw))
        x = torch.einsum("nctohpwq->nthwcopq", x)
        x = x.reshape(shape=(x.shape[0], t*h*w,  c*pt*ph*pw))
        return x

    def unpatchify(self, x, t, h, w, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c =  x.shape[2] // patch_size #self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list = None, start_offset=0):
        target_ndim = 3
        ndim = 5 - 2
        
        latents_size = [video_length+start_offset, height, width]

        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(
                s % self.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:], freqs_sin[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:]

    def forward(self, x, tt, th, tw, mouse_condition=None, keyboard_condition=None, block_mask_mouse=None, block_mask_keyboard=None, is_causal=False, kv_cache_mouse=None, kv_cache_keyboard=None, start_frame=0, use_rope_keyboard=True, num_frame_per_block=3):
        '''
        hidden_states: B, tt*th*tw, C
        mouse_condition: B, N_frames, C1
        keyboard_condition: B, N_frames, C2
        '''
        assert use_rope_keyboard == True

        B, N_frames, C = keyboard_condition.shape
        assert tt*th*tw == x.shape[1]
        assert ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        
        # Defined freqs_cis early so it's available for both mouse and keyboard
        freqs_cis = (self.freqs_cos, self.freqs_sin)

        assert (N_feats == tt and ((is_causal and kv_cache_mouse == None) or not is_causal)) or ((N_frames - 1) // self.vae_time_compression_ratio + 1 == start_frame + num_frame_per_block and is_causal)
        
        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th*tw) # 65*272*480 -> 17*(272//16)*(480//16) -> 8670
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x
        # padding
        
        pad_t = self.vae_time_compression_ratio * self.windows_size
        if self.enable_mouse and mouse_condition is not None:
            pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
            mouse_condition = torch.cat([pad, mouse_condition], dim=1)
            if is_causal and kv_cache_mouse is not None: 
                mouse_condition = mouse_condition[:, self.vae_time_compression_ratio*(N_feats - num_frame_per_block - self.windows_size) + pad_t:, :] 
                group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(num_frame_per_block)]
            else:
                group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
                
            group_mouse = torch.stack(group_mouse, dim = 1)

            S = th * tw 
            group_mouse = group_mouse.unsqueeze(-1).expand(B, num_frame_per_block, pad_t, C, S)
            group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(B * S, num_frame_per_block, pad_t * C) 

            group_mouse = torch.cat([hidden_states, group_mouse], dim = -1)
            group_mouse = self.mouse_mlp(group_mouse)
            # qkv
            mouse_qkv, _ = self.t_qkv(group_mouse)
            q, k, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num) # BHW F H C
            q = self.img_attn_q_norm(q).to(v)
            k = self.img_attn_k_norm(k).to(v)        
            # rope embd

            
            # freqs_cis = (self.freqs_cos, self.freqs_sin)
            
            
            q, k = apply_rotary_emb(q, k, freqs_cis, start_offset = start_frame, head_first=False)
            ## TODO: adding cache here
            if is_causal:
                if kv_cache_mouse is None:
                    assert q.shape[0] ==  k.shape[0] and q.shape[0] % 880 == 0 # == 880, f"{q.shape[0]},{k.shape[0]}"
                    padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                    padded_q = torch.cat(
                        [q,
                            torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                        device=q.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_k = torch.cat(
                        [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )
                    attn = flex_attention(
                        query=padded_q.transpose(2, 1), # after: B, HW, F, C
                        key=padded_k.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask_mouse
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    current_start = start_frame
                    current_end = current_start + q.shape[1]
                    
                    assert q.shape[1] == num_frame_per_block
                    sink_size = 0
                    max_attention_size = self.local_attn_size
                    sink_tokens = sink_size * 1
                    kv_cache_size = kv_cache_mouse["k"].shape[1]
                    num_new_tokens = q.shape[1]
                    

                    if (current_end > kv_cache_mouse["global_end_index"].item()) and (
                        num_new_tokens + kv_cache_mouse["local_end_index"].item() > kv_cache_size):
                        num_evicted_tokens = num_new_tokens + kv_cache_mouse["local_end_index"].item() - kv_cache_size
                        num_rolled_tokens = kv_cache_mouse["local_end_index"].item() - num_evicted_tokens - sink_tokens
                        kv_cache_mouse["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache_mouse["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache_mouse["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache_mouse["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        # Insert the new keys/values at the end
                        local_end_index = kv_cache_mouse["local_end_index"].item() + current_end - \
                            kv_cache_mouse["global_end_index"].item() - num_evicted_tokens
                        local_start_index = local_end_index - num_new_tokens
                    else:
                        local_end_index = kv_cache_mouse["local_end_index"].item() + current_end - kv_cache_mouse["global_end_index"].item()
                        local_start_index = local_end_index - num_new_tokens
                    kv_cache_mouse["k"][:, local_start_index:local_end_index] = k
                    kv_cache_mouse["v"][:, local_start_index:local_end_index] = v

                    attn = self.mouse_attn_layer(
                        q,
                        kv_cache_mouse["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                        kv_cache_mouse["v"][:, max(0, local_end_index - max_attention_size):local_end_index],
                    )
                    kv_cache_mouse["global_end_index"].fill_(current_end)
                    kv_cache_mouse["local_end_index"].fill_(local_end_index)
            else:
                attn = self.mouse_attn_layer(q, k, v)
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)',b=B)
            
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)
            attn, _ = self.proj_mouse(attn)
            
            hidden_states = hidden_states + attn
        
        if self.enable_keyboard and keyboard_condition is not None:
            pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
            keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
            if is_causal and kv_cache_keyboard is not None:
                keyboard_condition = keyboard_condition[:, self.vae_time_compression_ratio*(N_feats - num_frame_per_block - self.windows_size) + pad_t:, :] # keyboard_condition[:, self.vae_time_compression_ratio*(start_frame - self.windows_size) + pad_t:start_frame * self.vae_time_compression_ratio + pad_t,:]
                keyboard_condition = self.keyboard_embed(keyboard_condition)
                group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(num_frame_per_block)]
            else:
                keyboard_condition = self.keyboard_embed(keyboard_condition)
                group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
            group_keyboard = torch.stack(group_keyboard, dim = 1) # B F RW C
            group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0],group_keyboard.shape[1],-1))
            # apply cross attn
            mouse_q, _ = self.mouse_attn_q(hidden_states)
            keyboard_kv, _ = self.keyboard_attn_kv(group_keyboard)

            B, L, HD = mouse_q.shape
            D = HD // self.heads_num
            q = mouse_q.view(B, L, self.heads_num, D)

            B, L, KHD = keyboard_kv.shape
            k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(2, 0, 1, 3, 4)
        
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            
            q = self.key_attn_q_norm(q).to(v)
            k = self.key_attn_k_norm(k).to(v)
            S = th * tw
            assert S == 880
            # position embed 
            if use_rope_keyboard: 
                B, TS, H, D = q.shape
                T_ = TS // S 
                q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
                q, k = apply_rotary_emb(q, k, freqs_cis, start_offset = start_frame,head_first=False)

                k1, k2, k3, k4 = k.shape
                k = k.expand(S, k2, k3, k4)
                v = v.expand(S, k2, k3, k4)


                if is_causal:
                    if kv_cache_keyboard is None:
                        assert q.shape[0] == k.shape[0] and q.shape[0] % 880 == 0 

                        padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                        padded_q = torch.cat(
                            [q,
                                torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                            device=q.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_k = torch.cat(
                            [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                    device=k.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_v = torch.cat(
                            [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                            device=v.device, dtype=v.dtype)],
                            dim=1
                        )
                        attn = flex_attention(
                            query=padded_q.transpose(2, 1), # after: B, HW, F, C
                            key=padded_k.transpose(2, 1),
                            value=padded_v.transpose(2, 1),
                            block_mask=block_mask_keyboard
                        )[:, :, :-padded_length].transpose(2, 1)
                    else:
                        current_start = start_frame
                        current_end = current_start + k.shape[1]
                        assert k.shape[1] == num_frame_per_block
                        sink_size = 0
                        max_attention_size = self.local_attn_size
                        sink_tokens = sink_size * 1
                        kv_cache_size = kv_cache_keyboard["k"].shape[1]
                        num_new_tokens = k.shape[1]

                        if (current_end > kv_cache_keyboard["global_end_index"].item()) and (
                            num_new_tokens + kv_cache_keyboard["local_end_index"].item() > kv_cache_size):
                            num_evicted_tokens = num_new_tokens + kv_cache_keyboard["local_end_index"].item() - kv_cache_size
                            num_rolled_tokens = kv_cache_keyboard["local_end_index"].item() - num_evicted_tokens - sink_tokens
                            kv_cache_keyboard["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_keyboard["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            kv_cache_keyboard["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_keyboard["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            # Insert the new keys/values at the end
                            local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - \
                                kv_cache_keyboard["global_end_index"].item() - num_evicted_tokens
                            local_start_index = local_end_index - num_new_tokens
                        else:
                            local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - kv_cache_keyboard["global_end_index"].item()
                            local_start_index = local_end_index - num_new_tokens
                        assert k.shape[0] == 880 # BS == 1 or the cache should not be saved/ load method should be modified
                        kv_cache_keyboard["k"][:, local_start_index:local_end_index] = k[:1]
                        kv_cache_keyboard["v"][:, local_start_index:local_end_index] = v[:1]

                        attn = self.keyboard_attn_layer(
                            q,
                            kv_cache_keyboard["k"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                            kv_cache_keyboard["v"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                        )

                        kv_cache_keyboard["global_end_index"].fill_(current_end)
                        kv_cache_keyboard["local_end_index"].fill_(local_end_index)
                else:
                    attn = self.keyboard_attn_layer(q, k, v)
                attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
            else:
                if is_causal:
                    if kv_cache_keyboard is None:
                        
                        padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                        padded_q = torch.cat(
                            [q,
                                torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                            device=q.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_k = torch.cat(
                            [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                    device=k.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_v = torch.cat(
                            [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                            device=v.device, dtype=v.dtype)],
                            dim=1
                        )
                        attn = flex_attention(
                            query=padded_q.transpose(2, 1), # after: B, HW, F, C
                            key=padded_k.transpose(2, 1),
                            value=padded_v.transpose(2, 1),
                            block_mask=block_mask_keyboard
                        )[:, :, :-padded_length].transpose(2, 1)
                    else:
                        current_start = start_frame
                        current_end = current_start + k.shape[1]
                        assert k.shape[1] == num_frame_per_block
                        sink_size = 0
                        local_attn_size = self.local_attn_size
                        max_attention_size = self.local_attn_size
                        sink_tokens = sink_size * 1
                        kv_cache_size = kv_cache_keyboard["k"].shape[1]
                        num_new_tokens = k.shape[1]


                        if (current_end > kv_cache_keyboard["global_end_index"].item()) and (
                            num_new_tokens + kv_cache_keyboard["local_end_index"].item() > kv_cache_size):
                            num_evicted_tokens = num_new_tokens + kv_cache_keyboard["local_end_index"].item() - kv_cache_size
                            num_rolled_tokens = kv_cache_keyboard["local_end_index"].item() - num_evicted_tokens - sink_tokens
                            kv_cache_keyboard["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_keyboard["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            kv_cache_keyboard["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_keyboard["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            # Insert the new keys/values at the end
                            local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - \
                                kv_cache_keyboard["global_end_index"].item() - num_evicted_tokens
                            local_start_index = local_end_index - num_new_tokens

                            
                        else:
                            local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - kv_cache_keyboard["global_end_index"].item()
                            local_start_index = local_end_index - num_new_tokens
                        kv_cache_keyboard["k"][:, local_start_index:local_end_index] = k
                        kv_cache_keyboard["v"][:, local_start_index:local_end_index] = v
                        attn = self.keyboard_attn_layer(
                            q,
                            kv_cache_keyboard["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                            kv_cache_keyboard["v"][:, max(0, local_end_index - max_attention_size):local_end_index],
                        )
                        kv_cache_keyboard["global_end_index"].fill_(current_end)
                        kv_cache_keyboard["local_end_index"].fill_(local_end_index)
                else:
                    attn = self.keyboard_attn_layer(q, k, v)
                attn = rearrange(attn, 'B L H D -> B L (H D)')
            attn, _ = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn
        return hidden_states