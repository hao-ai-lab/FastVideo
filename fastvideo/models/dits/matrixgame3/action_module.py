# SPDX-License-Identifier: Apache-2.0
# Adapted from Matrix-Game-3.0: https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/action_module.py

from einops import rearrange
import torch
import torch.nn as nn

from fastvideo.attention import LocalAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import FP32LayerNorm
from fastvideo.layers.rotary_embedding import (
    get_nd_rotary_pos_embed as _fv_get_nd_rotary_pos_embed,
    _apply_rotary_emb,
)
from fastvideo.platforms import AttentionBackendEnum


class WanRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def _get_nd_rotary_pos_embed_matrixgame(
    rope_dim_list,
    rope_sizes,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
):
    cos, sin = _fv_get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=theta,
        theta_rescale_factor=theta_rescale_factor,
        dtype=torch.float32,
    )
    # convert from [S, D/2] to [S, D] format
    cos = cos.repeat_interleave(2, dim=1)
    sin = sin.repeat_interleave(2, dim=1)
    return cos, sin


def _apply_rotary_emb_qk(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = xq.shape[1]
    cos = freqs_cos[:seq_len]
    sin = freqs_sin[:seq_len]
    cos_half = cos[:, ::2]
    sin_half = sin[:, ::2]
    xq_out = _apply_rotary_emb(xq, cos_half, sin_half, is_neox_style=False)
    xk_out = _apply_rotary_emb(xk, cos_half, sin_half, is_neox_style=False)
    return xq_out, xk_out


class MatrixGame3ActionModule(nn.Module):
    """
    action module from https://arxiv.org/pdf/2501.08325
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
        patch_size: list | None = None,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list | None = None,
        rope_theta=256,
        mouse_qk_dim_list: list | None = None,
        enable_mouse=True,
        enable_keyboard=True,
        blocks: list | None = None,
    ):
        super().__init__()
        patch_size = patch_size if patch_size is not None else [1, 2, 2]
        rope_dim_list = (
            rope_dim_list if rope_dim_list is not None else [8, 28, 28]
        )
        mouse_qk_dim_list = (
            mouse_qk_dim_list if mouse_qk_dim_list is not None else [8, 28, 28]
        )
        blocks = blocks if blocks is not None else []
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = nn.Sequential(
                nn.Linear(keyboard_dim_in, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = nn.Sequential(
                nn.Linear(
                    mouse_dim_in * vae_time_compression_ratio * windows_size
                    + img_hidden_size,
                    c,
                    bias=True,
                ),
                nn.GELU(approximate="tanh"),
                nn.Linear(c, c),
                FP32LayerNorm(c, elementwise_affine=True),
            )

            head_dim = c // heads_num
            self.t_qkv = ReplicatedLinear(c, c * 3, bias=qkv_bias)
            self.img_attn_q_norm = (
                WanRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.img_attn_k_norm = (
                WanRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.proj_mouse = ReplicatedLinear(
                c, img_hidden_size, bias=qkv_bias
            )

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.key_attn_k_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
            )

            self.mouse_attn_q = ReplicatedLinear(
                img_hidden_size, keyboard_hidden_dim, bias=qkv_bias
            )
            self.keyboard_attn_kv = ReplicatedLinear(
                hidden_size * windows_size * vae_time_compression_ratio,
                keyboard_hidden_dim * 2,
                bias=qkv_bias,
            )
            self.proj_keyboard = ReplicatedLinear(
                keyboard_hidden_dim, img_hidden_size, bias=qkv_bias
            )

        self.mouse_attn_layer = (
            LocalAttention(
                num_heads=heads_num,
                head_size=mouse_hidden_dim // heads_num,
                causal=False,
                supported_attention_backends=(
                    AttentionBackendEnum.FLASH_ATTN,
                    AttentionBackendEnum.TORCH_SDPA,
                ),
            )
            if self.enable_mouse
            else None
        )

        self.keyboard_attn_layer = (
            LocalAttention(
                num_heads=heads_num,
                head_size=keyboard_hidden_dim // heads_num,
                causal=False,
                supported_attention_backends=(
                    AttentionBackendEnum.FLASH_ATTN,
                    AttentionBackendEnum.TORCH_SDPA,
                ),
            )
            if self.enable_keyboard
            else None
        )

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size
        # Lazy initialization: freqs will be created on first forward pass
        self._freqs_cos = None
        self._freqs_sin = None

    def patchify(self, x, patch_size):
        """
        x : (N C T H W)
        """
        pt, ph, pw = self.patch_size
        t, h, w = x.shape[2] // pt, x.shape[3] // ph, x.shape[4] // pw
        c = x.shape[1]
        x = x.reshape(shape=(x.shape[0], c, t, pt, h, ph, w, pw))
        x = torch.einsum("nctohpwq->nthwcopq", x)
        x = x.reshape(shape=(x.shape[0], t * h * w, c * pt * ph * pw))
        return x

    def unpatchify(self, x, t, h, w, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = x.shape[2] // patch_size
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def get_rotary_pos_embed(
        self,
        video_length,
        height,
        width,
        head_dim,
        rope_dim_list=None,
    ):
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length, height, width]

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
            rope_sizes = [1] * (
                target_ndim - len(rope_sizes)
            ) + rope_sizes  # time axis

        if rope_dim_list is None:
            rope_dim_list = [
                head_dim // target_ndim for _ in range(target_ndim)
            ]
        assert sum(rope_dim_list) == head_dim, (
            "sum(rope_dim_list) should equal to head_dim of attention layer"
        )
        # Use Matrix-Game wrapper for FastVideo's function
        freqs_cos, freqs_sin = _get_nd_rotary_pos_embed_matrixgame(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            theta_rescale_factor=1,
        )
        return freqs_cos[
            -video_length
            * rope_sizes[1]
            * rope_sizes[2]
            // self.patch_size[0] :
        ], freqs_sin[
            -video_length
            * rope_sizes[1]
            * rope_sizes[2]
            // self.patch_size[0] :
        ]

    def _forward_mouse(
        self,
        hidden_states: torch.Tensor,
        mouse_condition: torch.Tensor,
        mouse_cond_memory: torch.Tensor | None,
        *,
        pad_t: int,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        N_feats: int,
        B: int,
        C: int,
        th: int,
        tw: int,
    ) -> torch.Tensor:
        pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
        mouse_condition = torch.cat([pad, mouse_condition], dim=1)
        group_mouse = [
            mouse_condition[
                :,
                self.vae_time_compression_ratio * (i - self.windows_size)
                + pad_t : i * self.vae_time_compression_ratio + pad_t,
                :,
            ]
            for i in range(N_feats)
        ]
        group_mouse = torch.stack(group_mouse, dim=1)
        mem_len = 0
        if mouse_cond_memory is not None:
            mem_len = mouse_cond_memory.shape[1]
            mem = mouse_cond_memory.to(device=group_mouse.device, dtype=group_mouse.dtype)
            mem = mem.unsqueeze(2).expand(-1, -1, pad_t, -1)
            group_mouse = torch.cat([mem, group_mouse], dim=1)
        actual_num_frames = group_mouse.shape[1]

        S = th * tw
        group_mouse = group_mouse.unsqueeze(-1).expand(
            B, actual_num_frames, pad_t, C, S
        )
        group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(
            B * S, actual_num_frames, pad_t * C
        )

        group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)
        group_mouse = self.mouse_mlp(group_mouse)
        mouse_qkv, _ = self.t_qkv(group_mouse)
        q, k, v = rearrange(
            mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        q = self.img_attn_q_norm(q).to(v)
        k = self.img_attn_k_norm(k).to(v)
        if mem_len > 0:
            q_mem, k_mem = _apply_rotary_emb_qk(q[:, :mem_len], k[:, :mem_len], freqs_cis[0], freqs_cis[1])
            q_pred, k_pred = _apply_rotary_emb_qk(q[:, mem_len:], k[:, mem_len:], freqs_cis[0], freqs_cis[1])
            q = torch.cat([q_mem, q_pred], dim=1)
            k = torch.cat([k_mem, k_pred], dim=1)
        else:
            q, k = _apply_rotary_emb_qk(q, k, freqs_cis[0], freqs_cis[1])

        attn = self.mouse_attn_layer(q, k, v)
        attn = rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
        attn, _ = self.proj_mouse(attn)
        return attn

    def _forward_keyboard(
        self,
        hidden_states: torch.Tensor,
        keyboard_condition: torch.Tensor,
        keyboard_cond_memory: torch.Tensor | None,
        *,
        pad_t: int,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        N_feats: int,
        B: int,
        th: int,
        tw: int,
    ) -> torch.Tensor:
        pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
        keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
        keyboard_condition = self.keyboard_embed(keyboard_condition)
        group_keyboard = [
            keyboard_condition[
                :,
                self.vae_time_compression_ratio * (i - self.windows_size)
                + pad_t : i * self.vae_time_compression_ratio + pad_t,
                :,
            ]
            for i in range(N_feats)
        ]
        group_keyboard = torch.stack(group_keyboard, dim=1)
        mem_len = 0
        if keyboard_cond_memory is not None:
            mem_len = keyboard_cond_memory.shape[1]
            k_mem = keyboard_cond_memory.to(
                device=group_keyboard.device, dtype=group_keyboard.dtype
            )
            k_mem = self.keyboard_embed(k_mem)
            k_mem = k_mem.unsqueeze(2).expand(-1, -1, pad_t, -1)
            group_keyboard = torch.cat([k_mem, group_keyboard], dim=1)
        group_keyboard = group_keyboard.reshape(
            shape=(group_keyboard.shape[0], group_keyboard.shape[1], -1)
        )

        mouse_q, _ = self.mouse_attn_q(hidden_states)
        keyboard_kv, _ = self.keyboard_attn_kv(group_keyboard)

        B, L, HD = mouse_q.shape
        D = HD // self.heads_num
        q = mouse_q.view(B, L, self.heads_num, D)

        B, L, KHD = keyboard_kv.shape
        k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(
            2, 0, 1, 3, 4
        )

        q = self.key_attn_q_norm(q).to(v)
        k = self.key_attn_k_norm(k).to(v)
        S = th * tw

        B, TS, H, D = q.shape
        T_ = TS // S
        q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
        if mem_len > 0:
            q_mem, k_mem_r = _apply_rotary_emb_qk(q[:, :mem_len], k[:, :mem_len], freqs_cis[0], freqs_cis[1])
            q_pred, k_pred_r = _apply_rotary_emb_qk(q[:, mem_len:], k[:, mem_len:], freqs_cis[0], freqs_cis[1])
            q = torch.cat([q_mem, q_pred], dim=1)
            k = torch.cat([k_mem_r, k_pred_r], dim=1)
        else:
            q, k = _apply_rotary_emb_qk(q, k, freqs_cis[0], freqs_cis[1])

        k = k.repeat_interleave(S, dim=0)
        v = v.repeat_interleave(S, dim=0)

        attn = self.keyboard_attn_layer(q, k, v)
        attn = rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)
        attn, _ = self.proj_keyboard(attn)
        return attn

    def forward(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        mouse_condition: torch.Tensor | None = None,
        keyboard_condition: torch.Tensor | None = None,
        mouse_cond_memory: torch.Tensor | None = None,
        keyboard_cond_memory: torch.Tensor | None = None,
    ):
        """
        hidden_states: B, tt*th*tw, C
        mouse_condition: B, N_frames, C1
        keyboard_condition: B, N_frames, C2
        """
        target_device = x.device
        target_dtype = x.dtype
        if mouse_condition is not None:
            mouse_condition = mouse_condition.to(
                device=target_device, dtype=target_dtype
            )
        if keyboard_condition is not None:
            keyboard_condition = keyboard_condition.to(
                device=target_device, dtype=target_dtype
            )
        else:
            return x

        B, N_frames, C = keyboard_condition.shape
        assert tt * th * tw == x.shape[1]
        assert (
            ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
            or (N_frames % self.vae_time_compression_ratio == 0)
        )
        if ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0:
            N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        else:
            N_feats = N_frames // self.vae_time_compression_ratio

        # Lazy initialization of freqs on first forward pass
        if self._freqs_cos is None or self._freqs_sin is None:
            fc, fs = self.get_rotary_pos_embed(
                7500,
                self.patch_size[1],
                self.patch_size[2],
                64,
                self.mouse_qk_dim_list,
            )
            self._freqs_cos = fc.to(x.device)
            self._freqs_sin = fs.to(x.device)

        freqs_cis = (self._freqs_cos, self._freqs_sin)

        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(
                x, "B (T S) C -> (B S) T C", T=tt, S=th * tw
            )
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x

        pad_t = self.vae_time_compression_ratio * self.windows_size
        if self.enable_mouse and mouse_condition is not None:
            attn = self._forward_mouse(
                hidden_states,
                mouse_condition,
                mouse_cond_memory,
                pad_t=pad_t,
                freqs_cis=freqs_cis,
                N_feats=N_feats,
                B=B,
                C=C,
                th=th,
                tw=tw,
            )
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)
            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            attn = self._forward_keyboard(
                hidden_states,
                keyboard_condition,
                keyboard_cond_memory,
                pad_t=pad_t,
                freqs_cis=freqs_cis,
                N_feats=N_feats,
                B=B,
                th=th,
                tw=tw,
            )
            hidden_states = hidden_states + attn
        return hidden_states
