# SPDX-License-Identifier: Apache-2.0
# Adapted from Matrix-Game: https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/action_module.py

from einops import rearrange
import torch
import torch.nn as nn
import math
from torch.nn.attention.flex_attention import flex_attention, BlockMask

from fastvideo.attention import LocalAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import FP32LayerNorm, RMSNorm
from fastvideo.layers.rotary_embedding import (
    get_nd_rotary_pos_embed as _fv_get_nd_rotary_pos_embed,
    _apply_rotary_emb,
)
from fastvideo.platforms import AttentionBackendEnum


DISABLE_COMPILE = False
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)


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
    start_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = xq.shape[1]

    # Slice frequencies based on offset
    cos = freqs_cos[start_offset : start_offset + seq_len]  # [S, D]
    sin = freqs_sin[start_offset : start_offset + seq_len]  # [S, D]

    # Move to device
    cos = cos.to(xq.device)
    sin = sin.to(xq.device)

    # Convert from [S, D] (interleaved) back to [S, D/2]
    cos_half = cos[:, ::2]  # [S, D/2]
    sin_half = sin[:, ::2]  # [S, D/2]

    # xq/xk are [B, S, H, D], need to reshape for each batch
    B, S, H, D = xq.shape

    xq_out = _apply_rotary_emb(xq, cos_half, sin_half, is_neox_style=False)
    xk_out = _apply_rotary_emb(xk, cos_half, sin_half, is_neox_style=False)

    return xq_out, xk_out


def _padding_q_k_v(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    return torch.cat(
        [
            tensor,
            torch.zeros(
                [
                    tensor.shape[0],
                    padded_length,
                    tensor.shape[2],
                    tensor.shape[3],
                ],
                device=tensor.device,
                dtype=tensor.dtype,
            ),
        ],
        dim=1,
    )


def _update_kv_cache_and_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: dict[str, torch.Tensor],
    attn_layer: LocalAttention,
    start_frame: int,
    num_frame_per_block: int,
    local_attn_size: int,
    *,
    use_k_for_num_tokens: bool = False,
    store_first_only: bool = False,
    repeat_factor: int | None = None,
) -> torch.Tensor:
    """
    Update KV cache with new tokens and perform attention with cached values.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        kv_cache: Dictionary containing cache tensors and indices
        attn_layer: Attention layer to use
        start_frame: Starting frame index
        num_frame_per_block: Number of frames per block
        local_attn_size: Maximum attention window size
        use_k_for_num_tokens: If True, use k.shape[1] for num_new_tokens, else use q.shape[1]
        store_first_only: If True, only store k[:1] and v[:1] in cache (for keyboard with rope)
        repeat_factor: If provided, repeat cached k,v by this factor when retrieving (for keyboard with rope)

    Returns:
        Attention output tensor
    """
    current_start = start_frame
    current_end = current_start + (
        k.shape[1] if use_k_for_num_tokens else q.shape[1]
    )

    assert (
        k.shape[1] if use_k_for_num_tokens else q.shape[1]
    ) == num_frame_per_block

    sink_size = 0
    max_attention_size = local_attn_size
    sink_tokens = sink_size * 1
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = k.shape[1] if use_k_for_num_tokens else q.shape[1]

    # Check if we need to evict tokens
    if (current_end > kv_cache["global_end_index"].item()) and (
        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
    ):
        num_evicted_tokens = (
            num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        )
        num_rolled_tokens = (
            kv_cache["local_end_index"].item()
            - num_evicted_tokens
            - sink_tokens
        )
        # Roll k cache
        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
            kv_cache["k"][
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        )
        # Roll v cache
        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
            kv_cache["v"][
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        )
        # Calculate indices with eviction adjustment
        local_end_index = (
            kv_cache["local_end_index"].item()
            + current_end
            - kv_cache["global_end_index"].item()
            - num_evicted_tokens
        )
        local_start_index = local_end_index - num_new_tokens
    else:
        # Calculate indices without eviction
        local_end_index = (
            kv_cache["local_end_index"].item()
            + current_end
            - kv_cache["global_end_index"].item()
        )
        local_start_index = local_end_index - num_new_tokens

    # Store new k, v in cache
    if store_first_only:
        kv_cache["k"][:, local_start_index:local_end_index] = k[:1]
        kv_cache["v"][:, local_start_index:local_end_index] = v[:1]
    else:
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v

    # Retrieve from cache and perform attention
    cache_start = max(0, local_end_index - max_attention_size)
    cached_k = kv_cache["k"][:, cache_start:local_end_index]
    cached_v = kv_cache["v"][:, cache_start:local_end_index]

    if repeat_factor is not None:
        cached_k = cached_k.repeat(repeat_factor, 1, 1, 1)
        cached_v = cached_v.repeat(repeat_factor, 1, 1, 1)

    attn = attn_layer(q, cached_k, cached_v)

    # Update indices
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    return attn


class ActionModule(nn.Module):
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
        local_attn_size=6,
        blocks: list | None = None,
    ):
        super().__init__()
        # Initialize mutable defaults
        patch_size = patch_size if patch_size is not None else [1, 2, 2]
        rope_dim_list = (
            rope_dim_list if rope_dim_list is not None else [8, 28, 28]
        )
        mouse_qk_dim_list = (
            mouse_qk_dim_list if mouse_qk_dim_list is not None else [8, 28, 28]
        )
        blocks = blocks if blocks is not None else []
        self.local_attn_size = local_attn_size
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
                RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.img_attn_k_norm = (
                RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.proj_mouse = ReplicatedLinear(
                c, img_hidden_size, bias=qkv_bias
            )

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                RMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.key_attn_k_norm = (
                RMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
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
        c = x.shape[2] // patch_size  # self.unpatchify_channels
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
        start_offset=0,
    ):
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length + start_offset, height, width]

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
        *,
        is_causal: bool,
        kv_cache_mouse: dict[str, torch.Tensor] | None,
        pad_t: int,
        num_frame_per_block: int,
        block_mask_mouse: BlockMask | None,
        start_frame: int,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        N_feats: int,
        B: int,
        C: int,
        tt: int,
        th: int,
        tw: int,
    ) -> torch.Tensor:
        pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
        mouse_condition = torch.cat([pad, mouse_condition], dim=1)
        if is_causal and kv_cache_mouse is not None:
            mouse_condition = mouse_condition[
                :,
                self.vae_time_compression_ratio
                * (N_feats - num_frame_per_block - self.windows_size)
                + pad_t :,
                :,
            ]
            group_mouse = [
                mouse_condition[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size)
                    + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            local_num_frames = tt
            group_mouse = [
                mouse_condition[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size)
                    + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(local_num_frames)
            ]

        group_mouse = torch.stack(group_mouse, dim=1)
        actual_num_frames = group_mouse.shape[
            1
        ]  # Use actual stacked frame count

        S = th * tw
        group_mouse = group_mouse.unsqueeze(-1).expand(
            B, actual_num_frames, pad_t, C, S
        )
        group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(
            B * S, actual_num_frames, pad_t * C
        )

        group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)
        group_mouse = self.mouse_mlp(group_mouse)
        # qkv
        mouse_qkv, _ = self.t_qkv(group_mouse)
        q, k, v = rearrange(
            mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )  # BHW F H C
        q = self.img_attn_q_norm(q).to(v)
        k = self.img_attn_k_norm(k).to(v)
        # rope embd

        # freqs_cis = (self.freqs_cos, self.freqs_sin)

        q, k = _apply_rotary_emb_qk(
            q, k, freqs_cis[0], freqs_cis[1], start_offset=start_frame
        )
        ## TODO: adding cache here
        if is_causal:
            if kv_cache_mouse is None:
                assert (
                    q.shape[0] == k.shape[0] and q.shape[0] % S == 0
                )  # == 880, f"{q.shape[0]},{k.shape[0]}"
                padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                padded_q = _padding_q_k_v(q, padded_length)
                padded_k = _padding_q_k_v(k, padded_length)
                padded_v = _padding_q_k_v(v, padded_length)
                attn = flex_attention(
                    query=padded_q.transpose(2, 1),  # after: B, HW, F, C
                    key=padded_k.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask_mouse,
                )[:, :, :-padded_length].transpose(2, 1)
            else:
                attn = _update_kv_cache_and_attend(
                    q,
                    k,
                    v,
                    kv_cache_mouse,
                    self.mouse_attn_layer,
                    start_frame,
                    num_frame_per_block,
                    self.local_attn_size,
                    use_k_for_num_tokens=False,
                )
        else:
            attn = self.mouse_attn_layer(q, k, v)
        # Compute cu_squlens and max_seqlen for flash attention
        # qk norm
        attn = rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
        attn, _ = self.proj_mouse(attn)
        return attn

    def _forward_keyboard(
        self,
        hidden_states: torch.Tensor,
        keyboard_condition: torch.Tensor,
        *,
        is_causal: bool,
        use_rope_keyboard: bool,
        kv_cache_keyboard: dict[str, torch.Tensor] | None,
        pad_t: int,
        num_frame_per_block: int,
        block_mask_keyboard: BlockMask | None,
        start_frame: int,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        N_feats: int,
        B: int,
        tt: int,
        th: int,
        tw: int,
    ) -> torch.Tensor:
        pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
        keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
        if is_causal and kv_cache_keyboard is not None:
            keyboard_condition = keyboard_condition[
                :,
                self.vae_time_compression_ratio
                * (N_feats - num_frame_per_block - self.windows_size)
                + pad_t :,
                :,
            ]  # keyboard_condition[:, self.vae_time_compression_ratio*(start_frame - self.windows_size) + pad_t:start_frame * self.vae_time_compression_ratio + pad_t,:]
            keyboard_condition = self.keyboard_embed(keyboard_condition)
            group_keyboard = [
                keyboard_condition[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size)
                    + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            keyboard_condition = self.keyboard_embed(keyboard_condition)
            local_num_frames = tt
            group_keyboard = [
                keyboard_condition[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size)
                    + pad_t : i * self.vae_time_compression_ratio + pad_t,
                    :,
                ]
                for i in range(local_num_frames)
            ]
        group_keyboard = torch.stack(group_keyboard, dim=1)  # B F RW C
        group_keyboard = group_keyboard.reshape(
            shape=(group_keyboard.shape[0], group_keyboard.shape[1], -1)
        )
        # apply cross attn
        mouse_q, _ = self.mouse_attn_q(hidden_states)
        keyboard_kv, _ = self.keyboard_attn_kv(group_keyboard)

        B, L, HD = mouse_q.shape
        D = HD // self.heads_num
        q = mouse_q.view(B, L, self.heads_num, D)

        B, L, KHD = keyboard_kv.shape
        k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(
            2, 0, 1, 3, 4
        )

        # Compute cu_squlens and max_seqlen for flash attention
        # qk norm

        q = self.key_attn_q_norm(q).to(v)
        k = self.key_attn_k_norm(k).to(v)
        S = th * tw
        # assert S == 880
        # position embed
        if use_rope_keyboard:
            B, TS, H, D = q.shape
            T_ = TS // S
            q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
            q, k = _apply_rotary_emb_qk(
                q, k, freqs_cis[0], freqs_cis[1], start_offset=start_frame
            )

            k = k.repeat_interleave(S, dim=0)
            v = v.repeat_interleave(S, dim=0)

            if is_causal:
                if kv_cache_keyboard is None:
                    assert q.shape[0] == k.shape[0] and q.shape[0] % S == 0

                    padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                    padded_q = _padding_q_k_v(q, padded_length)
                    padded_k = _padding_q_k_v(k, padded_length)
                    padded_v = _padding_q_k_v(v, padded_length)
                    attn = flex_attention(
                        query=padded_q.transpose(2, 1),  # after: B, HW, F, C
                        key=padded_k.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask_keyboard,
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    assert (
                        k.shape[0] == S
                    )  # BS == 1 or the cache should not be saved/ load method should be modified
                    attn = _update_kv_cache_and_attend(
                        q,
                        k,
                        v,
                        kv_cache_keyboard,
                        self.keyboard_attn_layer,
                        start_frame,
                        num_frame_per_block,
                        self.local_attn_size,
                        use_k_for_num_tokens=True,
                        store_first_only=True,
                        repeat_factor=S,
                    )
            else:
                attn = self.keyboard_attn_layer(q, k, v)
            attn = rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)
        else:
            if is_causal:
                if kv_cache_keyboard is None:
                    padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                    padded_q = _padding_q_k_v(q, padded_length)
                    padded_k = _padding_q_k_v(k, padded_length)
                    padded_v = _padding_q_k_v(v, padded_length)
                    attn = flex_attention(
                        query=padded_q.transpose(2, 1),  # after: B, HW, F, C
                        key=padded_k.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask_keyboard,
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    attn = _update_kv_cache_and_attend(
                        q,
                        k,
                        v,
                        kv_cache_keyboard,
                        self.keyboard_attn_layer,
                        start_frame,
                        num_frame_per_block,
                        self.local_attn_size,
                        use_k_for_num_tokens=True,
                    )
            else:
                attn = self.keyboard_attn_layer(q, k, v)
            attn = rearrange(attn, "B L H D -> B L (H D)")
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
        block_mask_mouse: BlockMask | None = None,
        block_mask_keyboard: BlockMask | None = None,
        is_causal: bool = False,
        kv_cache_mouse: dict[str, torch.Tensor] | None = None,
        kv_cache_keyboard: dict[str, torch.Tensor] | None = None,
        start_frame: int = 0,
        use_rope_keyboard: bool = True,
        num_frame_per_block: int = 3,
    ):
        """
        hidden_states: B, tt*th*tw, C
        mouse_condition: B, N_frames, C1
        keyboard_condition: B, N_frames, C2
        """
        assert use_rope_keyboard

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
            (N_frames - 1) + self.vae_time_compression_ratio
        ) % self.vae_time_compression_ratio == 0
        N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1

        # Lazy initialization of freqs on first forward pass
        if self._freqs_cos is None or self._freqs_sin is None:
            self._freqs_cos, self._freqs_sin = self.get_rotary_pos_embed(
                7500,
                self.patch_size[1],
                self.patch_size[2],
                64,
                self.mouse_qk_dim_list,
                start_offset=0,
            )

        # Defined freqs_cis early so it's available for both mouse and keyboard
        freqs_cis = (self._freqs_cos, self._freqs_sin)

        if is_causal:
            assert (N_feats == tt and kv_cache_mouse is None) or (
                (N_frames - 1) // self.vae_time_compression_ratio + 1
                == start_frame + num_frame_per_block
            )
        # For non-causal (training), we trust that the caller provides correctly shaped inputs

        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(
                x, "B (T S) C -> (B S) T C", T=tt, S=th * tw
            )  # 65*272*480 -> 17*(272//16)*(480//16) -> 8670
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x
        # padding

        pad_t = self.vae_time_compression_ratio * self.windows_size
        if self.enable_mouse and mouse_condition is not None:
            attn = self._forward_mouse(
                hidden_states,
                mouse_condition,
                is_causal=is_causal,
                kv_cache_mouse=kv_cache_mouse,
                pad_t=pad_t,
                num_frame_per_block=num_frame_per_block,
                block_mask_mouse=block_mask_mouse,
                start_frame=start_frame,
                freqs_cis=freqs_cis,
                N_feats=N_feats,
                B=B,
                C=C,
                tt=tt,
                th=th,
                tw=tw,
            )
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)

            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            attn = self._forward_keyboard(
                hidden_states,
                keyboard_condition,
                is_causal=is_causal,
                use_rope_keyboard=use_rope_keyboard,
                kv_cache_keyboard=kv_cache_keyboard,
                pad_t=pad_t,
                num_frame_per_block=num_frame_per_block,
                block_mask_keyboard=block_mask_keyboard,
                start_frame=start_frame,
                freqs_cis=freqs_cis,
                N_feats=N_feats,
                B=B,
                tt=tt,
                th=th,
                tw=tw,
            )
            hidden_states = hidden_states + attn
        return hidden_states
