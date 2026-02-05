# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

PYTEST_DONT_REWRITE = True

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
from fastvideo.forward_context import get_forward_context

flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import (
    PatchEmbed,
    TimestepEmbedder,
    ModulateProjection,
)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import (
    WanT2VCrossAttention,
    WanImageEmbedding,
)
from fastvideo.platforms import AttentionBackendEnum, current_platform

from .action_module import ActionModule
from .model import MatrixGameCrossAttention

logger = init_logger(__name__)


def causal_rope_apply(
    x: torch.Tensor, grid_sizes: tuple[int, int, int], freqs: torch.Tensor, start_frame: int = 0
):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    f, h, w = grid_sizes

    for i in range(len(x)):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class CausalMatrixGameTimeImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(
            dim, factor=6, act_layer="silu"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, None, encoder_hidden_states_image


class CausalMatrixGameSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self._freqs_cache = None

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        grid_sizes: tuple[int, int, int],
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ):
        if cache_start is None:
            cache_start = current_start

        # Calculate start_frame for causal mode
        if kv_cache is not None:
            frame_seqlen = int(grid_sizes[1] * grid_sizes[2])
            start_frame = current_start // frame_seqlen
        else:
            start_frame = 0

        if self._freqs_cache is None or self._freqs_cache.device != q.device:
            d = self.head_dim
            self._freqs_cache = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ).to(q.device)

        freqs = self._freqs_cache

        roped_query = causal_rope_apply(
            q, grid_sizes, freqs, start_frame=start_frame
        ).type_as(v)
        roped_key = causal_rope_apply(
            k, grid_sizes, freqs, start_frame=start_frame
        ).type_as(v)

        if kv_cache is None:
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros(
                        [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros(
                        [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            padded_v = torch.cat(
                [
                    v,
                    torch.zeros(
                        [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask,
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            # Calculate frame_seqlen correctly from grid_sizes (single frame token count)
            if grid_sizes is not None:
                frame_seqlen = int(grid_sizes[1] * grid_sizes[2])
            else:
                # Fallback: assume q.shape[1] is single frame (shouldn't happen in causal mode)
                frame_seqlen = q.shape[1]
                logger.warning(
                    "grid_sizes not provided, using q.shape[1] as frame_seqlen"
                )

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen

            # Compute max_attention_size dynamically based on actual frame_seqlen
            max_attention_size = (
                15 * frame_seqlen
                if self.local_attn_size == -1
                else self.local_attn_size * frame_seqlen
            )

            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            if (current_end > kv_cache["global_end_index"]) and (
                num_new_tokens + kv_cache["local_end_index"] > kv_cache_size
            ):
                num_evicted_tokens = (
                    num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
                )
                num_rolled_tokens = (
                    kv_cache["local_end_index"]
                    - num_evicted_tokens
                    - sink_tokens
                )
                kv_cache["k"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["k"][
                    :,
                    sink_tokens + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                kv_cache["v"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["v"][
                    :,
                    sink_tokens + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                local_end_index = (
                    kv_cache["local_end_index"]
                    + current_end
                    - kv_cache["global_end_index"]
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = (
                    kv_cache["local_end_index"]
                    + current_end
                    - kv_cache["global_end_index"]
                )
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            kv_start = max(0, local_end_index - max_attention_size)
            k_for_attn = kv_cache["k"][:, kv_start:local_end_index]
            v_for_attn = kv_cache["v"][:, kv_start:local_end_index]

            x = torch.nn.functional.scaled_dot_product_attention(
                roped_query.transpose(1, 2),
                k_for_attn.transpose(1, 2),
                v_for_attn.transpose(1, 2),
            ).transpose(1, 2)

            kv_cache["global_end_index"] = current_end
            kv_cache["local_end_index"] = local_end_index

        return x


class CausalMatrixGameTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None,
        prefix: str = "",
        action_config: dict | None = None,
        block_idx: int = 0,
    ):
        super().__init__()
        action_config = action_config or {}

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = CausalMatrixGameSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        if added_kv_proj_dim is not None:
            self.attn2 = MatrixGameCrossAttention(
                dim, num_heads, qk_norm=qk_norm, eps=eps
            )
        else:
            self.attn2 = WanT2VCrossAttention(
                dim, num_heads, qk_norm=qk_norm, eps=eps
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        if len(action_config) > 0 and block_idx in action_config.get(
            "blocks", []
        ):
            self.action_model = ActionModule(
                heads_num=action_config["heads_num"],
                hidden_size=action_config["hidden_size"],
                img_hidden_size=action_config["img_hidden_size"],
                mouse_hidden_dim=action_config.get("mouse_hidden_dim", 0),
                keyboard_hidden_dim=action_config["keyboard_hidden_dim"],
                mouse_dim_in=action_config.get("mouse_dim_in", 0),
                keyboard_dim_in=action_config["keyboard_dim_in"],
                enable_mouse=action_config["enable_mouse"],
                enable_keyboard=action_config["enable_keyboard"],
                windows_size=action_config["windows_size"],
                rope_theta=action_config["rope_theta"],
                rope_dim_list=action_config["rope_dim_list"],
                mouse_qk_dim_list=action_config.get(
                    "mouse_qk_dim_list", [8, 28, 28]
                ),
                patch_size=action_config["patch_size"],
                local_attn_size=local_attn_size,
                qk_norm=action_config["qk_norm"],
                qkv_bias=action_config["qkv_bias"],
                vae_time_compression_ratio=action_config[
                    "vae_time_compression_ratio"
                ],
            )
        else:
            self.action_model = None

        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        grid_sizes: tuple[int, int, int],
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        block_mask_mouse: BlockMask | None = None,
        block_mask_keyboard: BlockMask | None = None,
        num_frame_per_block: int = 1,
        use_rope_keyboard: bool = True,
        kv_cache: dict | None = None,
        kv_cache_mouse: dict | None = None,
        kv_cache_keyboard: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)

        assert temb.ndim == 4
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # hidden_states = hidden_states.float()

        e = self.scale_shift_table + temb
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            e.chunk(6, dim=2)
        )

        norm_hidden_states = (
            self.norm1(hidden_states).unflatten(
                dim=1, sizes=(num_frames, frame_seqlen)
            )
            * (1 + scale_msa)
            + shift_msa
        ).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            grid_sizes,
            kv_cache,
            current_start,
            cache_start,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )

        # norm3 weights are loaded into self_attn_residual_norm.norm
        attn_output = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
            crossattn_cache=crossattn_cache,
        )
        # residual connection
        hidden_states = hidden_states + attn_output

        if self.action_model is not None:
            if mouse_cond is not None or keyboard_cond is not None:
                start_frame = current_start // frame_seqlen

                hidden_states = self.action_model(
                    hidden_states,
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    mouse_cond,
                    keyboard_cond,
                    block_mask_mouse,
                    block_mask_keyboard,
                    is_causal=True,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    start_frame=start_frame,
                    use_rope_keyboard=use_rope_keyboard,
                    num_frame_per_block=num_frame_per_block,
                )

        ff_output = self.ffn(
            (
                self.norm1(hidden_states).unflatten(
                    dim=1, sizes=(num_frames, frame_seqlen)
                )
                * (1 + c_scale_msa)
                + c_shift_msa
            ).flatten(1, 2)
        )
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        # Convert back to original dtype before return
        return hidden_states.to(orig_dtype)


_DEFAULT_MATRIXGAME_CONFIG = MatrixGameWanVideoConfig()


class CausalMatrixGameWanModel(BaseDiT):
    supports_action_input = True

    _fsdp_shard_conditions = _DEFAULT_MATRIXGAME_CONFIG._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_MATRIXGAME_CONFIG._compile_conditions
    _supported_attention_backends = (
        _DEFAULT_MATRIXGAME_CONFIG._supported_attention_backends
    )
    param_names_mapping = _DEFAULT_MATRIXGAME_CONFIG.param_names_mapping
    reverse_param_names_mapping = (
        _DEFAULT_MATRIXGAME_CONFIG.reverse_param_names_mapping
    )
    lora_param_names_mapping = (
        _DEFAULT_MATRIXGAME_CONFIG.lora_param_names_mapping
    )

    def __init__(
        self,
        config: MatrixGameWanVideoConfig,
        hf_config: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size

        arch_cfg = getattr(config, "arch_config", None)
        self.local_attn_size = (
            getattr(
                arch_cfg,
                "local_attn_size",
                getattr(config, "local_attn_size", -1),
            )
            if arch_cfg
            else getattr(config, "local_attn_size", -1)
        )
        self.sink_size = (
            getattr(arch_cfg, "sink_size", getattr(config, "sink_size", 0))
            if arch_cfg
            else getattr(config, "sink_size", 0)
        )

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = CausalMatrixGameTimeImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            image_embed_dim=config.image_dim,
        )

        # 2.1. Get action config
        arch_cfg = getattr(config, "arch_config", None)
        self.action_config = (
            getattr(arch_cfg, "action_config", {})
            if arch_cfg is not None
            else {}
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                CausalMatrixGameTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    self.local_attn_size,
                    self.sink_size,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{getattr(config, 'prefix', 'Wan')}.blocks.{i}",
                    action_config=self.action_config,
                    block_idx=i,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

        self.block_mask = None
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True

        self.num_frame_per_block = (
            getattr(
                arch_cfg,
                "num_frames_per_block",
                getattr(config, "num_frame_per_block", 1),
            )
            if arch_cfg
            else getattr(config, "num_frame_per_block", 1)
        )

        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_keyboard(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        total_length2 = num_frames * frame_seqlen
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(
            total_length2 + padded_length2, device=device, dtype=torch.long
        )

        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends2[q_idx])
                    & (kv_idx >= (ends2[q_idx] - local_attn_size))
                ) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(
            attention_mask2,
            B=None,
            H=None,
            Q_LEN=total_length2 + padded_length2,
            KV_LEN=num_frames + padded_length_kv2,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask for keyboard with block size of {num_frame_per_block} frames"
            )

        return block_mask2

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_action(
        device: torch.device | str,
        num_frames: int = 9,
        frame_seqlen: int = 1,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        total_length2 = num_frames * frame_seqlen
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(
            total_length2 + padded_length2, device=device, dtype=torch.long
        )

        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp : tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends2[q_idx])
                    & (kv_idx >= (ends2[q_idx] - local_attn_size))
                ) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(
            attention_mask2,
            B=None,
            H=None,
            Q_LEN=total_length2 + padded_length2,
            KV_LEN=num_frames + padded_length_kv2,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask for action with block size of {num_frame_per_block} frames"
            )

        return block_mask2

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor
        | list[torch.Tensor]
        | None = None,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        kv_cache: dict = None,
        kv_cache_mouse: dict = None,
        kv_cache_keyboard: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        # Extract num_frame_per_block from kwargs
        effective_num_frame_per_block = kwargs.pop(
            "num_frame_per_block", self.num_frame_per_block
        )

        if mouse_cond is not None or keyboard_cond is not None:
            assert (
                self.action_config is not None and len(self.action_config) > 0
            )

        if encoder_hidden_states is not None and not isinstance(
            encoder_hidden_states, torch.Tensor
        ):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        ctx = get_forward_context()
        batch = getattr(ctx, "forward_batch", None)

        if (
            batch is not None
            and getattr(batch, "image_latent", None) is not None
        ):
            image_latent = batch.image_latent
            if (
                isinstance(image_latent, torch.Tensor)
                and image_latent.ndim == 5
                and hidden_states.ndim == 5
            ):
                _, _, num_frames, _, _ = hidden_states.shape
                start = start_frame
                end = start + num_frames
                cond = image_latent
                if cond.shape[2] >= end:
                    cond = cond[:, :, start:end]
                elif cond.shape[2] > start:
                    cond = cond[:, :, start:]
                    pad_frames = num_frames - cond.shape[2]
                    if pad_frames > 0:
                        pad = torch.zeros(
                            cond.shape[0],
                            cond.shape[1],
                            pad_frames,
                            cond.shape[3],
                            cond.shape[4],
                            device=cond.device,
                            dtype=cond.dtype,
                        )
                        cond = torch.cat([cond, pad], dim=2)
                else:
                    pad = torch.zeros(
                        cond.shape[0],
                        cond.shape[1],
                        num_frames,
                        cond.shape[3],
                        cond.shape[4],
                        device=cond.device,
                        dtype=cond.dtype,
                    )
                    cond = pad
                hidden_states = torch.cat(
                    [hidden_states, cond.to(dtype=hidden_states.dtype)], dim=1
                )

        batch_size, num_channels, num_frames, height, width = (
            hidden_states.shape
        )
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (
                post_patch_num_frames * get_sp_world_size(),
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame,
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = (post_patch_num_frames, post_patch_height, post_patch_width)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.dim() == 2:
            timestep = timestep.flatten()

        (
            temb,
            timestep_proj,
            encoder_hidden_states_text,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(
            1, (6, self.hidden_size)
        ).unflatten(dim=0, sizes=(batch_size, post_patch_num_frames))

        if encoder_hidden_states_text is not None:
            if isinstance(encoder_hidden_states_text, list):
                encoder_hidden_states_text = encoder_hidden_states_text[0]
            elif encoder_hidden_states_text.ndim == 2:
                encoder_hidden_states_text = (
                    encoder_hidden_states_text.unsqueeze(0)
                )

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states_text is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states_text],
                    dim=1,
                )
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )
        if self.block_mask_keyboard is None:
            if not self.use_rope_keyboard:
                self.block_mask_keyboard = (
                    self._prepare_blockwise_causal_attn_mask_keyboard(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=post_patch_height * post_patch_width,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                )
            else:
                self.block_mask_keyboard = (
                    self._prepare_blockwise_causal_attn_mask_action(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                )
        if self.block_mask_mouse is None:
            self.block_mask_mouse = (
                self._prepare_blockwise_causal_attn_mask_action(
                    device=hidden_states.device,
                    num_frames=num_frames,
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
            )
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        if kv_cache_mouse is None:
            kv_cache_mouse = [None] * len(self.blocks)
        if kv_cache_keyboard is None:
            kv_cache_keyboard = [None] * len(self.blocks)
        if crossattn_cache is None:
            crossattn_cache = [None] * len(self.blocks)

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index]
                        if kv_cache_mouse
                        else None,
                        "kv_cache_keyboard": kv_cache_keyboard[block_index]
                        if kv_cache_keyboard
                        else None,
                        "crossattn_cache": crossattn_cache[block_index]
                        if crossattn_cache
                        else None,
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "num_frame_per_block": effective_num_frame_per_block,
                    }
                )
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, **kwargs
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index]
                        if kv_cache_mouse
                        else None,
                        "kv_cache_keyboard": kv_cache_keyboard[block_index]
                        if kv_cache_keyboard
                        else None,
                        "crossattn_cache": crossattn_cache[block_index]
                        if crossattn_cache
                        else None,
                        "current_start": current_start,
                        "cache_start": cache_start,
                    }
                )
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    block_mask=self.block_mask,
                    grid_sizes=grid_sizes,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                    block_mask_mouse=self.block_mask_mouse,
                    block_mask_keyboard=self.block_mask_keyboard,
                    num_frame_per_block=effective_num_frame_per_block,
                    use_rope_keyboard=self.use_rope_keyboard,
                    **kwargs,
                )

        temb = temb.unflatten(
            dim=0, sizes=(batch_size, post_patch_num_frames)
        ).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(
            2, dim=2
        )
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = self.unpatchify(hidden_states, grid_sizes)

        return hidden_states

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor
        | list[torch.Tensor]
        | None = None,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mouse_cond is not None or keyboard_cond is not None:
            assert (
                self.action_config is not None and len(self.action_config) > 0
            )

        if encoder_hidden_states is not None and not isinstance(
            encoder_hidden_states, torch.Tensor
        ):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        ctx = get_forward_context()
        batch = getattr(ctx, "forward_batch", None)
        if (
            batch is not None
            and getattr(batch, "image_latent", None) is not None
        ):
            image_latent = batch.image_latent
            if (
                isinstance(image_latent, torch.Tensor)
                and image_latent.ndim == 5
                and hidden_states.ndim == 5
            ):
                _, _, num_frames, _, _ = hidden_states.shape
                cond = image_latent
                if cond.shape[2] >= num_frames:
                    cond = cond[:, :, :num_frames]
                else:
                    pad_frames = num_frames - cond.shape[2]
                    pad = torch.zeros(
                        cond.shape[0],
                        cond.shape[1],
                        pad_frames,
                        cond.shape[3],
                        cond.shape[4],
                        device=cond.device,
                        dtype=cond.dtype,
                    )
                    cond = torch.cat([cond, pad], dim=2)
                hidden_states = torch.cat(
                    [hidden_states, cond.to(dtype=hidden_states.dtype)], dim=1
                )

        batch_size, num_channels, num_frames, height, width = (
            hidden_states.shape
        )
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (
                post_patch_num_frames * get_sp_world_size(),
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )
        if self.block_mask_keyboard is None:
            if not self.use_rope_keyboard:
                self.block_mask_keyboard = (
                    self._prepare_blockwise_causal_attn_mask_keyboard(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=post_patch_height * post_patch_width,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                )
            else:
                self.block_mask_keyboard = (
                    self._prepare_blockwise_causal_attn_mask_action(
                        device=hidden_states.device,
                        num_frames=num_frames,
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                )
        if self.block_mask_mouse is None:
            self.block_mask_mouse = (
                self._prepare_blockwise_causal_attn_mask_action(
                    device=hidden_states.device,
                    num_frames=num_frames,
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )
            )

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.tensor(
            [post_patch_num_frames, post_patch_height, post_patch_width],
            device=hidden_states.device,
        )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.dim() == 2:
            timestep = timestep.flatten()

        (
            temb,
            timestep_proj,
            encoder_hidden_states_text,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(
            1, (6, self.hidden_size)
        ).unflatten(dim=0, sizes=(batch_size, post_patch_num_frames))

        if encoder_hidden_states_text is not None:
            if isinstance(encoder_hidden_states_text, list):
                encoder_hidden_states_text = encoder_hidden_states_text[0]
            elif encoder_hidden_states_text.ndim == 2:
                encoder_hidden_states_text = (
                    encoder_hidden_states_text.unsqueeze(0)
                )

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states_text is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states_text],
                    dim=1,
                )
            else:
                encoder_hidden_states = encoder_hidden_states_image
        else:
            encoder_hidden_states = encoder_hidden_states_text

        kwargs = dict(
            e=timestep_proj,
            grid_sizes=grid_sizes,
            freqs_cis=freqs_cis,
            encoder_hidden_states=encoder_hidden_states,
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block,
        )

        # Initialize kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache if not provided
        kv_cache = kwargs.pop("kv_cache", [None] * len(self.blocks))
        kv_cache_mouse = kwargs.pop("kv_cache_mouse", [None] * len(self.blocks))
        kv_cache_keyboard = kwargs.pop(
            "kv_cache_keyboard", [None] * len(self.blocks)
        )
        crossattn_cache = kwargs.pop(
            "crossattn_cache", [None] * len(self.blocks)
        )

        for i, block in enumerate(self.blocks):
            block_crossattn_cache = None
            if crossattn_cache is not None:
                if crossattn_cache[i] is None:
                    crossattn_cache[i] = {"is_init": False}
                block_crossattn_cache = crossattn_cache[i]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    kv_cache=kv_cache[i],
                    kv_cache_mouse=kv_cache_mouse[i],
                    kv_cache_keyboard=kv_cache_keyboard[i],
                    crossattn_cache=block_crossattn_cache,
                    block_mask=self.block_mask,
                    grid_sizes=grid_sizes,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                    block_mask_mouse=self.block_mask_mouse,
                    block_mask_keyboard=self.block_mask_keyboard,
                    num_frame_per_block=self.num_frame_per_block,
                    use_rope_keyboard=self.use_rope_keyboard,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    kv_cache=kv_cache[i],
                    kv_cache_mouse=kv_cache_mouse[i],
                    kv_cache_keyboard=kv_cache_keyboard[i],
                    crossattn_cache=block_crossattn_cache,
                    block_mask=self.block_mask,
                    grid_sizes=grid_sizes,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                    block_mask_mouse=self.block_mask_mouse,
                    block_mask_keyboard=self.block_mask_keyboard,
                    num_frame_per_block=self.num_frame_per_block,
                    use_rope_keyboard=self.use_rope_keyboard,
                )

        temb = temb.unflatten(
            dim=0, sizes=(batch_size, post_patch_num_frames)
        ).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(
            2, dim=2
        )
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = self.unpatchify(hidden_states, grid_sizes)

        return hidden_states

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache") is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x: torch.Tensor, grid_sizes: tuple[int, int, int]) -> torch.Tensor:
        c = self.proj_out.out_features // math.prod(self.patch_size)
        p_t, p_h, p_w = self.patch_size
        f, h, w = grid_sizes

        x = x[:, : f * h * w].view(-1, f, h, w, p_t, p_h, p_w, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(-1, c, f * p_t, h * p_h, w * p_w)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.proj.weight.flatten(1))
        for m in self.condition_embedder.time_embedder.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.proj_out.weight)

        for block in self.blocks:
            if block.action_model is not None:
                try:
                    nn.init.zeros_(block.action_model.proj_mouse.weight)
                    if block.action_model.proj_mouse.bias is not None:
                        nn.init.zeros_(block.action_model.proj_mouse.bias)
                    nn.init.zeros_(block.action_model.proj_keyboard.weight)
                    if block.action_model.proj_keyboard.bias is not None:
                        nn.init.zeros_(block.action_model.proj_keyboard.bias)
                except AttributeError:
                    pass
