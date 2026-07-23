# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
import torch.distributed as dist

import fastvideo.envs as envs
from fastvideo.attention import (DistributedAttention,
                                 LocalAttention)
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import (_apply_rotary_emb,
                                               get_rotary_pos_embed)
from fastvideo.layers.visual_embedding import (PatchEmbed)
from fastvideo.logger import init_logger
from fastvideo.models.dits._causal_train_attention import (
    CausalTrainAttentionPlan,
    approx_relativistic_delta_max,
    build_sink_delta_tables,
    run_causal_train_attention,
    validate_causal_attention_geometry,
)
from fastvideo.models.dits._relative_rope import relativistic_window_offsets
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import WanI2VCrossAttention, WanT2VCrossAttention, WanTimeTextImageEmbedding
from fastvideo.platforms import AttentionBackendEnum, current_platform

logger = init_logger(__name__)

GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES = 21


def _blockwise_causal_attention_visible(q_idx,
                                        kv_idx,
                                        block_end,
                                        frame_seqlen: int,
                                        local_attn_size: int,
                                        sink_size: int = 0):
    """Token-level visibility shared by blockwise-causal training attention."""
    visible_before_block_end = kv_idx < block_end
    if local_attn_size == -1:
        return visible_before_block_end | (q_idx == kv_idx)

    rolling_size = max(0, int(local_attn_size) - int(sink_size))
    visible_in_window = kv_idx >= (block_end - rolling_size * frame_seqlen)
    visible_in_sink = kv_idx < 0
    if int(sink_size) > 0:
        visible_in_sink = kv_idx < int(sink_size) * frame_seqlen
    return (visible_before_block_end & (visible_in_sink | visible_in_window)) | (q_idx == kv_idx)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False,
                 rope_cache_policy: str = "absolute") -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.rope_cache_policy = rope_cache_policy

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor],
                block_mask: BlockMask | CausalTrainAttentionPlan,
                kv_cache: dict | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                frame_seqlen: int = 1560):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            frame_seqlen (int): Number of tokens per latent frame,
                e.g. 1560 for 480x832 resolution.
        """
        if cache_start is None:
            cache_start = current_start

        cos, sin = freqs_cis
        # relativistic defers roping until the cache window is known (and caches raw k)
        relativistic = self.rope_cache_policy == "relativistic" and kv_cache is not None
        if not relativistic:
            roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
            roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)

        if kv_cache is None:
            if isinstance(block_mask, CausalTrainAttentionPlan):
                # Fused sink + rolling-window blockwise attention (Triton or
                # reference); no 128-padding needed and the relativistic sink
                # RoPE correction is applied exactly.
                x = run_causal_train_attention(
                    roped_query.transpose(2, 1),
                    roped_key.transpose(2, 1),
                    v.transpose(2, 1),
                    block_mask,
                ).transpose(2, 1)
                return x

            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :q.shape[1]].transpose(2, 1)
        else:
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            if self.local_attn_size == -1:
                max_attention_size = (GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES * frame_seqlen)
            else:
                max_attention_size = self.local_attn_size * frame_seqlen
            if self.local_attn_size == -1 and current_end > max_attention_size:
                raise ValueError(
                    "Causal Wan local_attn_size=-1 keeps the previous "
                    f"{GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES}-latent-frame KV "
                    "window for compatibility. Set local_attn_size for "
                    f"longer rollouts; got current_end={current_end} tokens "
                    f"with frame_seqlen={frame_seqlen}.")
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            stored_key = k if relativistic else roped_key  # raw vs roped in cache
            global_end_index = (
                int(kv_cache["global_end_index"].item())
                if isinstance(kv_cache["global_end_index"], torch.Tensor)
                else int(kv_cache["global_end_index"])
            )
            local_end_index_prev = (
                int(kv_cache["local_end_index"].item())
                if isinstance(kv_cache["local_end_index"], torch.Tensor)
                else int(kv_cache["local_end_index"])
            )
            if self.local_attn_size != -1 and (current_end > global_end_index) and (
                    num_new_tokens + local_end_index_prev > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + local_end_index_prev - kv_cache_size
                num_rolled_tokens = local_end_index_prev - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = local_end_index_prev + current_end - \
                    global_end_index - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = stored_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = local_end_index_prev + current_end - global_end_index
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"] = kv_cache["k"].detach()
                kv_cache["v"] = kv_cache["v"].detach()
                # logger.info("kv_cache['k'] is in comp graph: %s", kv_cache["k"].requires_grad or kv_cache["k"].grad_fn is not None)
                kv_cache["k"][:, local_start_index:local_end_index] = stored_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            window_start = max(0, local_end_index - max_attention_size)
            if sink_tokens > 0 and window_start > 0:
                if sink_tokens >= max_attention_size:
                    raise ValueError(f"sink_size tokens ({sink_tokens}) must be smaller than "
                                     f"the attention budget ({max_attention_size})")
                local_start = local_end_index - (max_attention_size - sink_tokens)
                key_window = torch.cat(
                    [kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, local_start:local_end_index]], dim=1)
                value_window = torch.cat(
                    [kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, local_start:local_end_index]], dim=1)
            else:
                key_window = kv_cache["k"][:, window_start:local_end_index]
                value_window = kv_cache["v"][:, window_start:local_end_index]
            if relativistic:
                window_len, query_lo, query_hi = relativistic_window_offsets(
                    local_end_index, num_new_tokens, max_attention_size)
                roped_query = _apply_rotary_emb(
                    q, cos[query_lo:query_hi], sin[query_lo:query_hi],
                    is_neox_style=False).type_as(v)
                key_window = _apply_rotary_emb(
                    key_window, cos[:window_len], sin[:window_len],
                    is_neox_style=False).type_as(v)
            x = self.attn(roped_query, key_window, value_window)
            if isinstance(kv_cache["global_end_index"], torch.Tensor):
                kv_cache["global_end_index"].fill_(current_end)
            else:
                kv_cache["global_end_index"] = current_end
            if isinstance(kv_cache["local_end_index"], torch.Tensor):
                kv_cache["local_end_index"].fill_(local_end_index)
            else:
                kv_cache["local_end_index"] = local_end_index

        return x

class CausalWanTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = "",
                 rope_cache_policy: str = "absolute"):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
            rope_cache_policy=rope_cache_policy)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
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
            dtype=torch.float32)

        # 2. Cross-attention. I2V checkpoints prepend CLIP image tokens and
        # carry a second pair of projections, which must be preserved during
        # causal rollouts.
        if added_kv_proj_dim is not None:
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=f"{prefix}.attn2",
            )
        else:
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                prefix=f"{prefix}.attn2",
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | CausalTrainAttentionPlan,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        frame_seqlen: int | None = None,
    ) -> torch.Tensor:
        # hidden_states.shape: [batch_size, seq_length, inner_dim]
        # temb.shape: [batch_size, temb_seq_len, 6, inner_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        temb_seq_len = temb.shape[1]
        tokens_per_temb = hidden_states.shape[1] // temb_seq_len
        if frame_seqlen is None:
            frame_seqlen = tokens_per_temb
        else:
            frame_seqlen = int(frame_seqlen)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb
        # e.shape: [batch_size, temb_seq_len, 6, inner_dim]
        assert e.shape == (bs, temb_seq_len, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2)
        # *_msa.shape: [batch_size, temb_seq_len, 1, inner_dim]
        # assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states).unflatten(dim=1, sizes=(temb_seq_len, tokens_per_temb)) *
                        (1 + scale_msa) + shift_msa).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            frame_seqlen=frame_seqlen,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None,
                                 crossattn_cache=crossattn_cache)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states

class CausalWanTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str,
                                                               Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size
        self.sink_size = config.sink_size
        self.rope_cache_policy = config.arch_config.rope_cache_policy
        self.causal_train_attention = getattr(config.arch_config,
                                              "causal_train_attention", "flex")
        if self.causal_train_attention not in ("flex", "triton", "reference"):
            raise ValueError(
                "causal_train_attention must be one of 'flex', 'triton', "
                f"'reference'; got {self.causal_train_attention!r}")

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            CausalWanTransformerBlock(inner_dim,
                              config.ffn_dim,
                              config.num_attention_heads,
                              config.local_attn_size,
                              config.sink_size,
                              config.qk_norm,
                              config.cross_attn_norm,
                              config.eps,
                              config.added_kv_proj_dim,
                              self._supported_attention_backends,
                              prefix=f"{config.prefix}.blocks.{i}",
                              rope_cache_policy=config.arch_config.rope_cache_policy)
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.teacher_forcing_block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3
        self.independent_first_frame = False
        validate_causal_attention_geometry(
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            num_frame_per_block=self.num_frame_per_block,
            where=type(self).__name__,
        )
        self._relativistic_train_rope_warned = False

        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        sink_size: int = 0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return _blockwise_causal_attention_visible(
                q_idx,
                kv_idx,
                ends[q_idx],
                frame_seqlen,
                local_attn_size,
                sink_size,
            )
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        local_attn_size: int = -1, sink_size: int = 0
    ) -> BlockMask:
        """Attention mask for the teacher-forcing ``[clean | noisy]`` sequence.

        A noisy token attends to its own block plus the clean context of
        strictly previous blocks; clean tokens are block-wise causal. With
        ``local_attn_size != -1`` the visible context mirrors the rolling KV
        cache: the ``sink_size`` leading frames plus the trailing
        ``local_attn_size - sink_size`` frame window, whose budget includes
        the noisy block itself.
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        attention_block_size = frame_seqlen * num_frame_per_block
        rolling_tokens = (max(0, int(local_attn_size) - int(sink_size)) *
                          frame_seqlen if local_attn_size != -1 else 0)
        sink_tokens = int(sink_size) * frame_seqlen if local_attn_size != -1 else 0
        frame_indices = torch.arange(
            start=0, end=num_frames * frame_seqlen,
            step=attention_block_size, device=device, dtype=torch.long
        )
        for start in frame_indices:
            end = start + attention_block_size
            context_ends[start:end] = end
            if local_attn_size != -1:
                context_starts[start:end] = max(0, int(end) - rolling_tokens)

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size, device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = block_index * attention_block_size
            if local_attn_size != -1:
                block_end_tokens = (block_index + 1) * attention_block_size
                noise_context_starts[start:end] = max(0, block_end_tokens - rolling_tokens)

        def attention_mask(b, h, q_idx, kv_idx):
            in_sink = kv_idx < sink_tokens
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx]) & (
                (kv_idx >= context_starts[q_idx]) | in_sink)
            c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            c2 = (kv_idx < noise_context_ends[q_idx]) & (
                (kv_idx >= noise_context_starts[q_idx]) | in_sink)
            noise_mask = (q_idx >= clean_ends) & (c1 | c2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length,
            _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a teacher-forcing mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _get_train_attention_spec(
        self,
        *,
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        teacher_forcing: bool,
    ) -> BlockMask | CausalTrainAttentionPlan:
        """Build and cache the full-sequence causal training attention spec."""
        attr = "teacher_forcing_block_mask" if teacher_forcing else "block_mask"
        spec = getattr(self, attr)
        if spec is not None:
            return spec

        relativistic_sinks = (
            self.rope_cache_policy == "relativistic" and self.sink_size > 0
            and approx_relativistic_delta_max(
                num_frames=num_frames,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size) > 0)

        if self.causal_train_attention == "flex":
            if relativistic_sinks and not self._relativistic_train_rope_warned:
                logger.warning(
                    "rope_cache_policy='relativistic' with sink_size=%d: the "
                    "FlexAttention training path keeps absolute sink RoPE, so "
                    "query->sink phases differ from the re-indexed streaming "
                    "cache once the rolling window scrolls past the sink. Set "
                    "pipeline.dit_config.causal_train_attention: triton for "
                    "the exact training-time correction.", self.sink_size)
                self._relativistic_train_rope_warned = True
            if teacher_forcing:
                spec = self._prepare_teacher_forcing_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                )
            else:
                spec = self._prepare_blockwise_causal_attn_mask(
                    device=device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                )
        else:
            delta_cos = delta_sin = None
            if relativistic_sinks:
                tables = build_sink_delta_tables(
                    num_frames=num_frames,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    dtype=torch.float32 if current_platform.is_mps() else torch.float64,
                    device=device,
                )
                if tables is not None:
                    delta_cos, delta_sin = tables
            spec = CausalTrainAttentionPlan(
                kind="teacher_forcing" if teacher_forcing else "blockwise",
                impl=self.causal_train_attention,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                sm_scale=1.0 / math.sqrt(self.hidden_size // self.num_attention_heads),
                delta_cos=delta_cos,
                delta_sin=delta_sin,
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(
                    "cache a %s causal train-attention plan (impl=%s, "
                    "local_attn_size=%d, sink_size=%d, relativistic_sinks=%s)",
                    spec.kind,
                    spec.impl,
                    self.local_attn_size,
                    self.sink_size,
                    relativistic_sinks,
                )

        setattr(self, attr, spec)
        return spec

    def _forward_inference(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                kv_cache: dict = None,
                crossattn_cache: dict = None,
                current_start: int = 0,
                cache_start: int = 0,
                start_frame: int = 0,
                **kwargs) -> torch.Tensor:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)
        """

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (
                encoder_hidden_states_image[0]
                if encoder_hidden_states_image else None)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        if self.rope_cache_policy == "relativistic":
            # fixed table over [0, max_attention_frames); attention slices it per step
            max_attention_frames = (
                GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES
                if self.local_attn_size == -1 else self.local_attn_size)
            rope_num_frames = max_attention_frames * get_sp_world_size()
            rope_start_frame = 0
        else:
            rope_num_frames = post_patch_num_frames * get_sp_world_size()
            rope_start_frame = start_frame  # 0 when kv_cache is None
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (rope_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=rope_start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.tensor(
            hidden_states.shape[2:], dtype=torch.long).unsqueeze(0).repeat(
                batch_size, 1)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states_padding = encoder_hidden_states.new_zeros(
            batch_size, self.text_len - encoder_hidden_states.size(1),
            encoder_hidden_states.size(2))
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, encoder_hidden_states_padding], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                    "frame_seqlen": post_patch_height * post_patch_width,
                }
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    **causal_kwargs)
            else:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": (crossattn_cache[block_index]
                                        if crossattn_cache is not None else None),
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                    "frame_seqlen": post_patch_height * post_patch_width,
                }
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        **causal_kwargs)

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def _forward_train(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                start_frame: int = 0,
                clean_x: torch.Tensor | None = None,
                aug_t: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:

        orig_dtype = hidden_states.dtype
        teacher_forcing = clean_x is not None
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (
                encoder_hidden_states_image[0]
                if encoder_hidden_states_image else None)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sp_world_size(), post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000,
            start_frame=start_frame
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos,
                     freqs_sin) if freqs_cos is not None else None

        block_mask = self._get_train_attention_spec(
            device=hidden_states.device,
            num_frames=num_frames,
            frame_seqlen=post_patch_height * post_patch_width,
            teacher_forcing=teacher_forcing,
        )

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.tensor(
            hidden_states.shape[2:], dtype=torch.long).unsqueeze(0).repeat(
                batch_size, 1)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states_padding = encoder_hidden_states.new_zeros(
            batch_size, self.text_len - encoder_hidden_states.size(1),
            encoder_hidden_states.size(2))
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, encoder_hidden_states_padding], dim=1)
        encoder_hidden_states_text = encoder_hidden_states

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                        timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        encoder_hidden_states = encoder_hidden_states.to(
            orig_dtype) if current_platform.is_mps(
            ) else encoder_hidden_states  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        if teacher_forcing:
            # Tile RoPE/modulation so clean frame i and noisy frame i share a position.
            clean_tokens = self.patch_embedding(clean_x).flatten(2).transpose(1, 2)
            hidden_states = torch.cat([clean_tokens, hidden_states], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(timestep)
            _, timestep_proj_clean, _, _ = self.condition_embedder(
                aug_t.flatten(), encoder_hidden_states_text, None)
            timestep_proj_clean = timestep_proj_clean.unflatten(
                1, (6, self.hidden_size)).unflatten(dim=0, sizes=timestep.shape)
            timestep_proj = torch.cat([timestep_proj_clean, timestep_proj], dim=1)
            freqs_cis = (torch.cat([freqs_cos, freqs_cos], dim=0),
                         torch.cat([freqs_sin, freqs_sin], dim=0))

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis,
                    block_mask=block_mask)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                        timestep_proj, freqs_cis,
                                        block_mask=block_mask)

        if teacher_forcing:
            hidden_states = hidden_states[:, hidden_states.shape[1] // 2:]

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2,
                                                                    dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)


    def unpatchify(self, x, grid_sizes):
        r"""


        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,


        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """

        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            # u = torch.einsum('fhwpqrc->cfphqwr', u.contiguous())
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

# Entry point for model registry
EntryClass = CausalWanTransformer3DModel
