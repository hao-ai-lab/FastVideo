# SPDX-License-Identifier: Apache-2.0
"""NABLA block-sparse flex-attention backend (Kandinsky5 "nabla" checkpoints).

The block mask is data-dependent: nablaT_v2 mean-pools 64-token blocks of the
fractal-ordered sequence, thresholds the softmaxed block map, and ORs it with a
precomputed spatio-temporal-window (STA) mask carried on the attention
metadata. The mask spans the full sequence, so this backend does not support
sequence parallelism — use it via LocalAttention only.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    CAN_USE_FLEX_ATTN = True
except ImportError:
    CAN_USE_FLEX_ATTN = False

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)


def nablaT_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    sta: torch.Tensor,
    thr: float = 0.9,
) -> "BlockMask":
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()

    # Map estimation
    B, h, S, D = q.shape
    s1 = S // 64
    qa = q.reshape(B, h, s1, 64, D).mean(-2)
    ka = k.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    map = torch.softmax(map / math.sqrt(D), dim=-1)
    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    mask = torch.logical_or(mask, sta)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None)


class NablaAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "NABLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["NablaAttentionImpl"]:
        return NablaAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["NablaAttentionMetadata"]:
        return NablaAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["NablaAttentionMetadataBuilder"]:
        return NablaAttentionMetadataBuilder


@dataclass
class NablaAttentionMetadata(AttentionMetadata):
    # Block-level STA window mask [1, 1, S/64, S/64], precomputed once per run.
    sta_mask: torch.Tensor = None  # type: ignore[assignment]
    # Cumulative-probability threshold for block-map binarization.
    P: float = 0.9
    visual_shape: tuple[int, int, int] = (0, 0, 0)


class NablaAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        sta_mask: torch.Tensor,
        P: float,
        visual_shape: tuple[int, int, int],
        **kwargs: Any,
    ) -> NablaAttentionMetadata:
        return NablaAttentionMetadata(
            current_timestep=current_timestep,
            sta_mask=sta_mask,
            P=P,
            visual_shape=visual_shape,
        )


class NablaAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if not CAN_USE_FLEX_ATTN:
            raise RuntimeError("NABLA attention requires torch.nn.attention.flex_attention, "
                               "which is unavailable in this PyTorch build.")
        if causal:
            raise ValueError("NABLA attention does not support causal masking.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: NablaAttentionMetadata,
    ) -> torch.Tensor:
        # q/k/v: [B, S, heads, head_dim], fractal-ordered by the model; S % 64 == 0.
        block_mask = nablaT_v2(query, key, attn_metadata.sta_mask, thr=attn_metadata.P)
        return flex_attention(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)
