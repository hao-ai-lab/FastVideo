# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass
from torch.nn import functional as F
from fastvideo.attention.backends.abstract import (  # FlashAttentionMetadata,
    AttentionBackend, AttentionImpl, AttentionMetadata, AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int] | None:
        # torch.nn.functional.scaled_dot_product_attention is head-size
        # agnostic: the math backend handles any size and the fused kernels
        # fall back internally when they cannot. The previous list
        # ([32, 64, ..., 256]) was copied from FlashAttentionBackend in the
        # v1 refactor (#270) and wrongly excluded sizes such as 80 (e.g.
        # CLIP vision encoders). None means "no restriction".
        return None

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


@dataclass
class SDPAMetadata(AttentionMetadata):
    current_timestep: int
    attn_mask: torch.Tensor | None = None


class SDPAMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
            self,
            current_timestep: int,
            attn_mask: torch.Tensor,
    ) -> SDPAMetadata:
        # Store the mask exactly as passed. The metadata is cross-backend:
        # call sites (HYWorld, HunyuanVideo15) build SDPAMetadata while the
        # layer's selector may pick FLASH_ATTN, and the shared convention for
        # padding masks is the tokenizer-style 2D [batch, key_len]. Any
        # reshaping for torch.sdpa happens inside the SDPA impl
        # (_normalize_attn_mask_for_sdpa).
        return SDPAMetadata(current_timestep=current_timestep, attn_mask=attn_mask)


def _normalize_attn_mask_for_sdpa(
    attn_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None

    attn_mask = attn_mask.to(device=query.device)
    # F.scaled_dot_product_attention only accepts bool or float masks;
    # tokenizers commonly produce int64 0/1 padding masks.
    if attn_mask.dtype != torch.bool and not attn_mask.dtype.is_floating_point:
        attn_mask = attn_mask != 0

    key_len = key.shape[-2]
    if attn_mask.shape[-1] > key_len:
        raise ValueError("Invalid attention mask length for SDPA: "
                         f"expected at most {key_len}, got {attn_mask.shape[-1]}")
    if attn_mask.shape[-1] < key_len:
        # Front-pad as "attend": double-stream layouts (HYWorld) prepend
        # non-text tokens the tokenizer mask does not cover.
        valid_value = True if attn_mask.dtype == torch.bool else 0.0
        attn_mask = F.pad(attn_mask, (key_len - attn_mask.shape[-1], 0), value=valid_value)

    if attn_mask.dim() == 2:
        # In-tree producers pass 2D [batch, key_len] padding masks; lift to a
        # broadcastable [batch, 1, 1, key_len] here so torch.sdpa does not
        # reinterpret 2D as its documented [query_len, key_len] broadcast.
        return attn_mask[:, None, None, :]
    if attn_mask.dim() == 3:
        return attn_mask[:, None, :, :]
    if attn_mask.dim() == 4:
        return attn_mask
    raise ValueError(f"Unsupported attention mask shape for SDPA: {attn_mask.shape}")


class SDPAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SDPAMetadata,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_mask = attn_metadata.attn_mask if (attn_metadata is not None
                                                and hasattr(attn_metadata, "attn_mask")) else None
        attn_mask = _normalize_attn_mask_for_sdpa(attn_mask, query, key)
        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value, **attn_kwargs)
        output = output.transpose(1, 2)
        return output
