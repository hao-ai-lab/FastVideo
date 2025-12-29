# SPDX-License-Identifier: Apache-2.0

import torch
from dataclasses import dataclass
from fastvideo.attention.backends.abstract import (  # FlashAttentionMetadata,
    AttentionBackend, AttentionImpl, AttentionMetadata,
    AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SDPA"

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
        return SDPAMetadata(current_timestep=current_timestep,
                            attn_mask=attn_mask)


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

        attn_mask = attn_metadata.attn_mask if (
            attn_metadata is not None
            and hasattr(attn_metadata, "attn_mask")) else None
        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, **attn_kwargs)
        output = output.transpose(1, 2)
        return output
