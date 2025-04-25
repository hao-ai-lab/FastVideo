from typing import List, Optional, Type

import torch
from sageattention import sageattn

from fastvideo.v1.attention.backends.abstract import (
    AttentionBackend)  # FlashAttentionMetadata,
from fastvideo.v1.attention.backends.abstract import (AttentionImpl,
                                                      AttentionMetadata)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class SageAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["SageAttentionImpl"]:
        return SageAttentionImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SageAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
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
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        # query = query.transpose(1, 2)
        # key = key.transpose(1, 2)
        # value = value.transpose(1, 2)
        attn_kwargs = {
            "attn_mask": None,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        # output = torch.nn.functional.scaled_dot_product_attention(
        #     query, key, value, **attn_kwargs)
        output = sageattn(query,
                          key,
                          value,
                          tensor_layout="NHD",
                          is_causal=self.causal)
        #output = output.transpose(1, 2)
        assert torch.isnan(output).sum() == 0, "sage_attn has nan"
        return output
