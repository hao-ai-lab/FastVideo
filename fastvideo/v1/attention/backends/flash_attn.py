# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Type

import torch
from flash_attn import flash_attn_func

from fastvideo.v1.logger import init_logger

from .abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                       AttentionMetadataBuilder)

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        dropout_rate: float,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        output = flash_attn_func(query,
                                 key,
                                 value,
                                 dropout_p=self.dropout_rate,
                                 softmax_scale=self.softmax_scale,
                                 causal=self.causal)
        return output
