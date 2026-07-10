# SPDX-License-Identifier: Apache-2.0

import torch
from sageattn3 import sageattn3_blackwell

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class SageAttention3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_THREE"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SageAttention3Impl(AttentionImpl):

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

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Transpose stacked QKV from [3B, L, H, D] to [3B, H, L, D].

        Single bulk permute+contiguous on the entire stacked tensor rather than
        three separate transposed views for Q, K, V. The .contiguous() is
        required: sageattn_blackwell's fake kernel returns empty_like(q), so the
        op's output strides must match contiguous q under torch.compile.
        """
        return qkv.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Call sageattn3_blackwell directly. Input is already [B, H, L, D]
        and contiguous from preprocess_qkv."""
        output = sageattn3_blackwell(query, key, value, is_causal=self.causal)
        return output

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Transpose output from [B, H, L, D] back to [B, L, H, D]."""
        return output.permute(0, 2, 1, 3).contiguous()
