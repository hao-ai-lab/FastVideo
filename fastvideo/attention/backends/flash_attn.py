# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func as flash_attn_2_func
from dataclasses import dataclass
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    # flash_attn 3 no longer have a different API, see following commit:
    # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
    flash_attn_func = flash_attn_3_func
except ImportError:
    flash_attn_func = flash_attn_2_func

from fastvideo.attention.backends.abstract import (AttentionBackend,
                                                   AttentionImpl,
                                                   AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

@dataclass
class FlashAttnMetadata(AttentionMetadata):
    current_timestep: int
    attn_mask: torch.Tensor | None = None

class FlashAttnMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        attn_mask: torch.Tensor,
    ) -> FlashAttnMetadata:
        return FlashAttnMetadata(current_timestep=current_timestep, attn_mask=attn_mask)


class FlashAttentionImpl(AttentionImpl):

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: FlashAttnMetadata,
    ):
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            from fastvideo.attention.utils.flash_attn_no_pad import flash_attn_no_pad
            attn_mask = attn_metadata.attn_mask
            qkv = torch.stack([query, key, value], dim=2)

            attn_mask = F.pad(attn_mask, (qkv.shape[1] - attn_mask.shape[1], 0), value=True)
            output = flash_attn_no_pad(
                qkv,
                attn_mask,
                causal=False,
                dropout_p=0,
                softmax_scale=None
            )
        else:
            output = flash_attn_func(
                query,  # type: ignore[no-untyped-call]
                key,
                value,
                softmax_scale=self.softmax_scale,
                causal=self.causal)
        return output
