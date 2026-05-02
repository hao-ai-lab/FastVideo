# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from dataclasses import dataclass

try:
    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func

    fa_version = "4"
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func

        # flash_attn 3 no longer have a different API, see following commit:
        # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
        flash_attn_func = flash_attn_3_func
        fa_version = "3"
    except ImportError:
        from flash_attn import flash_attn_func as flash_attn_2_func
        flash_attn_func = flash_attn_2_func
        fa_version = "2"

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)
logger.info("Using FlashAttention-%s backend", fa_version)


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


def _key_padding_mask_from_attn_mask(attn_mask: torch.Tensor, key_len: int) -> torch.Tensor:
    # Normalize attn_mask to [B, key_len] where True means valid token.
    if attn_mask.dim() == 4:
        attn_mask = attn_mask[:, 0, 0, :]
    elif attn_mask.dim() == 3:
        attn_mask = attn_mask[:, 0, :]
    elif attn_mask.dim() != 2:
        raise ValueError(f"Unsupported attn_mask shape for FLASH_ATTN: {attn_mask.shape}")

    # SDPA additive mask convention: valid=0, masked=-inf/large negative.
    key_padding_mask = attn_mask if attn_mask.dtype == torch.bool else attn_mask >= 0

    if key_padding_mask.shape[-1] != key_len:
        raise ValueError("Invalid key padding mask length for FLASH_ATTN: "
                         f"expected {key_len}, got {key_padding_mask.shape[-1]}")
    return key_padding_mask


@dataclass
class FlashAttnMetadata(AttentionMetadata):
    current_timestep: int
    attn_mask: torch.Tensor | None = None
    # Per-step cache of converted/padded masks.  Mask conversion only
    # depends on shape and on ``attn_mask`` itself, both of which are
    # constant across all attention layers within one denoising step,
    # so we share the converted masks across layers instead of redoing
    # the work on every forward.  Populated lazily by the impl.
    # Keys are tuples like ("kpad", key_len), ("qpad", batch, q_len),
    # ("self_attn_pad", qkv_len).
    _mask_cache: dict | None = None


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
        if (attn_metadata is not None and hasattr(attn_metadata, "attn_mask") and attn_metadata.attn_mask is not None):
            from fastvideo.attention.utils.flash_attn_no_pad import (
                flash_attn_no_pad,
                flash_attn_varlen_qk_no_pad,
            )

            attn_mask = attn_metadata.attn_mask
            if attn_metadata._mask_cache is None:
                attn_metadata._mask_cache = {}
            cache = attn_metadata._mask_cache

            # flash_attn_no_pad packs q/k/v as one tensor and assumes equal q/k
            # sequence lengths. Cross-attention can violate this.
            if query.shape[1] != key.shape[1]:
                qpad_key = ("qpad", query.shape[0], query.shape[1])
                if qpad_key not in cache:
                    cache[qpad_key] = torch.ones(
                        (query.shape[0], query.shape[1]),
                        dtype=torch.bool,
                        device=query.device,
                    )
                query_padding_mask = cache[qpad_key]

                kpad_key = ("kpad", key.shape[1])
                if kpad_key not in cache:
                    cache[kpad_key] = _key_padding_mask_from_attn_mask(attn_mask, key.shape[1]).to(device=key.device)
                key_padding_mask = cache[kpad_key]

                return flash_attn_varlen_qk_no_pad(
                    query,
                    key,
                    value,
                    query_padding_mask=query_padding_mask,
                    key_padding_mask=key_padding_mask,
                    causal=self.causal,
                    dropout_p=0.0,
                    softmax_scale=self.softmax_scale,
                )

            qkv = torch.stack([query, key, value], dim=2)

            self_pad_key = ("self_attn_pad", qkv.shape[1])
            if self_pad_key not in cache:
                cache[self_pad_key] = F.pad(attn_mask, (qkv.shape[1] - attn_mask.shape[1], 0), value=True)
            attn_mask_padded = cache[self_pad_key]
            output = flash_attn_no_pad(qkv, attn_mask_padded, causal=False, dropout_p=0, softmax_scale=None)
        else:
            output = flash_attn_func(
                query,  # type: ignore[no-untyped-call]
                key,
                value,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
        return output
