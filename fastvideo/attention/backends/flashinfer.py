# SPDX-License-Identifier: Apache-2.0
"""FlashInfer dense prefill attention backend.

This backend uses FlashInfer's single-request NHD kernel once per batch item.
That path preserves FastVideo's BSHD/SP contract and supports self-attention,
cross-attention, GQA, causal attention, and tokenizer-style padding masks.
FlashInfer's prefill API is inference-only here; training must use FLASH_ATTN or
TORCH_SDPA.
"""

from dataclasses import dataclass

import torch

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)


@dataclass
class FlashInferMetadata(AttentionMetadata):
    current_timestep: int
    attn_mask: torch.Tensor | None = None
    is_causal: bool = False


class FlashInferMetadataBuilder(AttentionMetadataBuilder):

    def prepare(self):
        pass

    def build(self, current_timestep: int, attn_mask: torch.Tensor | None = None) -> FlashInferMetadata:  # type: ignore
        return FlashInferMetadata(current_timestep=current_timestep, attn_mask=attn_mask)


class FlashInferBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer 0.6.x's FA2 fallback is unsafe for some other dimensions.
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> type[FlashInferMetadata]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> type[FlashInferMetadataBuilder]:
        return FlashInferMetadataBuilder


def _mask_for_sample(attn_mask: torch.Tensor, sample: int, query_len: int, key_len: int) -> torch.Tensor:
    """Convert FastVideo's padding/additive mask to FlashInfer's [Q, K] bool mask."""
    mask = attn_mask.to(dtype=torch.bool) if not attn_mask.dtype.is_floating_point else attn_mask >= 0
    if mask.dim() == 2:
        if mask.shape[-1] > key_len:
            raise ValueError(f"Invalid FLASHINFER mask length: expected at most {key_len}, got {mask.shape[-1]}")
        key_mask = mask[sample]
        if key_mask.shape[0] < key_len:
            key_mask = torch.nn.functional.pad(key_mask, (key_len - key_mask.shape[0], 0), value=True)
        return key_mask.unsqueeze(0).expand(query_len, -1)
    if mask.dim() == 3:
        mask = mask[sample]
    elif mask.dim() == 4:
        mask = mask[sample, 0]
    else:
        raise ValueError(f"Unsupported FLASHINFER attention mask shape: {attn_mask.shape}")
    if mask.shape[-2:] != (query_len, key_len):
        if mask.shape[-2] == 1 and mask.shape[-1] == key_len:
            mask = mask.expand(query_len, -1)
        else:
            raise ValueError(f"FLASHINFER mask must broadcast to [{query_len}, {key_len}], got {mask.shape}")
    return mask


class FlashInferImpl(AttentionImpl):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 causal: bool,
                 softmax_scale: float,
                 num_kv_heads: int | None = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        del num_heads, head_size, num_kv_heads, prefix, extra_impl_args
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_metadata: FlashInferMetadata | None) -> torch.Tensor:
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
            raise RuntimeError("FLASHINFER backend is inference-only; use FLASH_ATTN or TORCH_SDPA for training.")

        from flashinfer.prefill import single_prefill_with_kv_cache

        original_dtype = query.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        mask = attn_metadata.attn_mask if attn_metadata is not None else None
        causal = self.causal or bool(attn_metadata is not None and getattr(attn_metadata, "is_causal", False))
        outputs = []
        for sample in range(query.shape[0]):
            custom_mask = None
            if mask is not None:
                custom_mask = _mask_for_sample(mask, sample, query.shape[1], key.shape[1]).to(query.device)
                if causal:
                    causal_mask = torch.ones((query.shape[1], key.shape[1]), dtype=torch.bool,
                                             device=query.device).tril(key.shape[1] - query.shape[1])
                    custom_mask = custom_mask & causal_mask
            outputs.append(
                single_prefill_with_kv_cache(query[sample],
                                             key[sample],
                                             value[sample],
                                             custom_mask=custom_mask,
                                             causal=causal and custom_mask is None,
                                             kv_layout="NHD",
                                             sm_scale=self.softmax_scale))
        output = torch.stack(outputs)
        return output.to(original_dtype) if output.dtype != original_dtype else output
