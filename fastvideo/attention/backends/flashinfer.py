# SPDX-License-Identifier: Apache-2.0
"""FlashInfer dense prefill attention backend.

This backend uses either FlashInfer's single-request NHD kernel once per batch
item or its batched cuDNN SDPA kernel. Select the implementation with
``FASTVIDEO_FLASHINFER_PREFILL_BACKEND=single|cudnn`` (default: ``single``).
Both paths preserve FastVideo's BSHD/SP contract and support self-attention,
cross-attention, GQA, and causal attention. Arbitrary masks remain on the
single-request path because FlashInfer's cuDNN entry point has no custom-mask
argument.
FlashInfer's prefill API is inference-only here; training must use FLASH_ATTN or
TORCH_SDPA.

``forward`` reads ``attn_mask``/``is_causal`` off ``attn_metadata`` by
attribute, not by ``isinstance(FlashInferMetadata)``: several model forwards
(e.g. HYWorld) build ``SDPAMetadata`` directly and hand it to whichever
backend the selector resolved. Any metadata dataclass with the same field
names satisfies this backend; do not add a strict type check here without
also updating those call sites.

A layer must list FLASHINFER explicitly in its ``supported_attention_backends``
to use this backend — there is no implicit bridge from FLASH_ATTN support, since
this kernel's masking/GQA conventions have not been vetted per-model.
"""

from dataclasses import dataclass

import torch

import fastvideo.envs as envs
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

    _CUDNN_WORKSPACE_BYTES = 128 * 1024 * 1024
    # FlashInfer documents 128 MiB as sufficient for typical prefill. Share one
    # allocation per device instead of reserving it once per transformer layer.
    _cudnn_workspaces: dict[torch.device, torch.Tensor] = {}

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 causal: bool,
                 softmax_scale: float,
                 num_kv_heads: int | None = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        del num_heads, num_kv_heads, prefix, extra_impl_args
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.head_size = head_size
        self.prefill_backend = envs.FASTVIDEO_FLASHINFER_PREFILL_BACKEND
        if self.prefill_backend not in ("single", "cudnn"):
            raise ValueError("FASTVIDEO_FLASHINFER_PREFILL_BACKEND must be 'single' or 'cudnn'; "
                             f"got {self.prefill_backend!r}")
        if self.prefill_backend == "cudnn" and head_size != 128:
            raise ValueError("FlashInfer cuDNN prefill requires head size 128 in FastVideo; "
                             f"got {head_size}. Use FASTVIDEO_FLASHINFER_PREFILL_BACKEND=single instead.")
    def _forward_cudnn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool) -> torch.Tensor:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]
        workspace = self._cudnn_workspaces.get(query.device)
        if workspace is None:
            workspace = torch.empty(self._CUDNN_WORKSPACE_BYTES, dtype=torch.uint8, device=query.device)
            self._cudnn_workspaces[query.device] = workspace

        query_offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * query_len
        key_offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device) * key_len
        output, _ = cudnn_batch_prefill_with_kv_cache(
            query.flatten(0, 1),
            key.flatten(0, 1),
            value.flatten(0, 1),
            self.softmax_scale,
            workspace,
            max_token_per_sequence=query_len,
            max_sequence_kv=key_len,
            batch_offsets_q=query_offsets,
            batch_offsets_o=query_offsets,
            batch_offsets_k=key_offsets,
            batch_offsets_v=key_offsets,
            batch_offsets_units="tokens",
            causal=causal,
            return_lse=False,
        )
        return output.view_as(query)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_metadata: FlashInferMetadata | None) -> torch.Tensor:
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
            raise RuntimeError("FLASHINFER backend is inference-only; use FLASH_ATTN or TORCH_SDPA for training.")

        original_dtype = query.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        mask = attn_metadata.attn_mask if attn_metadata is not None else None
        causal = self.causal or bool(attn_metadata is not None and getattr(attn_metadata, "is_causal", False))
        if self.prefill_backend == "cudnn":
            if mask is not None:
                raise ValueError("FlashInfer cuDNN prefill does not support arbitrary attention masks; "
                                 "use FASTVIDEO_FLASHINFER_PREFILL_BACKEND=single instead.")
            output = self._forward_cudnn(query, key, value, causal)
            return output.to(original_dtype) if output.dtype != original_dtype else output

        from flashinfer.prefill import single_prefill_with_kv_cache

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
