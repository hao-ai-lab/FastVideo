# SPDX-License-Identifier: Apache-2.0
"""FlashInfer SageAttention3 (SM120 dense NVFP4 attention) backend.

Wraps ``flashinfer.nvfp4_attention_sm120_quantize_qkv`` +
``flashinfer.nvfp4_attention_sm120_fwd`` (flashinfer-ai/flashinfer#3640).
The kernel is gated to compute capability 12.0 (consumer Blackwell, e.g.
RTX 5090), head_dim 64/128, fp16/bf16, and self-attention with equal
q/k/v shapes. flashinfer pads seq_len up to a multiple of 128 internally;
this backend trims the padding back off the output. Note flashinfer's own
accuracy tests only cover seq_len multiples of 128 — padded-key softmax
mass for other lengths is flashinfer's semantics, unverified here.

Anywhere the kernel cannot run (flashinfer missing, non-SM120 device,
unsupported head size/dtype/shape) the backend logs one warning and falls
back to torch SDPA, so it stays usable as a shape-collection vehicle
(``FASTVIDEO_ATTN_SHAPE_LOG``) on machines without the kernel — e.g.
DGX Spark (SM121), which the collected shape data is being gathered for.

Works with both ``LocalAttention`` and ``DistributedAttention`` (sequence
parallel): inputs arrive as [B, L, H, D] and are transposed internally.
"""

import torch

from fastvideo.attention import shape_logger
from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

try:
    from flashinfer import (nvfp4_attention_sm120_fwd, nvfp4_attention_sm120_quantize_qkv)
except ImportError:
    nvfp4_attention_sm120_fwd = None  # type: ignore[assignment]
    nvfp4_attention_sm120_quantize_qkv = None  # type: ignore[assignment]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_HEAD_DIMS = (64, 128)

_fallback_warned = False


def _fall_back(reason: str) -> bool:
    """Log the first fallback reason once per process; returns False."""
    global _fallback_warned
    if not _fallback_warned:
        logger.warning("FLASHINFER_SAGE_ATTN3: %s. Falling back to torch SDPA "
                       "(subsequent fallbacks are silent).", reason)
        _fallback_warned = True
    return False


def _can_use_flashinfer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: float) -> bool:
    """Whether the flashinfer SM120 kernel can serve this call ([B, H, L, D] inputs)."""
    if nvfp4_attention_sm120_fwd is None:
        return _fall_back("flashinfer with nvfp4_attention_sm120 is not installed")
    if not q.is_cuda:
        return _fall_back(f"device {q.device} is not CUDA")
    cap = torch.cuda.get_device_capability(q.device)
    if cap != (12, 0):
        return _fall_back(f"compute capability {cap[0]}.{cap[1]} is not 12.0 (kernel is SM120-only)")
    if q.dtype not in _SUPPORTED_DTYPES:
        return _fall_back(f"dtype {q.dtype} is not fp16/bf16")
    if q.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        return _fall_back(f"head_dim {q.shape[-1]} not in {_SUPPORTED_HEAD_DIMS}")
    if not (q.shape == k.shape == v.shape):
        return _fall_back(f"q/k/v shapes differ (q={tuple(q.shape)}, k={tuple(k.shape)}, "
                          f"v={tuple(v.shape)}); kernel requires equal-shape self-attention")
    if dropout > 0:
        return _fall_back(f"dropout_p={dropout} is not supported by the kernel")
    return True


class FlashInferSageAttention3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SAGE_ATTN3"

    @staticmethod
    def get_impl_cls() -> type["FlashInferSageAttention3Impl"]:
        return FlashInferSageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class FlashInferSageAttention3Impl(AttentionImpl):

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
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Input is [B, L, H, D] (FastVideo impl convention); output matches."""
        if shape_logger.enabled:
            shape_logger.record(FlashInferSageAttention3Backend.get_name(),
                                query,
                                key,
                                value,
                                causal=self.causal,
                                sm_scale=self.softmax_scale)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if _can_use_flashinfer(q, k, v, self.dropout):
            seq_len = q.shape[2]
            packed = nvfp4_attention_sm120_quantize_qkv(q.contiguous(), k.contiguous(), v.contiguous())
            output, _ = nvfp4_attention_sm120_fwd(*packed,
                                                  sm_scale=self.softmax_scale,
                                                  causal=self.causal,
                                                  out_dtype=query.dtype)
            # flashinfer pads seq_len to a multiple of 128; trim it back off.
            return output[:, :, :seq_len, :].transpose(1, 2).contiguous()

        attn_kwargs = {
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if q.shape[1] != k.shape[1]:
            attn_kwargs["enable_gqa"] = True
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, **attn_kwargs)
        return output.transpose(1, 2)
