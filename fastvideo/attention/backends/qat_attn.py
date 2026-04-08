# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Add local repo roots to sys.path for kernel-side imports during development.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_kernel_root = _project_root / "fastvideo-kernel"
_kernel_python_root = _kernel_root / "python"
for _path in (_project_root, _kernel_root, _kernel_python_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.logger import init_logger
from fastvideo_kernel.triton_kernels.qat_attn import attention

logger = init_logger(__name__)


def qat_attn(q_BLHD: torch.Tensor, k_BLHD: torch.Tensor, v_BLHD: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    q_BHLD = q_BLHD.permute(0, 2, 1, 3).contiguous()
    k_BHLD = k_BLHD.permute(0, 2, 1, 3).contiguous()
    v_BHLD = v_BLHD.permute(0, 2, 1, 3).contiguous()

    use_qat_qkv_backward = True
    smooth_k = False
    warp_specialize = True
    is_qat = True
    two_level_quant_p_sage3 = False
    fake_quant_p_bwd = True
    use_high_prec_o = True
    smooth_q = False
    sm_scale = 1.0 / (q_BHLD.shape[-1] ** 0.5)
    use_global_sf_qkv = False
    use_global_sf_p = False

    o_BHLD = attention(
        q_BHLD,
        k_BHLD,
        v_BHLD,
        is_causal,
        sm_scale,
        use_qat_qkv_backward,
        smooth_k,
        warp_specialize,
        is_qat,
        two_level_quant_p_sage3,
        fake_quant_p_bwd,
        use_high_prec_o,
        smooth_q,
        use_global_sf_p,
        use_global_sf_qkv,
    )
    return o_BHLD.permute(0, 2, 1, 3).contiguous()


class QATAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "QAT_ATTN"

    @staticmethod
    def get_impl_cls() -> type["QATAttentionImpl"]:
        return QATAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class QATAttentionImpl(AttentionImpl):

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
        return qat_attn(query, key, value, is_causal=self.causal)
