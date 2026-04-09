# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent.parent
_kernel_root = _project_root / "fastvideo-kernel"
_kernel_python_root = _kernel_root / "python"
_attn_qat_train_attention: Callable[..., torch.Tensor] | None = None
_attn_qat_train_import_attempted = False


def _ensure_kernel_paths() -> None:
    for path in (_project_root, _kernel_root, _kernel_python_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_attn_qat_train_attention() -> Callable[..., torch.Tensor] | None:
    global _attn_qat_train_attention
    global _attn_qat_train_import_attempted

    if _attn_qat_train_import_attempted:
        return _attn_qat_train_attention

    _attn_qat_train_import_attempted = True
    _ensure_kernel_paths()

    try:
        _attn_qat_train_attention = importlib.import_module("fastvideo_kernel.triton_kernels.attn_qat_train").attention
    except ImportError:
        _attn_qat_train_attention = None

    return _attn_qat_train_attention


def attn_qat_train(q_BLHD: torch.Tensor,
                   k_BLHD: torch.Tensor,
                   v_BLHD: torch.Tensor,
                   is_causal: bool = False) -> torch.Tensor:
    attention = _get_attn_qat_train_attention()
    if attention is None:
        raise ImportError("fastvideo_kernel.triton_kernels.attn_qat_train is not available. "
                          "Please ensure the FastVideo kernel package is installed.")

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
    sm_scale = 1.0 / (q_BHLD.shape[-1]**0.5)
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


class AttnQatTrainBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "ATTN_QAT_TRAIN"

    @staticmethod
    def get_impl_cls() -> type["AttnQatTrainImpl"]:
        return AttnQatTrainImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class AttnQatTrainImpl(AttentionImpl):

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
        return attn_qat_train(query, key, value, is_causal=self.causal)
