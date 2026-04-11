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
_attn_qat_infer: Callable[..., torch.Tensor] | None = None
_attn_qat_infer_import_attempted = False


def _ensure_kernel_paths() -> None:
    for path in (_project_root, _kernel_root, _kernel_python_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_attn_qat_infer() -> Callable[..., torch.Tensor] | None:
    global _attn_qat_infer
    global _attn_qat_infer_import_attempted

    if _attn_qat_infer_import_attempted:
        return _attn_qat_infer

    _attn_qat_infer_import_attempted = True
    _ensure_kernel_paths()

    try:
        # Prefer the in-repo kernel implementation during local development.
        _attn_qat_infer = importlib.import_module("attn_qat_infer").sageattn_blackwell
    except ImportError:
        _attn_qat_infer = None

    return _attn_qat_infer


def is_attn_qat_infer_available() -> bool:
    return _get_attn_qat_infer() is not None


class AttnQatInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "ATTN_QAT_INFER"

    @staticmethod
    def get_impl_cls() -> type["AttnQatInferImpl"]:
        return AttnQatInferImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class AttnQatInferImpl(AttentionImpl):

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
        attn_qat_infer = _get_attn_qat_infer()
        if attn_qat_infer is None:
            raise ImportError("attn_qat_infer is not available. Please ensure the "
                              "attn_qat_infer kernel package is installed.")

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        output = attn_qat_infer(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=self.causal,
        )
        return output.transpose(1, 2).contiguous()
