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
_sageattn_blackwell: Callable[..., torch.Tensor] | None = None
_sageattn_import_attempted = False


def _ensure_kernel_paths() -> None:
    for path in (_project_root, _kernel_root, _kernel_python_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_sageattn_blackwell() -> Callable[..., torch.Tensor] | None:
    global _sageattn_blackwell
    global _sageattn_import_attempted

    if _sageattn_import_attempted:
        return _sageattn_blackwell

    _sageattn_import_attempted = True
    _ensure_kernel_paths()

    try:
        # Prefer the in-repo kernel implementation during local development.
        _sageattn_blackwell = importlib.import_module("modified_sageattn").sageattn_blackwell
    except ImportError:
        _sageattn_blackwell = None

    return _sageattn_blackwell


def is_modified_sageattn_available() -> bool:
    return _get_sageattn_blackwell() is not None


class ModifiedSageAttention3Backend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "MODIFIED_SAGE_ATTN_THREE"

    @staticmethod
    def get_impl_cls() -> type["ModifiedSageAttention3Impl"]:
        return ModifiedSageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class ModifiedSageAttention3Impl(AttentionImpl):

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
        sageattn_blackwell = _get_sageattn_blackwell()
        if sageattn_blackwell is None:
            raise ImportError("modified_sageattn is not available. Please ensure the "
                              "modified SageAttention3 kernel is installed.")

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        output = sageattn_blackwell(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=self.causal,
        )
        return output.transpose(1, 2).contiguous()


SageAttention3Backend = ModifiedSageAttention3Backend
SageAttention3Impl = ModifiedSageAttention3Impl
