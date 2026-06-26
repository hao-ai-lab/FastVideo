# SPDX-License-Identifier: Apache-2.0
"""Tests for the selector's capability-mismatch warning (#1254, step 3).

`warn_if_backend_incompatible` is exercised in isolation with a dummy
backend, so these run on CPU without GPU kernels or the platform machinery.
"""
from unittest import mock

import torch

from fastvideo.attention import selector
from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)


class _Restricted(AttentionBackend):
    """Backend that only supports head sizes 64 and 128."""

    @staticmethod
    def get_name() -> str:
        return "RESTRICTED"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


def test_warns_on_incompatible_head_size():
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        selector.warn_if_backend_incompatible(_Restricted, head_size=96, dtype=torch.float16)
    mock_warn.assert_called_once()
    # backend name and reason are passed as format args.
    assert "RESTRICTED" in mock_warn.call_args.args


def test_warns_on_incompatible_dtype():
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        selector.warn_if_backend_incompatible(_Restricted, head_size=64, dtype=torch.float32)
    mock_warn.assert_called_once()


def test_no_warning_when_compatible():
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        selector.warn_if_backend_incompatible(_Restricted, head_size=128, dtype=torch.float16)
    mock_warn.assert_not_called()
