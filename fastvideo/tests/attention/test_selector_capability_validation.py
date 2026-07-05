# SPDX-License-Identifier: Apache-2.0
"""Tests for selector-side capability validation (#1254).

The selector checks the resolved backend against its self-described
capabilities (AttentionBackend.validate_compatibility): an explicitly
selected backend that is incompatible hard-fails, mirroring how the
platform layer hard-fails on missing explicitly-requested backends,
while an auto-selected backend only warns -- once per cached resolution.
These use a dummy backend and a stubbed platform, so they run on CPU.
"""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import fastvideo.platforms as platforms
from fastvideo.attention import selector
from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.platforms.interface import AttentionBackendEnum

SUPPORTED_BACKENDS = (AttentionBackendEnum.FLASH_ATTN, )


class _RestrictedBackend(AttentionBackend):
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


@pytest.fixture(autouse=True)
def reset_attention_backend_selector():
    # Also clears the selector cache between tests.
    selector.global_force_attn_backend(None)
    yield
    selector.global_force_attn_backend(None)


@pytest.fixture(autouse=True)
def stub_backend_resolution(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    platform = SimpleNamespace(
        device_name="test device",
        get_attn_backend_cls=lambda selected_backend, head_size, dtype: "RESTRICTED",
    )
    monkeypatch.setattr(platforms, "_current_platform", platform)
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", {"RESTRICTED": _RestrictedBackend}.__getitem__)


def test_explicitly_selected_incompatible_backend_raises(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    with pytest.raises(ValueError, match="head_size"):
        selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)


def test_auto_selected_incompatible_backend_warns_once():
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        first = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)
        # A second lookup is a cache hit and must not warn again.
        second = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)
    assert first is _RestrictedBackend
    assert second is _RestrictedBackend
    mock_warn.assert_called_once()
    # Backend name and reason are passed as format args.
    assert "RESTRICTED" in mock_warn.call_args.args


def test_compatible_explicit_selection_passes(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        backend = selector.get_attn_backend(64, torch.float16, SUPPORTED_BACKENDS)
    assert backend is _RestrictedBackend
    mock_warn.assert_not_called()
