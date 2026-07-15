# SPDX-License-Identifier: Apache-2.0
"""Tests for selector-side capability validation (#1254).

The selector checks the resolved backend against its self-described
capabilities (AttentionBackend.validate_compatibility): an explicitly
selected backend hard-fails only when the resolution honored the
selection and that very backend is incompatible, mirroring how the
platform layer hard-fails on missing explicitly-requested backends. A
resolution that fell back to a different backend (the pin is outside the
layer's supported set, or the platform substituted a fallback such as
SDPA for an unsupported head size) only warns -- once per cached
resolution -- as does auto-selection.
These use dummy backends and a stubbed platform, so they run on CPU.
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
    """Backend that only supports head sizes 64 and 128.

    Named FLASH_ATTN so the stubbed platform models an honored explicit
    selection: the resolved class reports the selected backend's name.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

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


class _FallbackBackend(_RestrictedBackend):
    """A platform-substituted fallback: same restrictions, different name."""

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"


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


def test_explicitly_selected_incompatible_sdpa_backend_raises(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.TORCH_SDPA.name)
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", {"RESTRICTED": _FallbackBackend}.__getitem__)
    with pytest.raises(ValueError, match="head_size"):
        selector.get_attn_backend(96, torch.float16, (AttentionBackendEnum.TORCH_SDPA, ))


def test_auto_selected_incompatible_backend_warns_once():
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        first = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)
        # A second lookup is a cache hit and must not warn again.
        second = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)
    assert first is _RestrictedBackend
    assert second is _RestrictedBackend
    mock_warn.assert_called_once()
    # Backend name and reason are passed as format args.
    assert "FLASH_ATTN" in mock_warn.call_args.args


def test_compatible_explicit_selection_passes(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        backend = selector.get_attn_backend(64, torch.float16, SUPPORTED_BACKENDS)
    assert backend is _RestrictedBackend
    mock_warn.assert_not_called()


def test_pinned_backend_platform_fallback_warns_instead_of_raising(monkeypatch):
    # matrixgame2's CLIP vision encoder under a pinned FLASH_ATTN: the layer's
    # head size (80) makes the platform substitute SDPA, whose declared
    # capabilities also reject the head size. The pin was never honored for
    # this layer, so the load must warn (once) instead of hard-failing.
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    platform = SimpleNamespace(
        device_name="test device",
        get_attn_backend_cls=lambda selected_backend, head_size, dtype: "FALLBACK",
    )
    monkeypatch.setattr(platforms, "_current_platform", platform)
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", {"FALLBACK": _FallbackBackend}.__getitem__)

    with mock.patch.object(selector.logger, "warning") as mock_warn:
        first = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)
        # A second lookup is a cache hit and must not warn again.
        second = selector.get_attn_backend(96, torch.float16, SUPPORTED_BACKENDS)

    assert first is _FallbackBackend
    assert second is _FallbackBackend
    mock_warn.assert_called_once()
    assert "TORCH_SDPA" in mock_warn.call_args.args


def test_pin_outside_layer_supported_set_falls_back_with_warning(monkeypatch):
    # A layer whose declared supported set excludes the pinned backend (e.g. an
    # SDPA-only aux encoder) never participates in the global pin: it falls
    # back to automatic selection with a warning instead of raising.
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    with mock.patch.object(selector.logger, "warning") as mock_warn:
        backend = selector.get_attn_backend(64, torch.float16, (AttentionBackendEnum.TORCH_SDPA, ))
    assert backend is _RestrictedBackend
    mock_warn.assert_called_once()
