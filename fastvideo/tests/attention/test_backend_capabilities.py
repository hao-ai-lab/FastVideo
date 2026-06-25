# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AttentionBackend capability self-description API.

These exercise the additive capability hooks added for #1254 using a small
dummy backend, so they run on CPU without GPU kernels or optional
third-party attention packages installed.
"""
import torch

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)


class _DummyBackend(AttentionBackend):
    """Minimal concrete backend that relies on the base-class defaults."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class _RestrictedBackend(_DummyBackend):
    """Overrides the capability hooks to a restrictive set."""

    @staticmethod
    def get_name() -> str:
        return "RESTRICTED"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False


def test_defaults_are_permissive():
    assert _DummyBackend.is_available() is True
    assert _DummyBackend.get_supported_head_sizes() is None
    assert _DummyBackend.supports_varlen() is False
    assert _DummyBackend.supports_attention_mask() is True
    assert torch.float16 in _DummyBackend.get_supported_dtypes()
    assert torch.bfloat16 in _DummyBackend.get_supported_dtypes()


def test_compatible_request_returns_none():
    # No head-size restriction, supported dtype, no mask required.
    assert _DummyBackend.validate_compatibility(123, torch.float16) is None


def test_unsupported_dtype_reports_reason():
    reason = _DummyBackend.validate_compatibility(64, torch.float32)
    assert reason is not None
    assert "dtype" in reason
    assert "DUMMY" in reason


def test_unsupported_head_size_reports_reason():
    reason = _RestrictedBackend.validate_compatibility(96, torch.float16)
    assert reason is not None
    assert "head_size" in reason


def test_supported_head_size_passes():
    assert _RestrictedBackend.validate_compatibility(128, torch.float16) is None


def test_mask_requirement_reports_reason():
    reason = _RestrictedBackend.validate_compatibility(64, torch.float16, needs_attention_mask=True)
    assert reason is not None
    assert "mask" in reason


def test_real_backends_declare_capabilities():
    # Backends that ship head-size lists should expose them through the hook.
    from fastvideo.attention.backends.sdpa import SDPABackend
    assert SDPABackend.get_supported_head_sizes() is not None
    assert torch.float32 in SDPABackend.get_supported_dtypes()
