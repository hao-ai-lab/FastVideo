# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

import fastvideo.attention.selector as selector
import fastvideo.platforms as platforms
from fastvideo.platforms import AttentionBackendEnum


class _FakePlatform:

    device_name = "fake"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        del cls, head_size, dtype
        assert selected_backend is not None
        return selected_backend.name


def test_role_override_invalidates_cached_backend(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "_current_platform", _FakePlatform())
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", lambda name: name)
    selector.global_force_attn_backend(None)

    supported = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    kwargs = {
        "head_size": 64,
        "dtype": torch.bfloat16,
        "supported_attention_backends": supported,
        "default_backend": AttentionBackendEnum.FLASH_ATTN,
    }

    try:
        assert selector.get_attn_backend(**kwargs) == "FLASH_ATTN"
        with selector.global_force_attn_backend_context_manager(
                AttentionBackendEnum.TORCH_SDPA):
            # Same cache key as above; the role-local override must still win.
            assert selector.get_attn_backend(**kwargs) == "TORCH_SDPA"

        # Exiting the role scope restores the previous resolution as well.
        assert selector.get_attn_backend(**kwargs) == "FLASH_ATTN"
    finally:
        # Do not leave fake-platform resolutions cached for later tests.
        selector.global_force_attn_backend(None)


def test_explicit_backend_config_rejects_typos() -> None:
    try:
        selector.coerce_attn_backend("attn_qat_typo")
    except ValueError as exc:
        assert "Unknown attention backend" in str(exc)
    else:
        raise AssertionError("invalid backend name was accepted")
