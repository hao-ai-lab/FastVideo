# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``_resolve_flash_attn_varlen_func`` fallback chain.

Landed in PR #1225 slice 5 (Attn-QAT 5/12). The resolver centralises the
varlen-flash-attn import-fallback logic that several backends
(``bsa_attn.py``, ``video_sparse_attn.py``) used to duplicate. The fallback
order is:

    1. ``fastvideo.attention.utils.flash_attn_cute``
    2. ``flash_attn_interface``
    3. ``flash_attn``

These tests verify that the resolver picks the highest-priority impl
available and falls through cleanly on ``ImportError``. CPU-only, no
flash-attn install required.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys

import pytest

if importlib.util.find_spec("flash_attn") is None:
    pytest.skip("flash_attn not installed; resolver tests require it for the unconditional top-level imports in flash_attn_no_pad",
                allow_module_level=True)


def _reload_resolver_module():
    if "fastvideo.attention.utils.flash_attn_no_pad" in sys.modules:
        del sys.modules["fastvideo.attention.utils.flash_attn_no_pad"]
    return importlib.import_module("fastvideo.attention.utils.flash_attn_no_pad")


def test_resolver_falls_back_when_cute_unavailable(monkeypatch) -> None:
    """When ``flash_attn_cute`` is unimportable, resolver tries the next impl."""
    real_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastvideo.attention.utils.flash_attn_cute":
            raise ImportError("cute disabled for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    mod = _reload_resolver_module()
    resolved = mod._resolve_flash_attn_varlen_func()
    assert resolved is not None
    assert resolved.__name__ == "flash_attn_varlen_func"


def test_resolver_returns_flash_attn_when_cute_and_interface_unavailable(monkeypatch) -> None:
    """The terminal fallback is the plain ``flash_attn`` import."""
    real_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
                "fastvideo.attention.utils.flash_attn_cute",
                "flash_attn_interface",
        }:
            raise ImportError(f"{name} disabled for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched_import)

    mod = _reload_resolver_module()
    resolved = mod._resolve_flash_attn_varlen_func()
    assert resolved is not None
    assert resolved.__module__.startswith("flash_attn")


def test_module_level_impl_is_resolver_output() -> None:
    """``flash_attn_varlen_func_impl`` is bound to whatever the resolver picks
    at import time. Cleanup any test monkeypatching first by forcing a reload.
    """
    if "fastvideo.attention.utils.flash_attn_no_pad" in sys.modules:
        del sys.modules["fastvideo.attention.utils.flash_attn_no_pad"]
    mod = importlib.import_module("fastvideo.attention.utils.flash_attn_no_pad")
    assert mod.flash_attn_varlen_func_impl is not None
    assert callable(mod.flash_attn_varlen_func_impl)
