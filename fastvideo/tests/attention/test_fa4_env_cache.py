# SPDX-License-Identifier: Apache-2.0
"""FA4 JIT-cache invalidation on FA4_*/FA_* tuning-env changes.

The CPU tests cover the fingerprint + invalidation logic with fake kernel
objects, so they run on any machine. The last test needs a real FA4 install and
a Blackwell/Hopper GPU: it shows that a knob change inside one process reuses
the first arm's kernel without the guard (the bug) and compiles a second kernel
with it (the fix).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.attention.utils import fa4_env_cache


@pytest.fixture(autouse=True)
def _fresh_fingerprint():
    fa4_env_cache.reset_fa4_env_fingerprint()
    yield
    fa4_env_cache.reset_fa4_env_fingerprint()


def _fake_kernels(*, fwd_entries: dict | None = None, bwd_entries: dict | None = None):
    fwd = SimpleNamespace(compile_cache={} if fwd_entries is None else dict(fwd_entries))
    bwd = SimpleNamespace(
        compile_cache={} if bwd_entries is None else dict(bwd_entries),
        compile_cache_pre={},
        compile_cache_post={},
    )
    return fwd, bwd


def test_fingerprint_selects_only_fa4_tuning_vars() -> None:
    environ = {
        "FA4_FP8_PV_USE_FUSED_PACK": "1",
        "FA_DISABLE_2CTA": "0",
        "FASTVIDEO_FA4": "1",  # a FastVideo flag, not an FA4 kernel knob
        "PATH": "/usr/bin",
    }
    assert fa4_env_cache.fa4_env_fingerprint(environ) == (
        ("FA4_FP8_PV_USE_FUSED_PACK", "1"),
        ("FA_DISABLE_2CTA", "0"),
    )


def test_empty_environment_has_empty_fingerprint() -> None:
    assert fa4_env_cache.fa4_env_fingerprint({}) == ()


def test_first_call_drops_preexisting_kernels() -> None:
    fwd, bwd = _fake_kernels(fwd_entries={"a": object()}, bwd_entries={"b": object()})

    cleared = fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {"FA4_X": "1"})

    assert cleared is True
    assert fwd.compile_cache == {}
    assert bwd.compile_cache == {}


def test_unchanged_environment_keeps_the_compiled_kernel() -> None:
    fwd, bwd = _fake_kernels()
    environ = {"FA4_X": "1"}
    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), environ) is True
    fwd.compile_cache["a"] = object()

    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), environ) is False
    assert list(fwd.compile_cache) == ["a"]


def test_changed_knob_drops_the_compiled_kernel() -> None:
    fwd, bwd = _fake_kernels()
    fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {"FA4_X": "1"})
    fwd.compile_cache["a"] = object()

    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {"FA4_X": "0"}) is True
    assert fwd.compile_cache == {}

    fwd.compile_cache["a"] = object()
    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {"FA4_X": "1"}) is True
    assert fwd.compile_cache == {}


def test_unrelated_environment_changes_do_not_drop_kernels() -> None:
    fwd, bwd = _fake_kernels()
    fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {"FA4_X": "1"})
    fwd.compile_cache["a"] = object()

    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, bwd), {
        "FA4_X": "1",
        "CUDA_VISIBLE_DEVICES": "3",
    }) is False
    assert list(fwd.compile_cache) == ["a"]


class _JITCacheLike:
    """Shape of ``flash_attn.cute.cache_utils.JITCache`` at the pinned FA4 ref: not a dict."""

    def __init__(self, entries: dict) -> None:
        self.cache = dict(entries)

    def __contains__(self, key) -> bool:
        return key in self.cache

    def clear(self) -> None:
        self.cache.clear()


def test_jit_cache_objects_are_cleared_through_their_own_clear() -> None:
    """The pinned FA4 wraps its compile cache in a JITCache; a dict-only check cleared nothing."""
    fwd = SimpleNamespace(compile_cache=_JITCacheLike({"a": object(), "b": object()}))
    bwd = SimpleNamespace(compile_cache=_JITCacheLike({}), compile_cache_pre=_JITCacheLike({"c": object()}))

    assert fa4_env_cache.clear_fa4_jit_caches((fwd, bwd)) == 3
    assert fwd.compile_cache.cache == {}
    assert bwd.compile_cache_pre.cache == {}

    fwd.compile_cache.cache["a"] = object()
    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, ), {"FA_X": "1"}) is True
    fwd.compile_cache.cache["a"] = object()
    assert fa4_env_cache.invalidate_fa4_jit_caches_on_env_change((fwd, ), {"FA_X": "0"}) is True
    assert "a" not in fwd.compile_cache


def test_missing_or_non_dict_caches_are_ignored() -> None:
    bare = SimpleNamespace()
    non_dict = SimpleNamespace(compile_cache=["not", "a", "dict"])

    assert fa4_env_cache.clear_fa4_jit_caches((bare, non_dict)) == 0
    assert non_dict.compile_cache == ["not", "a", "dict"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs a CUDA device")
def test_real_fa4_kernel_is_recompiled_for_each_knob_value(monkeypatch) -> None:
    """Regression test for the silent stale-kernel reuse.

    Without the guard the second knob value is served the first value's kernel
    (one ``cute.compile`` for two settings). With the guard each setting compiles
    its own kernel.
    """
    interface = pytest.importorskip("flash_attn.cute.interface")
    pytest.importorskip("flash_attn.cute.flash_fwd_sm100")
    from fastvideo.attention.utils import flash_attn_cute

    torch.manual_seed(0)
    q = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16)

    compiles = {"n": 0}
    original_compile = interface.cute.compile

    def counting_compile(*args, **kwargs):
        compiles["n"] += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(interface.cute, "compile", counting_compile)

    def two_arms() -> int:
        interface._flash_attn_fwd.compile_cache.clear()
        fa4_env_cache.reset_fa4_env_fingerprint()
        compiles["n"] = 0
        for value in ("1", "0"):
            with envs.override_external("FA_DISABLE_2CTA", value):
                flash_attn_cute.flash_attn_func(q, k, v)
        return compiles["n"]

    monkeypatch.setattr(flash_attn_cute, "_guard_fa4_jit_cache", lambda: None)
    assert two_arms() == 1

    monkeypatch.undo()
    monkeypatch.setattr(interface.cute, "compile", counting_compile)
    assert two_arms() == 2
