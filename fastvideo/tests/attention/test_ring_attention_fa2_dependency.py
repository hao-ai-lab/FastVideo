# SPDX-License-Identifier: Apache-2.0
"""Construction-time validation of the FA2-only Ring kernel."""
import pytest

import fastvideo.attention.ring_attention as ring_module
from fastvideo.platforms import AttentionBackendEnum


@pytest.mark.parametrize(
    ("ring_size", "has_fa2", "expect_error"),
    [(1, False, False), (2, False, True), (2, True, False)],
)
def test_ring_fa2_dependency(monkeypatch, ring_size, has_fa2, expect_error):
    # No real process groups or GPU kernels are needed for construction.
    monkeypatch.setattr(ring_module, "get_ring_size", lambda: ring_size)
    monkeypatch.setattr(ring_module, "get_sp_world_size", lambda: 2)
    def load_fa2():
        if not has_fa2:
            raise RuntimeError("Ring Attention requires FlashAttention-2 (flash_attn.flash_attn_interface); ring_size=1")
    monkeypatch.setattr(ring_module, "load_flash_attn", load_fa2)
    kwargs = dict(num_heads=4, num_kv_heads=4, softmax_scale=0.125,
                  causal=False, backend=AttentionBackendEnum.FLASH_ATTN)
    if expect_error:
        with pytest.raises(RuntimeError, match="requires FlashAttention-2") as exc:
            ring_module.RingAttention.create_if_enabled(**kwargs)
        assert "ring_size=1" in str(exc.value)
        assert "flash_attn.flash_attn_interface" in str(exc.value)
    else:
        attention = ring_module.RingAttention.create_if_enabled(**kwargs)
        assert (attention is None) == (ring_size == 1)


@pytest.mark.parametrize("heads,ring_size,valid", [(12, 8, True), (12, 2, True), (10, 2, False)])
def test_head_divisibility_uses_ulysses(monkeypatch, heads, ring_size, valid):
    monkeypatch.setattr(ring_module, "get_ring_size", lambda: ring_size)
    monkeypatch.setattr(ring_module, "get_sp_world_size", lambda: 8)
    monkeypatch.setattr(ring_module, "load_flash_attn", lambda: None)
    kwargs = dict(num_heads=heads, num_kv_heads=heads, softmax_scale=0.125,
                  causal=False, backend=AttentionBackendEnum.FLASH_ATTN)
    if valid:
        ring_module.RingAttention.create_if_enabled(**kwargs)
    else:
        with pytest.raises(NotImplementedError, match="divisible"):
            ring_module.RingAttention.create_if_enabled(**kwargs)


def test_causal_rejected_before_fa2_load(monkeypatch):
    monkeypatch.setattr(ring_module, "get_ring_size", lambda: 2)
    monkeypatch.setattr(ring_module, "get_sp_world_size", lambda: 2)
    def unexpected_load():
        pytest.fail("causal configuration must fail before loading FA2")
    monkeypatch.setattr(ring_module, "load_flash_attn", unexpected_load)
    with pytest.raises(NotImplementedError, match="non-causal"):
        ring_module.RingAttention.create_if_enabled(
            num_heads=4, num_kv_heads=4, softmax_scale=0.125,
            causal=True, backend=AttentionBackendEnum.FLASH_ATTN)


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_fa2_loader_reports_missing_or_broken_install(monkeypatch, error):
    import builtins
    from fastvideo.attention.ring.capabilities import load_flash_attn
    original_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise error("unavailable FA2")
        return original_import(name, *args, **kwargs)
    load_flash_attn.cache_clear()
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="requires FlashAttention-2"):
        load_flash_attn()
    load_flash_attn.cache_clear()


def test_ring_modules_import_without_loading_fa2():
    import subprocess
    import sys
    code = '''
import builtins
import importlib.util
import sys
import types
from pathlib import Path
import torch
root = Path("fastvideo/attention/ring")
# Load the vendor modules without importing unrelated FastVideo backends.
package = types.ModuleType("ring_import_check")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "flash_attn" or name.startswith("flash_attn."):
        raise AssertionError("FA2 imported before Ring was enabled")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import ring_import_check.capabilities
import ring_import_check.kernels.attention
'''
    subprocess.run([sys.executable, "-c", code], check=True)
