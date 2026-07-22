"""T0: attention backend selection — pure logic, no torch, no GPU."""
import pytest

from fastvideo2.layers.attention import BACKENDS, backend_policy, select_backend


def test_auto_prefers_flash_when_usable():
    assert select_backend(None, flash_available=True, on_cuda=True,
                          true_fp32=False) == "FLASH_ATTN"


def test_auto_uses_sdpa_for_true_fp32_and_off_cuda_and_no_flash():
    # the exact-math anchor path: flash cannot run fp32
    assert select_backend(None, flash_available=True, on_cuda=True,
                          true_fp32=True) == "SDPA"
    assert select_backend(None, flash_available=True, on_cuda=False,
                          true_fp32=False) == "SDPA"
    assert select_backend(None, flash_available=False, on_cuda=True,
                          true_fp32=False) == "SDPA"


def test_explicit_request_wins_and_fails_closed():
    assert select_backend("SDPA", flash_available=True, on_cuda=True,
                          true_fp32=False) == "SDPA"
    # forcing flash in true-fp32 mode keeps official's cast-to-bf16 behavior
    assert select_backend("flash_attn", flash_available=True, on_cuda=True,
                          true_fp32=True) == "FLASH_ATTN"
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        select_backend("FLASH_ATTN", flash_available=False, on_cuda=True,
                       true_fp32=False)
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        select_backend("FLASH_ATTN", flash_available=True, on_cuda=False,
                       true_fp32=False)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend("SAGE", flash_available=True, on_cuda=True, true_fp32=False)


def test_policy_reads_env(monkeypatch):
    monkeypatch.delenv("FASTVIDEO2_ATTENTION_BACKEND", raising=False)
    assert backend_policy() == "AUTO"
    monkeypatch.setenv("FASTVIDEO2_ATTENTION_BACKEND", "sdpa")
    assert backend_policy() == "SDPA"
    assert set(BACKENDS) == {"FLASH_ATTN", "SDPA"}
