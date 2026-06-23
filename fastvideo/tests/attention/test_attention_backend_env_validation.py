import pytest

import fastvideo.envs as envs
from fastvideo.platforms.interface import AttentionBackendEnum


def test_unknown_attention_backend_env_fails_loudly(monkeypatch):
    # Given: a typo'd backend name in the env var.
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATN")

    # Then: reading the env var raises instead of silently auto-selecting,
    # and the error names the offending value plus the valid backends.
    with pytest.raises(ValueError) as excinfo:
        _ = envs.FASTVIDEO_ATTENTION_BACKEND
    message = str(excinfo.value)
    assert "FLASH_ATN" in message
    assert AttentionBackendEnum.FLASH_ATTN.name in message


def test_valid_attention_backend_env_passes_through(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.VIDEO_SPARSE_ATTN.name)
    assert envs.FASTVIDEO_ATTENTION_BACKEND == AttentionBackendEnum.VIDEO_SPARSE_ATTN.name


def test_unset_attention_backend_env_is_none(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    assert envs.FASTVIDEO_ATTENTION_BACKEND is None
