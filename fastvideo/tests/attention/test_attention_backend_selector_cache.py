from types import SimpleNamespace

import pytest
import torch

import fastvideo.platforms as platforms
from fastvideo.attention import selector
from fastvideo.platforms.interface import AttentionBackendEnum


class FlashBackend:
    pass


class VideoSparseBackend:
    pass


@pytest.fixture(autouse=True)
def reset_attention_backend_selector():
    selector.global_force_attn_backend(None)
    yield
    selector.global_force_attn_backend(None)


@pytest.fixture
def stub_backend_resolution(monkeypatch):
    requested_backends = []

    def get_attn_backend_cls(selected_backend, head_size, dtype):
        requested_backends.append(selected_backend)
        return selected_backend.name

    platform = SimpleNamespace(
        device_name="test device",
        get_attn_backend_cls=get_attn_backend_cls,
    )
    backend_classes = {
        AttentionBackendEnum.FLASH_ATTN.name: FlashBackend,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN.name: VideoSparseBackend,
    }
    monkeypatch.setattr(platforms, "_current_platform", platform)
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", backend_classes.__getitem__)
    return requested_backends


def test_changing_env_backend_cannot_reuse_cached_backend(monkeypatch, stub_backend_resolution):
    supported_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN,
    )
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)

    first_backend = selector.get_attn_backend(64, torch.float16, supported_backends)

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.VIDEO_SPARSE_ATTN.name)
    second_backend = selector.get_attn_backend(64, torch.float16, supported_backends)

    assert first_backend is FlashBackend
    assert second_backend is VideoSparseBackend
    assert stub_backend_resolution == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN,
    ]


def test_invalid_env_backend_cannot_be_hidden_by_cache_hit(monkeypatch, stub_backend_resolution):
    supported_backends = (AttentionBackendEnum.FLASH_ATTN, )
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.FLASH_ATTN.name)
    assert selector.get_attn_backend(64, torch.float16, supported_backends) is FlashBackend

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATN")

    with pytest.raises(ValueError, match="FLASH_ATN"):
        selector.get_attn_backend(64, torch.float16, supported_backends)
    assert stub_backend_resolution == [AttentionBackendEnum.FLASH_ATTN]


def test_global_force_still_takes_precedence_over_env(monkeypatch, stub_backend_resolution):
    supported_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN,
    )
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", AttentionBackendEnum.VIDEO_SPARSE_ATTN.name)
    selector.global_force_attn_backend(AttentionBackendEnum.FLASH_ATTN)

    backend = selector.get_attn_backend(64, torch.float16, supported_backends)

    assert backend is FlashBackend
    assert stub_backend_resolution == [AttentionBackendEnum.FLASH_ATTN]
