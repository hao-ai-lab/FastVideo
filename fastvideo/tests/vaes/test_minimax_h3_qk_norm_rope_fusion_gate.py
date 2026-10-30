# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.models.vaes.minimax_h3_video import _can_fuse_qk_norm_rope, _rotary_tables_fit


@pytest.fixture
def cuda_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def _query_key(batch: int = 2, seq_len: int = 5, heads: int = 4, head_dim: int = 16):
    return (torch.randn(batch, seq_len, heads, head_dim), torch.randn(batch, seq_len, heads, head_dim))


def test_packed_rotary_tables_fuse(cuda_tensors) -> None:
    query, key = _query_key()
    cos = torch.randn(2, 5, 1, 8)
    sin = torch.randn(2, 5, 1, 8)

    assert _can_fuse_qk_norm_rope(query, key, (cos, sin))


def test_broadcast_rotary_tables_stay_eager(cuda_tensors) -> None:
    query, key = _query_key()
    cos = torch.randn(1, 5, 1, 8)
    sin = torch.randn(1, 5, 1, 8)

    assert not _can_fuse_qk_norm_rope(query, key, (cos, sin))
    assert not _can_fuse_qk_norm_rope(query, key, (cos.expand(2, -1, -1, -1), sin.expand(2, -1, -1, -1)))


def test_non_contiguous_rotary_tables_stay_eager(cuda_tensors) -> None:
    query, key = _query_key()
    cos = torch.randn(2, 5, 1, 16)[..., :8]
    sin = torch.randn(2, 5, 1, 16)[..., :8]

    assert not cos.is_contiguous() and not sin.is_contiguous()
    assert not _can_fuse_qk_norm_rope(query, key, (cos, sin))
    assert not _rotary_tables_fit(query, cos, sin)


def test_rotary_tables_must_match_query_tokens(cuda_tensors) -> None:
    query, key = _query_key()
    cos = torch.randn(2, 6, 1, 8)
    sin = torch.randn(2, 6, 1, 8)

    assert not _can_fuse_qk_norm_rope(query, key, (cos, sin))


def test_grad_required_stays_eager(cuda_tensors) -> None:
    query, key = _query_key()
    query.requires_grad_(True)
    cos = torch.randn(2, 5, 1, 8)
    sin = torch.randn(2, 5, 1, 8)

    assert torch.is_grad_enabled()
    assert not _can_fuse_qk_norm_rope(query, key, (cos, sin))
    with torch.no_grad():
        assert _can_fuse_qk_norm_rope(query, key, (cos, sin))
