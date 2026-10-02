# SPDX-License-Identifier: Apache-2.0
"""CPU checks for padding bookkeeping; CUDA parity lives in distributed/."""
import importlib

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("seq_len", [1, 7, 8, 15, 16])
@pytest.mark.parametrize("rank", range(4))
def test_ring_padding_matches_dense_attention(monkeypatch, seq_len, rank):
    ring = importlib.import_module("fastvideo.attention.ring.ring_flash_attn")
    world_size, shard_len = 4, 4
    generator = torch.Generator().manual_seed(71)
    q, k, v = [torch.randn(2, seq_len, 2, 8, generator=generator) for _ in range(3)]
    padded = [F.pad(t, (0, 0, 0, 0, 0, world_size * shard_len - seq_len)) for t in (q, k, v)]
    local = [t[:, rank * shard_len:(rank + 1) * shard_len].contiguous() for t in padded]

    class FakeRingComm:
        def __init__(self, group):
            self.rank, self.world_size = rank, world_size
            self.calls = 0

        def send_recv(self, tensor):
            assert tensor.shape == local[1].shape  # Never send shortened buffers.
            owner = (rank - self.calls // 2 - 1) % world_size
            source = padded[1 + self.calls % 2]
            self.calls += 1
            return source[:, owner * shard_len:(owner + 1) * shard_len].contiguous()

        def commit(self):
            pass

        def wait(self):
            pass

    def dense_block(q, k, v, **kwargs):
        assert k.shape[1] > 0
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * kwargs["softmax_scale"]
        return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v), scores.logsumexp(-1)

    monkeypatch.setattr(ring, "RingComm", FakeRingComm)
    monkeypatch.setattr(ring, "_FIRST_RING_LOG", False)
    monkeypatch.setattr(ring, "select_flash_attn_impl", lambda *a, **kw: dense_block)
    actual, lse = ring.ring_flash_attn_forward(
        None, *local, softmax_scale=8**-0.5, causal=False, original_seq_len=seq_len,
    )
    expected, _ = dense_block(local[0], k, v, softmax_scale=8**-0.5)
    valid = max(0, min(shard_len, seq_len - rank * shard_len))
    expected[:, valid:] = 0
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_rope_padding_uses_identity(monkeypatch, rank):
    module = importlib.import_module("fastvideo.attention.ring_attention")
    monkeypatch.setattr(module, "get_ring_rank", lambda: rank)
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    actual_cos, actual_sin = module.RingAttention._slice_local_rope((cos, sin), 2, 5)
    for i in range(2):
        pos = rank * 2 + i
        torch.testing.assert_close(actual_cos[i], cos[pos] if pos < 5 else torch.ones(8))
        torch.testing.assert_close(actual_sin[i], sin[pos] if pos < 5 else torch.zeros(8))
    with pytest.raises(ValueError, match="shorter"):
        module.RingAttention._slice_local_rope((cos[:1], sin[:1]), 2, 5)
