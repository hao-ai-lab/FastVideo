# SPDX-License-Identifier: Apache-2.0
"""``DistributedAttention.forward`` is the stacked path plus a ``torch.cat``."""
import torch

from fastvideo.attention.layer import DistributedAttention


class _Recorder:

    def __init__(self):
        self.qkv = None
        self.kwargs = None


def test_forward_cats_then_delegates_to_forward_stacked():
    rec = _Recorder()

    def fake_forward_stacked(self, qkv, **kwargs):
        rec.qkv = qkv
        rec.kwargs = kwargs
        return torch.zeros(2, 3, 4, 5), None

    original = DistributedAttention.forward_stacked
    DistributedAttention.forward_stacked = fake_forward_stacked
    try:
        q = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        k = q + 1
        v = q + 2
        # Bypass __init__ (it builds a real backend); ``forward`` only needs the
        # compiler-policy flag and the patched ``forward_stacked``.
        dummy = object.__new__(DistributedAttention)
        dummy._compile_forward_enabled = False
        out, replicated = DistributedAttention.forward(dummy, q, k, v, original_seq_len=3)
    finally:
        DistributedAttention.forward_stacked = original

    assert replicated is None
    assert rec.qkv.shape == (6, 3, 4, 5)
    assert torch.equal(rec.qkv[:2], q) and torch.equal(rec.qkv[2:4], k) and torch.equal(rec.qkv[4:], v)
    assert rec.kwargs["original_seq_len"] == 3
