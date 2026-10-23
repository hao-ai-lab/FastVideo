# SPDX-License-Identifier: Apache-2.0
"""The causal KV cache keeps its window offsets as plain Python ints.

``WanCausalDenoisingBase._initialize_kv_cache`` (shared by
``CausalDMDDenosingStage`` and ``CausalDenoisingStage``) stores ``global_end_index`` /
``local_end_index`` as ints so the attention block can slice the cache without
a host sync per layer per step. ``CausalWanSelfAttention`` still accepts a
device tensor for callers that hand it one, so both spellings must stay
equivalent.
"""

from types import SimpleNamespace

import pytest
import torch

import fastvideo.models.wan.causal_transformer as causal_transformer
from fastvideo.forward_context import set_forward_context
from fastvideo.models.wan.causal_transformer import CausalWanSelfAttention
from fastvideo.pipelines.basic.wan.stages.causal_denoising import CausalDMDDenosingStage

NUM_HEADS = 2
HEAD_DIM = 8
SEQ_LEN = 4
FRAME_SEQLEN = 4


def _fake_transformer() -> SimpleNamespace:
    """Only the attributes the stage reads at construction and cache init."""
    return SimpleNamespace(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        blocks=[None, None],
        config=SimpleNamespace(
            arch_config=SimpleNamespace(num_frames_per_block=1, sliding_window_num_frames=2)),
        local_attn_size=-1,
        sink_size=0,
    )


def test_initialize_kv_cache_stores_int_offsets():
    """The offsets are ints, not device tensors, so the attention never syncs."""
    stage = CausalDMDDenosingStage(_fake_transformer(), scheduler=SimpleNamespace())
    stage.frame_seq_length = FRAME_SEQLEN

    kv_cache = stage._initialize_kv_cache(batch_size=1, dtype=torch.float32, device=torch.device("cpu"))

    assert len(kv_cache) == 2
    for entry in kv_cache:
        assert isinstance(entry["global_end_index"], int)
        assert isinstance(entry["local_end_index"], int)
        assert entry["global_end_index"] == 0
        assert entry["local_end_index"] == 0


@pytest.fixture
def single_rank_sp(monkeypatch):
    """The attention splits heads across the SP group; stand in a one-rank group."""
    monkeypatch.setattr(causal_transformer, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(causal_transformer, "get_sp_parallel_rank", lambda: 0)


def _run_attention(counter):
    """One self-attention step against a fresh cache; returns output and cache."""
    torch.manual_seed(0)
    attn = CausalWanSelfAttention(dim=NUM_HEADS * HEAD_DIM, num_heads=NUM_HEADS)
    q = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    freqs_cis = (torch.randn(SEQ_LEN, HEAD_DIM), torch.randn(SEQ_LEN, HEAD_DIM))
    kv_cache = {
        "k": torch.zeros(1, 2 * FRAME_SEQLEN, NUM_HEADS, HEAD_DIM),
        "v": torch.zeros(1, 2 * FRAME_SEQLEN, NUM_HEADS, HEAD_DIM),
        "global_end_index": counter(),
        "local_end_index": counter(),
    }
    with set_forward_context(current_timestep=0, attn_metadata=None):
        out = attn(q, k, v, freqs_cis, None, kv_cache=kv_cache, frame_seqlen=FRAME_SEQLEN)
    return out, kv_cache


def test_int_and_tensor_offsets_are_equivalent(single_rank_sp):
    """An int counter must take the same path as the legacy device-tensor counter."""
    out_int, cache_int = _run_attention(lambda: 0)
    out_tensor, cache_tensor = _run_attention(lambda: torch.tensor([0], dtype=torch.long))

    assert torch.equal(out_int, out_tensor)
    assert cache_int["global_end_index"] == SEQ_LEN
    assert cache_int["local_end_index"] == SEQ_LEN
    assert int(cache_tensor["global_end_index"].item()) == SEQ_LEN
    assert int(cache_tensor["local_end_index"].item()) == SEQ_LEN
