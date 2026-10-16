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


@pytest.mark.parametrize("window,chunk,sink", [(3, 1, 0), (3, 1, 1), (6, 3, 0), (9, 3, 1)])
@pytest.mark.parametrize("tensor_counters", [False, True])
@pytest.mark.parametrize("explicit_chunk_start", [False, True])
def test_cache_rollout_matches_independent_rolling_window(window, chunk, sink, tensor_counters, explicit_chunk_start):
    """Check contents, old offset expressions and sink preservation over 32 chunks."""
    capacity, tokens, sink_tokens = window * 2, chunk * 2, sink * 2
    cache = {"k": torch.zeros(1, capacity, 1, 1), "v": torch.zeros(1, capacity, 1, 1),
             "global_end_index": torch.tensor([0]) if tensor_counters else 0,
             "local_end_index": torch.tensor([0]) if tensor_counters else 0}
    expected = []
    for block in range(32):
        current_end = (block + 1) * tokens
        for step in range(3):
            values = list(range((block * 3 + step) * tokens + 1, (block * 3 + step + 1) * tokens + 1))
            key = torch.tensor(values, dtype=torch.float32).reshape(1, tokens, 1, 1)
            global_end = int(cache["global_end_index"])
            local_end = int(cache["local_end_index"])
            evicts = current_end > global_end and tokens + local_end > capacity
            old_end = local_end + current_end - global_end - (tokens + local_end - capacity if evicts else 0)
            if step == 0:
                expected += values
                if len(expected) > capacity:
                    expected = expected[:sink_tokens] + expected[-(capacity - sink_tokens):]
            else:
                expected[-tokens:] = values
            flag = step == 0 if explicit_chunk_start else None
            if evicts:
                start, end = CausalWanSelfAttention._evict_and_write(
                    cache, key, -key, current_end=current_end, global_end_index=global_end,
                    local_end_index_prev=local_end, sink_tokens=sink_tokens)
            else:
                start, end = CausalWanSelfAttention._write_in_place(
                    cache, key, -key, current_end=current_end, global_end_index=global_end,
                    local_end_index_prev=local_end, is_chunk_start=flag)
            assert (start, end) == (old_end - tokens, old_end)
            CausalWanSelfAttention._update_cache_counters(cache, current_end, end)
            torch.testing.assert_close(cache["k"][0, :end, 0, 0], torch.tensor(expected, dtype=torch.float32),
                                       atol=0, rtol=0)
            assert torch.equal(cache["v"], -cache["k"])
            assert int(cache["global_end_index"]) == current_end
            assert int(cache["local_end_index"]) == min((block + 1) * tokens, capacity)


def test_eviction_preserves_pre_refactor_offset_for_noncontiguous_progress():
    cache = {"k": torch.arange(12.).reshape(1, 12, 1, 1), "v": torch.arange(12.).reshape(1, 12, 1, 1)}
    key = torch.ones(1, 4, 1, 1)
    assert CausalWanSelfAttention._evict_and_write(
        cache, key, key, current_end=14, global_end_index=12, local_end_index_prev=12, sink_tokens=2) == (6, 10)
