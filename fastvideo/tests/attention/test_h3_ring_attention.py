# SPDX-License-Identifier: Apache-2.0
"""H3 topology, partial RoPE, and padded Ring regression tests."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from fastvideo.attention.ring_attention import RingAttention
from fastvideo.attention.ring.ring_flash_attn import ring_chunk_valid_length
from fastvideo.platforms import AttentionBackendEnum


@pytest.mark.parametrize("length,expected", [(16, [4, 4, 4, 4]), (13, [4, 4, 4, 1]),
                                              (5, [4, 1, 0, 0]), (1, [1, 0, 0, 0])])
def test_ring_chunk_lengths(length, expected):
    assert [ring_chunk_valid_length(length, 4, rank) for rank in range(4)] == expected


@pytest.mark.parametrize("length", [0, -1, 9, 2.5, True])
def test_invalid_original_seq_len(monkeypatch, length):
    import fastvideo.attention.ring_attention as ring
    monkeypatch.setattr(ring, "get_sp_world_size", lambda: 2)
    attention = object.__new__(RingAttention)
    attention.causal = False
    with pytest.raises(ValueError, match="original_seq_len"):
        attention._validate_ring_inputs(torch.zeros(1, 4, 3, 128, dtype=torch.bfloat16), training=False,
                                        original_seq_len=length, replicated_q=None, replicated_k=None, replicated_v=None)


@pytest.mark.parametrize("sp,ulysses,heads,valid", [(4, 1, 3, True), (4, 2, 6, True),
                                                  (4, 2, 3, False), (4, 4, 6, False)])
def test_h3_heads_use_ulysses_size(monkeypatch, sp, ulysses, heads, valid):
    import fastvideo.models.dits.minimax_h3 as h3
    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
    monkeypatch.setattr(h3, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(h3, "get_sp_world_size", lambda: sp)
    monkeypatch.setattr(h3, "get_ulysses_size", lambda: ulysses)
    config = MiniMaxH3Config(arch_config=MiniMaxH3ArchConfig(
        num_attention_heads=heads, hidden_size=24, ffn_dim=48, num_layers=0, num_refiner_layers=0,
        text_dim=24, time_embed_hidden_dim=24, time_embed_dim=24))
    with torch.device("meta"):
        if valid:
            h3.MiniMaxH3Transformer3DModel(config, {})
        else:
            with pytest.raises(ValueError, match="Ulysses subgroup size"):
                h3.MiniMaxH3Transformer3DModel(config, {})


def test_h3_vsa_ring_conflict(monkeypatch):
    import fastvideo.models.dits.minimax_h3 as h3
    monkeypatch.setattr(h3, "get_ring_size", lambda: 2)
    monkeypatch.setattr(h3, "get_attn_backend", lambda *a, **k: SimpleNamespace(get_name=lambda: "VIDEO_SPARSE_ATTN_H3"))
    with pytest.raises(NotImplementedError, match="VSA-H3.*Ring"):
        h3.MiniMaxH3Attention(24, 3, 128, 1e-5, (AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,), None, "test")


def test_h3_ring_regional_compile_falls_back(monkeypatch):
    import fastvideo.models.dits.minimax_h3 as h3
    monkeypatch.setattr(h3, "get_ring_size", lambda: 2)
    assert "regional fullgraph" in h3.MiniMaxH3Transformer3DModel.prepare_for_regional_compile(None)


@pytest.mark.parametrize("sp", [2, 4])
def test_h3_partial_rope_sharding(monkeypatch, sp):
    import fastvideo.models.dits.minimax_h3 as h3
    import fastvideo.distributed.communication_op as comm
    torch.manual_seed(7)
    x = torch.randn(2, 13, 3, 128)
    rope = h3.MiniMaxH3RotaryPosEmbed(16, 10000.0)(torch.arange(39).reshape(13, 3))
    expected = h3.MiniMaxH3Attention._apply_rotary_emb(x, rope)
    monkeypatch.setattr(comm, "get_sp_world_size", lambda: sp)
    chunks = []
    for rank in range(sp):
        monkeypatch.setattr(comm, "get_sp_group", lambda: SimpleNamespace(
            shard=lambda tensor, dim, scale_grad: tensor.chunk(sp, dim=dim)[rank]))
        local, _ = comm.sequence_model_parallel_shard(x, dim=1)
        cos, _ = comm.sequence_model_parallel_shard(rope[0], dim=0)
        sin, _ = comm.sequence_model_parallel_shard(rope[1], dim=0)
        chunks.append(h3.MiniMaxH3Attention._apply_rotary_emb(local, (cos, sin)))
    actual = torch.cat(chunks, dim=1)[:, :13]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., 96:], x[..., 96:], rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 5, 13, 16])
def test_padded_ring_numerics_with_simulated_transport(monkeypatch, length):
    """Exercise real step/merge logic, including empty chunks and nonzero padding."""
    ring = importlib.import_module("fastvideo.attention.ring.ring_flash_attn")
    monkeypatch.setattr(ring, "_FIRST_RING_LOG", False)
    torch.manual_seed(3)
    q, k, v = [torch.randn(2, 16, 3, 8) for _ in range(3)]
    # Poison padding: any accidental participation changes the result sharply.
    k[:, length:] = 20
    v[:, length:] = 100

    def flash(q, k, v, softmax_scale, **kwargs):
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale
        return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v), scores.logsumexp(-1)

    monkeypatch.setattr(ring, "select_flash_attn_impl", lambda *a, **k: flash)
    outputs = []
    for rank in range(4):
        class FakeComm:
            world_size = 4

            def __init__(self, group):
                self.rank = rank
                self.calls = 0

            def send_recv(self, tensor):
                source = k if self.calls % 2 == 0 else v
                owner = (rank - self.calls // 2 - 1) % 4
                self.calls += 1
                assert tensor.shape[1] == 4
                return source[:, owner * 4:(owner + 1) * 4].contiguous()

            def commit(self):
                pass

            def wait(self):
                pass

        monkeypatch.setattr(ring, "RingComm", FakeComm)
        output, _ = ring.ring_flash_attn_forward(None, q[:, rank * 4:(rank + 1) * 4],
                                                k[:, rank * 4:(rank + 1) * 4],
                                                v[:, rank * 4:(rank + 1) * 4],
                                                softmax_scale=8**-0.5, causal=False, original_seq_len=length)
        assert output.shape == (2, 4, 3, 8)
        outputs.append(output)
    actual = torch.cat(outputs, dim=1)
    expected, _ = flash(q[:, :length], k[:, :length], v[:, :length], 8**-0.5)
    torch.testing.assert_close(actual[:, :length], expected, atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(actual[:, length:]) == 0


@pytest.mark.parametrize("backend,heads,kv_heads,match", [
    (AttentionBackendEnum.TORCH_SDPA, 3, 3, "FlashAttention backend"),
    (AttentionBackendEnum.FLASH_ATTN, 4, 2, "GQA"),
])
def test_ring_rejects_unsupported_attention(monkeypatch, backend, heads, kv_heads, match):
    import fastvideo.attention.ring_attention as ring
    monkeypatch.setattr(ring, "get_ring_size", lambda: 2)
    monkeypatch.setattr(ring, "get_sp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match=match):
        RingAttention(num_heads=heads, num_kv_heads=kv_heads, softmax_scale=128**-0.5,
                      causal=False, backend=backend)


def test_ring_rejects_training():
    attention = object.__new__(RingAttention)
    with pytest.raises(NotImplementedError, match="training/backward"):
        attention._validate_ring_inputs(torch.zeros(1, 4, 3, 128, dtype=torch.bfloat16), training=True,
                                        original_seq_len=7, replicated_q=None, replicated_k=None, replicated_v=None)


def test_h3_ring_dispatch_with_compile_graph_break(monkeypatch):
    """Dynamo may compile H3 projections/RoPE while Ring keeps its eager boundary."""
    import fastvideo.attention.ring_attention as ring
    import fastvideo.models.dits.minimax_h3 as h3
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    monkeypatch.setenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "1")
    monkeypatch.setattr(ring, "get_ring_size", lambda: 2)
    monkeypatch.setattr(ring, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(ring, "get_ulysses_group", lambda: None)
    monkeypatch.setattr(ring, "get_ring_group", lambda: SimpleNamespace(device_group=None))
    monkeypatch.setattr(ring, "ulysses_all_to_all_4D", lambda tensor, **kwargs: tensor)
    seen = []

    def kernel(q, k, v, **kwargs):
        assert kwargs["original_seq_len"] == 7
        seen.append(q.clone())
        return q

    def unexpected_rope(*args):
        raise AssertionError("H3 must not rotate Q/K again inside RingAttention")

    monkeypatch.setattr(ring, "ring_flash_attn_func", kernel)
    monkeypatch.setattr(RingAttention, "_slice_local_rope", unexpected_rope)
    import fastvideo.attention.layer as layer
    from fastvideo.attention.backends.flash_attn import FlashAttentionBackend
    monkeypatch.setattr(layer, "get_compute_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(h3, "get_compute_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(layer, "get_attn_backend", lambda *a, **kw: FlashAttentionBackend)
    monkeypatch.setattr(h3, "get_attn_backend", lambda *a, **kw: FlashAttentionBackend)
    attention = h3.MiniMaxH3Attention(24, 3, 128, 1e-5, (AttentionBackendEnum.FLASH_ATTN,), None, "test")
    attention = attention.bfloat16().eval()
    with torch.no_grad():
        for param in attention.parameters():
            if param.ndim == 1:
                param.fill_(1)
            else:
                param.normal_(std=0.05)
    x = torch.randn(1, 4, 24, dtype=torch.bfloat16)
    rope = h3.MiniMaxH3RotaryPosEmbed(16, 10000.0)(torch.arange(12).reshape(4, 3))
    with torch.no_grad():
        eager = attention(x, rope, 7)
        compiled = torch.compile(attention, backend="eager", fullgraph=False)(x, rope, 7)
    torch.testing.assert_close(compiled, eager)
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1])
