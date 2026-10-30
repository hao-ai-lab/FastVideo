# SPDX-License-Identifier: Apache-2.0
"""A real H3 transformer block runs the MotionKernel tier and matches the Sol tier.

Small random-initialised block, SDPA backend, no checkpoint: this exercises the
``mk_tier`` dispatch of every fused call site (including the stacked QKV
prologue) on the actual module wiring rather than on standalone kernels.
"""
import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.minimax_h3 import MiniMaxH3TransformerBlock
from fastvideo.models.dits.minimax_h3_fusions import mk
from fastvideo.platforms import AttentionBackendEnum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not mk.HAVE_TRITON,
                                reason="MotionKernel H3 kernels need CUDA + Triton")

HIDDEN, HEADS, HEAD_DIM, ROTARY_DIM, FFN, TABLE_ROWS, TOKENS = 512, 4, 128, 96, 1024, 6, 256


def _block(mk_tier: bool, fused: bool = True) -> MiniMaxH3TransformerBlock:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    try:
        return MiniMaxH3TransformerBlock(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            ffn_dim=FFN,
            time_embed_dim=128,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA, ),
            quant_config=None,
            prefix="test",
            fuse_modulate=fused,
            fuse_qknorm_rope=fused,
            fuse_swiglu=fused,
            mk_tier=mk_tier,
        ).cuda()
    finally:
        torch.set_default_dtype(torch.float32)


def _inputs():
    torch.manual_seed(1)
    return (torch.randn(1, TOKENS, HIDDEN, device="cuda",
                        dtype=torch.bfloat16), torch.randn(TABLE_ROWS, 128, device="cuda", dtype=torch.bfloat16),
            torch.randint(0, TABLE_ROWS, (TOKENS, ),
                          device="cuda", dtype=torch.int64), (torch.randn(TOKENS,
                                                                          ROTARY_DIM,
                                                                          device="cuda",
                                                                          dtype=torch.bfloat16),
                                                              torch.randn(TOKENS,
                                                                          ROTARY_DIM,
                                                                          device="cuda",
                                                                          dtype=torch.bfloat16)), TOKENS)


def test_mk_tier_dispatches_and_matches_sol(monkeypatch, distributed_setup):
    calls = {
        name: 0
        for name in ("qkv_prologue", "rmsnorm_modulate", "residual_gate_rmsnorm_modulate", "swiglu", "gated_residual")
    }
    for name in calls:
        target = getattr(mk, name)

        def counted(*args, _target=target, _name=name, **kwargs):
            calls[_name] += 1
            return _target(*args, **kwargs)

        monkeypatch.setattr(mk, name, counted)

    sol, mk_block, eager = _block(False), _block(True), _block(False, fused=False)
    mk_block.load_state_dict(sol.state_dict())
    eager.load_state_dict(sol.state_dict())
    # The loader zero-initialises ReplicatedLinear (weights arrive from the
    # checkpoint) and that would make the block an identity, hiding every kernel
    # difference. Give both copies the same non-trivial weights instead.
    torch.manual_seed(2)
    for block in (sol, mk_block, eager):
        for name, param in block.named_parameters():
            if name.endswith(".weight") and param.dim() == 2 or name.endswith(".bias"):
                param.data.normal_(0.0, 0.02)
            elif name.endswith(".weight"):
                param.data.fill_(1.0)
    inputs = _inputs()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        expected = sol(*inputs)
        actual = mk_block(*inputs)
        unfused = eager(*inputs)

    assert all(count > 0 for count in calls.values()), calls
    base = torch.linalg.vector_norm(expected.float())

    def rel(other):
        return (torch.linalg.vector_norm(other.float() - expected.float()) / base).item()

    floor, rel = rel(unfused), rel(actual)
    print(f"eager-vs-sol block rel-L2 = {floor:.3e}; mk-vs-sol block rel-L2 = {rel:.3e}")
    assert rel <= 3 * floor, (rel, floor)


def test_mk_tier_defers_to_sol_while_compiling(monkeypatch, distributed_setup):
    """A traced block must keep using the compile-safe Sol custom ops."""
    calls = []
    for name in ("qkv_prologue", "rmsnorm_modulate", "residual_gate_rmsnorm_modulate", "swiglu", "gated_residual"):
        target = getattr(mk, name)

        def counted(*args, _target=target, _name=name, **kwargs):
            calls.append(_name)
            return _target(*args, **kwargs)

        monkeypatch.setattr(mk, name, counted)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    block = _block(True)
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        block(*_inputs())
    assert calls == []
