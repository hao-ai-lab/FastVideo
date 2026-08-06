# SPDX-License-Identifier: Apache-2.0
"""
Parity tests for the sm_100a CUDA forward of block-causal-sink attention.

The CUDA kernel replaces only the forward; the Triton backward is reused, so it consumes the
lse the CUDA forward writes. A wrong lse is silently wrong gradients rather than a crash, which
is why these compare backward as well as forward.

Run with: python -m pytest tests/test_block_causal_sink_sm100a.py -v
"""

import pytest
import torch

from fastvideo.attention.kernels import block_causal_sink as bcs
from fastvideo.attention.kernels import block_causal_sink_cuda as bcs_cuda
from fastvideo.models.dits._causal_train_attention import CausalTrainAttentionPlan

HEAD_DIM = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0)
    or not bcs_cuda._HAS_CUDA_BCS,
    reason="requires Blackwell (sm_100a) and a built fastvideo_kernel extension",
)


def make_delta(num_frames, num_frame_per_block):
    num_blocks = num_frames // num_frame_per_block
    inv_freq = 1.0 / (10000.0**(torch.arange(0, HEAD_DIM, 2, device="cuda").float() / HEAD_DIM))
    angle = torch.arange(num_blocks, device="cuda").float()[:, None] * inv_freq[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().float().contiguous(), angle.sin().float().contiguous()


def make_plan(num_frames=6, frame_seqlen=1456, num_frame_per_block=3, sink_size=1,
              local_attn_size=6, rope_delta=False):
    delta_cos, delta_sin = (make_delta(num_frames, num_frame_per_block) if rope_delta else
                            (None, None))
    return CausalTrainAttentionPlan(
        kind="blockwise",
        impl="triton",
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=num_frame_per_block,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        sm_scale=1.0 / (HEAD_DIM**0.5),
        delta_cos=delta_cos,
        delta_sin=delta_sin,
    )


def make_qkv(plan, num_heads, seed=0):
    torch.manual_seed(seed)
    length = plan.num_frames * plan.frame_seqlen
    return [
        torch.randn(1, num_heads, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True) for _ in range(3)
    ]


def run(plan, q, k, v, grad_out, use_cuda):
    for t in (q, k, v):
        t.grad = None
    saved, bcs_cuda._HAS_CUDA_BCS = bcs_cuda._HAS_CUDA_BCS, use_cuda
    try:
        out = bcs.block_causal_sink_attention(q, k, v, plan)
        out.backward(grad_out)
    finally:
        bcs_cuda._HAS_CUDA_BCS = saved
    return out.detach(), q.grad.clone(), k.grad.clone(), v.grad.clone()


def assert_close(cuda_result, triton_result, atol=0.02):
    for name, a, b in zip(("out", "dq", "dk", "dv"), cuda_result, triton_result):
        diff = (a.float() - b.float()).abs().max().item()
        assert diff < atol, f"{name}: max |diff| = {diff:.6f} exceeds {atol}"


@pytest.mark.parametrize("num_heads", [8, 12])
def test_forward_and_backward_match_triton(num_heads):
    plan = make_plan()
    q, k, v = make_qkv(plan, num_heads)
    torch.manual_seed(1)
    grad_out = torch.randn_like(q, dtype=torch.bfloat16)

    assert bcs_cuda.is_supported(plan, q)
    assert_close(run(plan, q, k, v, grad_out, use_cuda=True),
                 run(plan, q, k, v, grad_out, use_cuda=False))


@pytest.mark.parametrize("num_frames", [3, 12])
def test_matches_triton_across_sequence_lengths(num_frames):
    plan = make_plan(num_frames=num_frames)
    q, k, v = make_qkv(plan, 8)
    torch.manual_seed(1)
    grad_out = torch.randn_like(q, dtype=torch.bfloat16)

    assert_close(run(plan, q, k, v, grad_out, use_cuda=True),
                 run(plan, q, k, v, grad_out, use_cuda=False))


@pytest.mark.parametrize("num_frames", [6, 21])
def test_rope_delta_matches_triton(num_frames):
    plan = make_plan(num_frames=num_frames, rope_delta=True)
    q, k, v = make_qkv(plan, 8)
    torch.manual_seed(1)
    grad_out = torch.randn_like(q, dtype=torch.bfloat16)

    assert bcs_cuda.is_supported(plan, q)
    assert_close(run(plan, q, k, v, grad_out, use_cuda=True),
                 run(plan, q, k, v, grad_out, use_cuda=False))


def test_rope_delta_changes_the_result():
    q, k, v = make_qkv(make_plan(), 8)
    out_plain = bcs_cuda.block_causal_sink_forward_cuda(q, k, v, None, make_plan())[0]

    plan = make_plan(rope_delta=True)
    q_sink = bcs._rotated_sink_queries(q, plan.delta_cos, plan.delta_sin,
                                       bcs._plan_scalars(plan))
    out_rope = bcs_cuda.block_causal_sink_forward_cuda(q, k, v, q_sink, plan)[0]

    assert (out_plain.float() - out_rope.float()).abs().max().item() > 1e-3


def test_non_contiguous_input_is_rejected():
    plan = make_plan()
    length = plan.num_frames * plan.frame_seqlen
    torch.manual_seed(0)
    base = torch.randn(1, length, 8, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q = base.transpose(1, 2)
    assert not q.is_contiguous() and q.stride(-1) == 1
    assert not bcs_cuda.is_supported(plan, q)


def test_lse_matches_triton():
    plan = make_plan()
    q, k, v = make_qkv(plan, 8)
    _, lse_cuda = bcs_cuda.block_causal_sink_forward_cuda(q, k, v, None, plan)

    captured = {}
    original = bcs._fwd_kernel

    class _Spy:

        def __getitem__(self, grid):
            launcher = original[grid]

            def run_kernel(*args, **kwargs):
                for arg in args:
                    if (isinstance(arg, torch.Tensor) and arg.dtype == torch.float32
                            and arg.shape == lse_cuda.shape):
                        captured["lse"] = arg
                return launcher(*args, **kwargs)

            return run_kernel

    bcs._fwd_kernel = _Spy()
    saved, bcs_cuda._HAS_CUDA_BCS = bcs_cuda._HAS_CUDA_BCS, False
    try:
        bcs.block_causal_sink_attention(q, k, v, plan)
    finally:
        bcs._fwd_kernel = original
        bcs_cuda._HAS_CUDA_BCS = saved

    assert "lse" in captured
    assert (lse_cuda - captured["lse"]).abs().max().item() < 0.02


def test_unsupported_plan_falls_back():
    plan = make_plan()
    q, k, v = make_qkv(plan, 8)

    assert not bcs_cuda.is_supported(make_plan(num_frames=7), q)
    assert not bcs_cuda.is_supported(
        CausalTrainAttentionPlan(kind="teacher_forcing", impl="triton", num_frames=6,
                                 frame_seqlen=1456, num_frame_per_block=3, local_attn_size=6,
                                 sink_size=1, sm_scale=1.0 / (HEAD_DIM**0.5)), q)
    assert not bcs_cuda.is_supported(plan, q.float())
