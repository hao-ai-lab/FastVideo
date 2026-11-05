# SPDX-License-Identifier: Apache-2.0
"""MotionKernel H3 kernels match the in-tree Sol/eager paths on the production shapes.

GPU-only (Triton on CUDA); the shapes use H3's real hidden width, head layout and
AdaLN table stride, only the token count is reduced to keep the test quick.
"""
import pytest
import torch

from fastvideo.models.dits.minimax_h3_fusions import (fused_qknorm_rope, fused_residual_gate_rmsnorm_modulate,
                                                      fused_rmsnorm_modulate, minimax_h3_swiglu)
from fastvideo.models.dits.minimax_h3_fusions import mk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not mk.HAVE_TRITON,
                                reason="MotionKernel H3 kernels need CUDA + Triton")

HIDDEN, HEADS, HEAD_DIM, ROTARY_DIM, FFN_DIM, EPS = 5376, 56, 128, 96, 14336, 1e-5
TOKENS = 512


def _rel_l2(actual, expected):
    return (torch.linalg.vector_norm(actual.float() - expected.float()) /
            torch.linalg.vector_norm(expected.float())).item()


@pytest.fixture(scope="module")
def production_inputs():
    torch.manual_seed(0)
    device = "cuda"
    mod = torch.randn(6, 6, HIDDEN, device=device, dtype=torch.bfloat16)
    # The production AdaLN tables are row-strided views of this [rows, 6*D] buffer.
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (mod[:, i, :] for i in range(6))
    return {
        "x": torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
        "branch": torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
        "weight": torch.randn(HIDDEN, device=device, dtype=torch.bfloat16),
        "qk_weight": torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16),
        "scale_msa": scale_msa,
        "shift_msa": shift_msa,
        "gate_msa": gate_msa,
        "scale_mlp": scale_mlp,
        "shift_mlp": shift_mlp,
        "gate_mlp": gate_mlp,
        "index": torch.randint(0, 6, (TOKENS, ), device=device, dtype=torch.int64),
    }


def test_rmsnorm_modulate_matches_sol(production_inputs):
    i = production_inputs
    expected = fused_rmsnorm_modulate(i["x"], i["weight"], i["scale_msa"], i["shift_msa"], i["index"], EPS)
    actual = mk.rmsnorm_modulate(i["x"], i["weight"], i["scale_msa"], i["shift_msa"], i["index"])
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert _rel_l2(actual, expected) < 5e-3


def test_residual_gate_rmsnorm_modulate_matches_sol(production_inputs):
    i = production_inputs
    args = (i["branch"], i["gate_msa"], i["weight"], i["scale_mlp"], i["shift_mlp"], i["index"])
    exp_hidden, exp_normed = fused_residual_gate_rmsnorm_modulate(i["x"], *args, EPS)
    act_hidden, act_normed = mk.residual_gate_rmsnorm_modulate(i["x"], *args)
    assert _rel_l2(act_hidden, exp_hidden) < 1e-3
    assert _rel_l2(act_normed, exp_normed) < 5e-3


def test_swiglu_matches_sol(production_inputs):
    packed = torch.randn(TOKENS, 2 * FFN_DIM, device="cuda", dtype=torch.bfloat16)
    assert _rel_l2(mk.swiglu(packed), minimax_h3_swiglu(packed)) < 1e-3


def test_gated_residual_matches_eager(production_inputs):
    i = production_inputs
    expected = i["x"] + i["gate_mlp"].index_select(0, i["index"]) * i["branch"]
    assert _rel_l2(mk.gated_residual(i["x"], i["gate_mlp"], i["branch"], i["index"]), expected) < 5e-3


def test_qkv_prologue_matches_two_sol_launches(production_inputs):
    i = production_inputs
    q = torch.randn(1, TOKENS, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cos = torch.randn(TOKENS, ROTARY_DIM, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(TOKENS, ROTARY_DIM, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat(
        [fused_qknorm_rope(q, i["qk_weight"], cos, sin, EPS),
         fused_qknorm_rope(k, i["qk_weight"], cos, sin, EPS), v],
        dim=0)
    actual = mk.qkv_prologue(q, k, v, i["qk_weight"], i["qk_weight"], cos, sin)
    assert actual.shape == (3, TOKENS, HEADS, HEAD_DIM)
    assert _rel_l2(actual, expected) < 1e-4


def test_kernels_reject_cpu_tensors():
    with pytest.raises(ValueError):
        mk.swiglu(torch.randn(4, 8))
