"""Regression: Triton block-sparse backward gradient parity at realistic activation scale.

The backward used to fold ``sm_scale / ln(2)`` into K in bf16 before the
exp2-based logit recompute. The bf16 rounding error on the pre-scaled K grows
proportionally to |logit| and exp2 amplifies it into exponentially wrong
probabilities, so dQ/dK/dV were correct at unit scale (every pre-existing test)
but off by orders of magnitude at real activation magnitudes.

This test sweeps the input scale and checks the Triton kernel's gradients
against an fp32 masked-dense SDPA reference. The unit-scale case is the
control (it passed even with the broken kernel); the large-scale cases are
the regression.
"""

import pytest
import torch

from fastvideo_kernel.block_sparse_attn import _map_to_index, block_sparse_attn_triton

from .utils import generate_block_sparse_mask_for_function

BLOCK = 64


@pytest.fixture(autouse=True)
def _seed_rng():
    """Pin the RNG so these cases do not depend on what ran before them.

    Same convention as test_vsa_varlen.py: every tensor here comes from the
    global torch RNG and the checks use tight thresholds, so an unseeded run
    would shift inputs whenever an earlier test file draws a different number
    of randoms.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def _dense_reference(q, k, v, block_mask):
    """fp32 masked-dense SDPA over the token-expanded block mask.

    q/k/v: [B, H, S, D]; block_mask: [B, H, S // BLOCK, S // BLOCK] bool.
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    token_mask = block_mask.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)
    logits = torch.matmul(qf, kf.transpose(-2, -1)) * (q.shape[-1]**-0.5)
    logits = logits.masked_fill(~token_mask, float("-inf"))
    return torch.matmul(logits.softmax(dim=-1), vf)


@pytest.mark.cuda
@pytest.mark.parametrize("scale", [1.0, 4.0, 16.0])
def test_triton_backward_grad_parity_across_input_scales(scale: float) -> None:
    """Kernel dQ/dK/dV must stay within a few percent of the fp32 reference
    regardless of input magnitude.

    With the bf16 K pre-scaling bug, scale<=4.0 passes at this geometry while
    scale=16.0 fails (measured on GB200: dq relative L2 error 5.9e-1 vs 6.9e-3
    fixed); at larger geometries and real activation magnitudes the broken
    kernel is off by orders of magnitude. The passing unit-scale case is
    exactly how the bug survived the original test suite.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, heads, dim = 1, 4, 128
    num_blocks = 8
    seq = num_blocks * BLOCK

    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * scale
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * scale
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    grad_out = torch.randn_like(q)

    block_mask = generate_block_sparse_mask_for_function(heads, num_blocks, num_blocks, k=3,
                                                         device=device).unsqueeze(0)
    q2k_idx, q2k_num = _map_to_index(block_mask)
    variable_block_sizes = torch.full((num_blocks, ), BLOCK, dtype=torch.int32, device=device)

    q_ker, k_ker, v_ker = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out_ker, _ = block_sparse_attn_triton(q_ker, k_ker, v_ker, q2k_idx, q2k_num, variable_block_sizes)
    (out_ker.float() * grad_out.float()).sum().backward()

    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out_ref = _dense_reference(q_ref, k_ref, v_ref, block_mask)
    (out_ref * grad_out.float()).sum().backward()

    # Forward is exact at any scale; this pins the harness itself.
    fwd_rel = ((out_ker.float() - out_ref).norm() / out_ref.norm()).item()
    assert fwd_rel < 2e-2, f"scale={scale}: forward rel err {fwd_rel:.3e}"

    for name, g_ker, g_ref in (
        ("dq", q_ker.grad, q_ref.grad),
        ("dk", k_ker.grad, k_ref.grad),
        ("dv", v_ker.grad, v_ref.grad),
    ):
        assert torch.isfinite(g_ker).all().item(), f"scale={scale}: non-finite {name}"
        ref_norm = g_ref.float().norm()
        rel = ((g_ker.float() - g_ref.float()).norm() / ref_norm.clamp_min(1e-12)).item()
        ratio = (g_ker.float().norm() / ref_norm.clamp_min(1e-12)).item()
        print(f"scale={scale} {name}: rel_l2={rel:.4e} norm_ratio={ratio:.4f}")
        assert rel < 5e-2, f"scale={scale}: {name} rel l2 err {rel:.3e} >= 5e-2"
        assert 0.98 < ratio < 1.02, f"scale={scale}: {name} grad-norm ratio {ratio:.4f}"
