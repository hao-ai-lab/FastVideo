"""GB10-only identity-STE dV contract tests; QAT forward quantization stays enabled."""
from contextlib import contextmanager

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from fastvideo import envs
from fastvideo_kernel.triton_kernels import attn_qat_train as kernel
from .qat_forward_reference import forward_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1),
    reason="Requires GB10 SM121",
)

# Uniform attention quantizes P = 1 as E4M3(1/6) * E2M1(1 / E4M3(1/6)) = 0.171875 * 6.
UNIFORM_QUANTIZED_P = 0.171875 * 6
TENSOR_NAMES = ("output", "dQ", "dK", "STE output", "M", "dV")


@contextmanager
def full_precision_matmul():
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_bf16


def qat(q, k, v):
    """The validated GB10 training configuration (same flags as the ATTN_QAT_TRAIN backend)."""
    return kernel.attention(
        q,
        k,
        v,
        False,  # causal
        128**-0.5,  # sm_scale
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        False,  # warp_specialize
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )


def forward_and_grads(source, do):
    q, k, v = [x.clone().requires_grad_(True) for x in source]
    out = qat(q, k, v)
    ste, m = out.grad_fn.saved_tensors[3:5]
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
    return out.detach(), dq, dk, ste, m, dv


def assert_bitwise_equal(expected, actual, names=TENSOR_NAMES):
    for name, old, new in zip(names, expected, actual):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8)), name


@pytest.mark.parametrize("nq,nk", [(64, 64), (128, 128), (170, 170), (171, 171), (2048, 2048),
                                       (31200, 31200), (257, 241), (2113, 2081)])
def test_uniform_dv_uses_forward_quantized_probability(nq, nk):
    q = torch.zeros((1, 1, nq, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros((1, 1, nk, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.ones_like(k, requires_grad=True)
    out = qat(q, k, v)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
    expected = torch.full_like(dv, UNIFORM_QUANTIZED_P * nq / nk)
    assert torch.equal(dv, expected), (nq, nk, dv.float().mean().item())
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(dk) == 0
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("nq,nk,seed", [(128, 128, 0), (128, 128, 11), (128, 128, 42),
                                           (257, 257, 0), (257, 241, 11), (257, 241, 42),
                                           (2113, 2081, 0)])
def test_dv_matches_independent_forward_weight_ste(nq, nk, seed):
    torch.manual_seed(seed)
    q = torch.randn((1, 3, nq, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1, 3, nk, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    do = torch.randn_like(q)
    out = qat(q, k, v)
    actual_dv = torch.autograd.grad(out, (q, k, v), do)[2]
    with full_precision_matmul():
        ref_out, ref_dv, _, _, _, _ = forward_reference(q.detach(), k.detach(), v.detach(), do)
    difference = actual_dv.float() - ref_dv.float()
    relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref_dv.float())
    assert relative < 1e-3, relative.item()
    cosine = torch.nn.functional.cosine_similarity(actual_dv.float().flatten(), ref_dv.float().flatten(), dim=0)
    assert cosine > 0.99999, cosine.item()
    output_relative = torch.linalg.vector_norm((out - ref_out).float()) / torch.linalg.vector_norm(ref_out.float())
    assert output_relative < 1e-3, output_relative.item()


@pytest.mark.parametrize("mode", ["save", "recompute"])
@pytest.mark.parametrize("nq,nk,heads", [(128, 128, 3), (257, 241, 3), (2113, 2081, 3), (31200, 31200, 3)])
def test_corrected_dv_preserves_other_forward_and_gradient_tensors(mode, nq, nk, heads):
    torch.manual_seed(11)
    source = [torch.randn((1, heads, n, 128), device="cuda", dtype=torch.bfloat16) for n in (nq, nk, nk)]
    do = torch.randn_like(source[0])
    with envs.override_external("FASTVIDEO_ATTN_QAT_SM121_FWD_DV", "0"):
        legacy = forward_and_grads(source, do)
    with envs.override_external("FASTVIDEO_ATTN_QAT_SM121_DV_STATS", mode):
        actual = forward_and_grads(source, do)
        repeated = forward_and_grads(source, do)
    assert_bitwise_equal(legacy[:5], actual[:5])  # everything but dV is unchanged
    for name, tensor in zip(TENSOR_NAMES, actual):
        assert torch.isfinite(tensor).all(), name
    assert_bitwise_equal(actual[-1:], repeated[-1:], names=("dV",))  # deterministic


@pytest.mark.parametrize("mode", ["save", "recompute"])
@pytest.mark.parametrize("nq,nk", [(2048, 2048), (257, 241)])
def test_dv_supports_non_reentrant_activation_checkpoint(mode, nq, nk):
    torch.manual_seed(11)
    source = [torch.randn((1, 3, length, 128), device="cuda", dtype=torch.bfloat16)
              for length in (nq, nk, nk)]
    upstream = torch.randn_like(source[0])

    def run(use_checkpoint):
        leaves = [x.clone().requires_grad_(True) for x in source]
        out = checkpoint(qat, *leaves, use_reentrant=False) if use_checkpoint else qat(*leaves)
        grads = torch.autograd.grad(out, leaves, upstream)
        return out.detach(), *grads

    with envs.override_external("FASTVIDEO_ATTN_QAT_SM121_DV_STATS", mode):
        plain = run(False)
        checkpointed = run(True)
    for name, tensor in zip(("output", "dQ", "dK", "dV"), checkpointed):
        assert torch.isfinite(tensor).all(), name
    assert_bitwise_equal(plain, checkpointed, names=("output", "dQ", "dK", "dV"))
