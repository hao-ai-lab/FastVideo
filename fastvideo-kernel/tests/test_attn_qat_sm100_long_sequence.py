# SPDX-License-Identifier: Apache-2.0
"""SM100 long-sequence route and exact output/backward regression tests."""
import pytest
import torch
from fastvideo_kernel.triton_kernels import attn_qat_train as kernel


@pytest.fixture(autouse=True)
def sm100_options(monkeypatch):
    for key, value in {"SM100_OPTIMIZED": "1", "SM100_WIDE_BWD": "1", "FWD_MODE": "fast", "FWD_EXACT_M": "0"}.items():
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_" + key, value)


@pytest.mark.parametrize("qlen,kvlen,dtype,optimized,mode", [
    (2112, 2112, torch.bfloat16, True, "fast"), (16384, 16400, torch.bfloat16, True, "fast"),
    (31200, 31200, torch.float16, True, "fast"), (31200, 31200, torch.float32, True, "fast"),
    (31200, 31200, torch.bfloat16, False, "fast"), (31200, 31200, torch.bfloat16, True, "balanced"),
    (31200, 31200, torch.bfloat16, True, "reference"),
])
def test_long_sequence_route_unsupported_shapes(qlen, kvlen, dtype, optimized, mode):
    assert not kernel._sm100_long_sequence_route(qlen, kvlen, dtype, optimized, mode)


def test_long_sequence_route_rejects_exact_m(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", "1")
    assert not kernel._sm100_long_sequence_route(31200, 31200, torch.bfloat16, True, "fast")


def test_long_sequence_route_enabled_for_production_shape():
    assert kernel._sm100_long_sequence_route(31200, 31200, torch.bfloat16, True, "fast")


def test_wide_backward_launch_config():
    tuned = kernel._sm100_backward_launch_config(128, torch.bfloat16)
    legacy = kernel._sm100_backward_launch_config(64, torch.bfloat16)
    assert tuned == ((4, 2), (8, 3))
    assert legacy == ((8, 2), (8, 3))


def _legacy_route_patches(patch):
    patch.setattr(kernel, "_sm100_long_sequence_route", lambda *args, **kwargs: False)
    patch.setattr(kernel, "_select_sm100_backward_blocks", lambda *args, **kwargs: (64, 64))
    patch.setattr(kernel, "_sm100_backward_launch_config", lambda block_n, dtype: ((8, 2), (8, 3)))


requires_sm100 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
                                    reason="SM100 bitwise production/tail regression")


def _assert_bitwise_matches_legacy_route(monkeypatch, batch, heads, qlen, kvlen):
    torch.manual_seed(1000)
    inputs = tuple(torch.randn((batch, heads, length, 128), device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for length in (qlen, kvlen, kvlen))
    grad = torch.randn_like(inputs[0])

    def run(legacy: bool):
        with monkeypatch.context() as patch:
            if legacy:
                _legacy_route_patches(patch)
            out = kernel.attention(*inputs, False, 128**-0.5,
                                   True, False, True, True, False, True, True, False, False, False)
            saved = out.grad_fn.saved_tensors
            # _attention.backward reads high_prec_o then M from ctx.saved_tensors.
            high, m = saved[-2], saved[-1]
            gradients = torch.autograd.grad(out, inputs, grad)
            return out.detach(), *gradients, high.detach(), m.detach()

    reference = run(True)
    actual = run(False)
    for name, a, b in zip(("output", "dQ", "dK", "dV", "high_output", "M"), reference, actual):
        integer = torch.int32 if a.element_size() == 4 else torch.int16
        assert torch.isfinite(b).all(), name
        assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer)), name


@requires_sm100
@pytest.mark.parametrize("batch,heads,qlen,kvlen", [
    (1, 1, 16384, 16384), (1, 1, 16400, 16400), (1, 6, 31200, 31200), (1, 1, 31232, 31232),
    (1, 1, 2112, 2112), (1, 1, 2112, 2080), (1, 1, 16384, 16400),
    (2, 3, 16384, 16384), (2, 3, 16400, 16400),
    # Non-16-multiple lengths: the forward tail tile crosses a 16-column
    # quant group (tail 17 and tail 127 over the 128-wide KV tile).
    (1, 1, 16401, 16401), (1, 1, 16511, 16511),
])
def test_long_sequence_route_bitwise_output_statistics_and_gradients(monkeypatch, batch, heads, qlen, kvlen):
    _assert_bitwise_matches_legacy_route(monkeypatch, batch, heads, qlen, kvlen)


@requires_sm100
@pytest.mark.parametrize("mode,exact_m", [("balanced", "0"), ("reference", "0"), ("fast", "1"), ("balanced", "1")])
@pytest.mark.parametrize("qlen", [16384, 16400])
def test_wide_backward_bitwise_in_comparison_modes(monkeypatch, mode, exact_m, qlen):
    # These modes keep the masked forward loop but still take the 64x128 backward.
    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_MODE", mode)
    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", exact_m)
    assert not kernel._sm100_long_sequence_route(qlen, qlen, torch.bfloat16, True, mode)
    assert kernel._select_sm100_backward_blocks(qlen, qlen) == (64, 128)
    _assert_bitwise_matches_legacy_route(monkeypatch, 1, 1, qlen, qlen)
