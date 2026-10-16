# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from fastvideo_kernel.triton_kernels import attn_qat_train as kernel


def _production_route_kwargs():
    return {
        "device": torch.device("cuda"),
        "head_dim": 128,
        "causal": False,
        "is_qat": True,
        "fake_quant_p": True,
        "two_level_quant_p": False,
        "use_global_sf_p": False,
    }


def test_sm100_production_configuration_uses_optimized_route(monkeypatch):
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", raising=False)

    assert kernel._use_sm100_optimized_qat(**_production_route_kwargs())


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("head_dim", 64),
        ("causal", True),
        ("is_qat", False),
        ("fake_quant_p", False),
        ("two_level_quant_p", True),
        ("use_global_sf_p", True),
    ],
)
def test_unsupported_configuration_keeps_legacy_route(monkeypatch, override, value):
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    kwargs = _production_route_kwargs()
    kwargs[override] = value

    assert not kernel._use_sm100_optimized_qat(**kwargs)


def test_non_sm100_and_debug_switch_keep_legacy_route(monkeypatch):
    kwargs = _production_route_kwargs()
    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: False)
    assert not kernel._use_sm100_optimized_qat(**kwargs)

    monkeypatch.setattr(kernel, "is_sm100", lambda device=None: True)
    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", "0")
    assert not kernel._use_sm100_optimized_qat(**kwargs)


def test_exact_m_is_opt_in(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", raising=False)
    assert not kernel._sm100_exact_m_enabled()

    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", "1")
    assert kernel._sm100_exact_m_enabled()


def test_sm100_wide_backward_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM100_WIDE_BWD", raising=False)
    assert kernel._sm100_wide_backward_enabled()
    assert kernel._select_sm100_backward_blocks(31_200, 31_200) == (64, 128)
    assert kernel._select_sm100_backward_blocks(2_112, 2_112) == (64, 64)
    assert kernel._select_sm100_backward_blocks(31_200, 16_384) == (64, 64)

    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM100_WIDE_BWD", "0")
    assert not kernel._sm100_wide_backward_enabled()
    assert kernel._select_sm100_backward_blocks(31_200, 31_200) == (64, 64)


def test_sm120_joined_pv_is_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", raising=False)
    assert kernel._consumer_blackwell_join_qat_pv_enabled()

    monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", "0")
    assert not kernel._consumer_blackwell_join_qat_pv_enabled()


@pytest.mark.parametrize(
    ("n_ctx", "mode", "expected"),
    [
        (2_048, "fast", (32, 32, 4, 5)),
        (4_096, "fast", (128, 128, 8, 3)),
        (4_096, "balanced", (64, 32, 4, 4)),
        (4_096, "reference", (32, 32, 4, 5)),
        (31_200, "reference", (32, 32, 4, 4)),
    ],
)
def test_sm100_forward_config_selection(n_ctx, mode, expected):
    assert kernel._select_sm100_forward_config(n_ctx, n_ctx, mode) == expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 parity test",
)
@pytest.mark.parametrize(("q_length", "kv_length"), [(2_112, 2_112), (2_112, 2_080)])
def test_sm100_optimized_forward_backward_matches_legacy(monkeypatch, q_length, kv_length):
    torch.manual_seed(7)
    q_shape = (1, 1, q_length, 128)
    kv_shape = (1, 1, kv_length, 128)
    inputs = [
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
    ]
    grad_out = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    flags = (
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        True,  # warp_specialize (disabled internally on Blackwell)
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )

    def run(optimized: bool):
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM100_OPTIMIZED", "1" if optimized else "0")
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_MODE", "fast")
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_FWD_EXACT_M", "1")
        q, k, v = [tensor.clone().requires_grad_(True) for tensor in inputs]
        output = kernel.attention(
            q,
            k,
            v,
            False,
            1.0 / math.sqrt(q_shape[-1]),
            *flags,
        )
        output.backward(grad_out)
        return output.detach(), q.grad, k.grad, v.grad

    legacy = run(False)
    optimized = run(True)

    assert (optimized[0].float() - legacy[0].float()).abs().max().item() <= 1e-2
    assert (optimized[1].float() - legacy[1].float()).abs().max().item() <= 4e-3
    assert (optimized[2].float() - legacy[2].float()).abs().max().item() <= 4e-3
    assert torch.equal(optimized[3], legacy[3])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 joined P@V parity test; GB10 uses the split path",
)
@pytest.mark.parametrize(("q_length", "kv_length"), [(2_112, 2_112), (2_112, 2_080)])
def test_sm120_joined_pv_forward_backward_matches_split_path(monkeypatch, q_length, kv_length):
    torch.manual_seed(11)
    q_shape = (1, 1, q_length, 128)
    kv_shape = (1, 1, kv_length, 128)
    inputs = [
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16),
    ]
    grad_out = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    flags = (
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        True,  # warp_specialize (disabled internally on Blackwell)
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )

    def run(joined_pv: bool):
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", "1" if joined_pv else "0")
        q, k, v = [tensor.clone().requires_grad_(True) for tensor in inputs]
        output = kernel.attention(
            q,
            k,
            v,
            False,
            1.0 / math.sqrt(q_shape[-1]),
            *flags,
        )
        output.backward(grad_out)
        return output.detach(), q.grad, k.grad, v.grad

    split = run(False)
    joined = run(True)

    assert torch.equal(joined[0], split[0])
    assert torch.equal(joined[1], split[1])
    assert torch.equal(joined[2], split[2])
    assert torch.equal(joined[3], split[3])


def _set_join_switch(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", raising=False)
    else:
        monkeypatch.setenv("FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV", value)


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (12, 2), (10, 0), (9, 0)])
@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_joined_pv_requires_sm120_on_the_input_device(monkeypatch, capability, switch):
    device = torch.device("cuda", 1)

    def get_capability(selected_device):
        assert selected_device == device
        return capability

    monkeypatch.setattr(kernel, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    _set_join_switch(monkeypatch, switch)
    assert kernel._use_consumer_blackwell_joined_qat_pv(device) == (capability == (12, 0) and switch != "0")


def test_joined_pv_does_not_probe_cuda_for_another_backend(monkeypatch):
    monkeypatch.setattr(kernel, "is_cuda", lambda: False)

    def unexpected_probe(device):
        pytest.fail("Non-CUDA routing must not probe a CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_probe)
    assert not kernel._use_consumer_blackwell_joined_qat_pv(torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1),
    reason="GB10 split-path parity test requires SM121",
)
@pytest.mark.parametrize(
    ("q_length", "kv_length", "heads", "seed"),
    [(2_112, 2_112, 1, 11), (2_112, 2_080, 1, 11), (31_200, 31_200, 3, 0)],
)
def test_sm121_forward_backward_keeps_split_path(monkeypatch, q_length, kv_length, heads, seed):
    torch.manual_seed(seed)
    q_shape = (1, heads, q_length, 128)
    kv_shape = (1, heads, kv_length, 128)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in (q_shape, kv_shape, kv_shape)
    ]
    grad_out = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    flags = tuple(
        dict(
            use_qat_qkv_backward=True,
            smooth_k=False,
            warp_specialize=True,
            IS_QAT=True,
            two_level_quant_P=False,
            fake_quant_P=True,
            use_high_prec_o=True,
            smooth_q=False,
            use_global_sf_P=False,
            use_global_sf_QKV=False,
        ).values())
    launched_joined_pv = []
    original_forward = kernel._attn_fwd

    class ForwardRecorder:

        def __getitem__(self, grid):
            launch = original_forward[grid]

            def record_launch(*args, **kwargs):
                launched_joined_pv.append(kwargs["JOIN_QAT_PV"])
                return launch(*args, **kwargs)

            return record_launch

    monkeypatch.setattr(kernel, "_attn_fwd", ForwardRecorder())

    def run(switch):
        _set_join_switch(monkeypatch, switch)
        q, k, v = [tensor.clone().requires_grad_(True) for tensor in inputs]
        output = kernel.attention(q, k, v, False, 1.0 / math.sqrt(q_shape[-1]), *flags)
        ste_output, stats = output.grad_fn.saved_tensors[3:]
        grads = torch.autograd.grad(output, (q, k, v), grad_out)
        return output.detach(), *grads, ste_output, stats

    split = run("0")
    for switch in (None, "1"):
        actual = run(switch)
        for name, candidate, reference in zip(("output", "dQ", "dK", "dV", "STE output", "M"), actual, split):
            assert torch.isfinite(candidate).all(), name
            assert torch.equal(candidate, reference), name

    assert launched_joined_pv == [False] * 3, launched_joined_pv
