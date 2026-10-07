# SPDX-License-Identifier: Apache-2.0
"""FA2 policy integration on CUDA; skip when CUDA or native FA2 is absent."""
from contextlib import contextmanager

import pytest
import torch

@contextmanager
def mode(enabled):
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(enabled, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@pytest.fixture
def modules():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from fastvideo.attention.utils import flash_attn_default as default
    from fastvideo.attention.utils import flash_attn_no_pad as masked
    import flash_attn.flash_attn_interface as native
    if default.fa_version != "2" or masked._FA_VARLEN_VERSION != "2":
        pytest.skip("This hardware acceptance covers FA2 only")
    return default, masked, native


@contextmanager
def observe(modules):
    records, handles = [], []
    for module, name in [(modules[0], "_fa2_backward"), (modules[1], "_fa2_varlen_backward"),
                         (modules[2], "_wrapped_flash_attn_varlen_backward")]:
        original = getattr(module, name)
        def wrapped(*args, _original=original, **kwargs):
            value = kwargs.get("deterministic", args[20] if len(args) > 20 else None)
            records.append(value)
            return _original(*args, **kwargs)
        setattr(module, name, wrapped)
        handles.append((module, name, original))
    try:
        yield records
    finally:
        for module, name, original in reversed(handles):
            setattr(module, name, original)


def make_call(modules, route, length=257):
    default, masked, _ = modules
    torch.manual_seed(1000)
    shape = (1, length, 3, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    sk = length if route.startswith("qkv") else length - 16
    k = torch.randn((1, sk, 3, 128), dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    if route == "default":
        return default.flash_attn_func_compilable, (q, k, v)
    if route.startswith("qkv"):
        qkv = torch.stack([q, k, v], dim=2)
        mask = torch.arange(length, device=q.device)[None] < length - 13
        fn = masked.flash_attn_no_pad if route.endswith("direct") else masked.flash_attn_no_pad_compilable
        return lambda x: fn(x, mask), (qkv,)
    q_mask = torch.arange(length, device=q.device)[None] < length - 7
    k_mask = torch.arange(sk, device=q.device)[None] < sk - 11
    fn = masked.flash_attn_varlen_qk_no_pad if route.endswith("direct") else masked.flash_attn_varlen_qk_no_pad_compilable
    return lambda q, k, v: fn(q, k, v, q_mask, k_mask), (q, k, v)


def run(call, inputs, forward_mode, backward_mode):
    values = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    with mode(forward_mode):
        out = call(*values)
    with mode(backward_mode):
        gradients = torch.autograd.grad(out, values, torch.ones_like(out))
    torch.cuda.synchronize()
    assert all(torch.isfinite(t).all() for t in (out, *gradients))
    return out.detach(), tuple(g.detach() for g in gradients)


def same_bytes(left, right):
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("route", ["default", "qkv", "varlen", "qkv-direct", "varlen-direct"])
@pytest.mark.parametrize("length", [64, 257])
def test_strict_repeated_backward_ten_times(modules, route, length):
    call, inputs = make_call(modules, route, length)
    first = None
    with observe(modules) as records:
        for _ in range(10):
            result = run(call, inputs, True, True)
            if first is None:
                first = result
            else:
                same_bytes(first[0], result[0])
                for a, b in zip(first[1], result[1], strict=True):
                    same_bytes(a, b)
    assert records == [True] * 10


@pytest.mark.parametrize("route", ["default", "qkv", "varlen", "qkv-direct", "varlen-direct"])
@pytest.mark.parametrize("forward_mode,backward_mode", [(False, False), (False, True), (True, False)])
def test_backward_policy_changes(modules, route, forward_mode, backward_mode):
    call, inputs = make_call(modules, route)
    with observe(modules) as records:
        run(call, inputs, forward_mode, backward_mode)
    assert records == [forward_mode or backward_mode]


@pytest.mark.parametrize("route", ["default", "qkv", "varlen"])
def test_compile_cache_and_backward_runtime_policy(modules, route):
    call, inputs = make_call(modules, route, 64)
    compiled = torch.compile(call, fullgraph=True)
    with observe(modules) as records:
        for forward_mode, backward_mode in [(False, False), (True, True), (True, False), (False, False), (True, True)]:
            records.clear()
            result = run(compiled, inputs, forward_mode, backward_mode)
            assert records and all(value is (forward_mode or backward_mode) for value in records)
            if forward_mode and backward_mode:
                expected = run(call, inputs, True, True)
                same_bytes(result[0], expected[0])
                for a, b in zip(result[1], expected[1], strict=True):
                    same_bytes(a, b)


@pytest.mark.parametrize("route", ["default", "qkv", "varlen"])
def test_compile_rejects_enabling_strict_after_forward(modules, route):
    call, inputs = make_call(modules, route, 64)
    compiled = torch.compile(call, fullgraph=True)
    values = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    with observe(modules) as records:
        with mode(False):
            output = compiled(*values)
        with mode(True), pytest.raises(RuntimeError, match="previously generated during the forward"):
            torch.autograd.grad(output, values, torch.ones_like(output))
    assert records == []  # The framework rejects the switch before any FA2 backward.
    assert all(x.grad is None for x in values)


@pytest.mark.parametrize("route", ["qkv-direct", "varlen-direct"])
def test_direct_dropout_keeps_native_saved_rng(modules, route):
    default, masked, native = modules
    from flash_attn.bert_padding import pad_input, unpad_input
    torch.manual_seed(42)
    q = torch.randn((1, 64, 3, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.ones((1, 64), dtype=torch.bool, device="cuda")
    if route == "qkv-direct":
        data = torch.stack((q, k, v), dim=2)
        candidate = lambda x: masked.flash_attn_no_pad(x, mask, dropout_p=0.1)
        def reference(x):
            packed, indices, cu, maximum, _ = unpad_input(x, mask)
            out = native.flash_attn_varlen_qkvpacked_func(packed, cu, maximum, dropout_p=0.1, deterministic=True)
            return pad_input(out, indices, 1, 64)
        inputs = (data,)
    else:
        candidate = lambda q, k, v: masked.flash_attn_varlen_qk_no_pad(q, k, v, mask, mask, dropout_p=0.1)
        def reference(q, k, v):
            a, indices, cu, maximum, _ = unpad_input(q, mask)
            b, _, _, _, _ = unpad_input(k, mask)
            c, _, _, _, _ = unpad_input(v, mask)
            out = native.flash_attn_varlen_func(a, b, c, cu, cu, maximum, maximum, dropout_p=0.1, deterministic=True)
            return pad_input(out, indices, 1, 64)
        inputs = (q, k, v)
    rng = torch.cuda.get_rng_state()
    actual = run(candidate, inputs, True, True)
    torch.cuda.set_rng_state(rng)
    expected = run(reference, inputs, True, True)
    same_bytes(actual[0], expected[0])
    for a, b in zip(actual[1], expected[1], strict=True):
        same_bytes(a, b)


@pytest.mark.parametrize("route", ["default", "qkv", "varlen"])
def test_strided_inputs_have_correct_fake_gradient_layout(modules, route):
    call, inputs = make_call(modules, route, 64)
    # Preserve values while changing the physical order of sequence/head axes.
    strided = tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in inputs)
    with mode(True):
        eager = run(call, strided, True, True)
        compiled = run(torch.compile(call, fullgraph=True), strided, True, True)
    same_bytes(eager[0], compiled[0])
    for a, b in zip(eager[1], compiled[1], strict=True):
        same_bytes(a, b)
