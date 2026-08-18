# SPDX-License-Identifier: Apache-2.0
"""Counts how often a stand-in attention kernel runs under each checkpointing mode.

One call per block means the output was retained; two means it was recomputed in
backward. Both copies of the policy are exercised -- fastvideo/training/ and
fastvideo/train/utils/ -- because they forked with identical defects.

Loaded by path rather than through ``import fastvideo``: the package __init__
pulls the whole pipeline stack, which needs a CUDA driver, while the module under
test is a leaf with no fastvideo dependencies.
"""

from __future__ import annotations

import collections
import importlib.util
import pathlib

import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODULES = {
    "legacy": _ROOT / "fastvideo" / "training" / "activation_checkpoint.py",
    "modular": _ROOT / "fastvideo" / "train" / "utils" / "activation_checkpoint.py",
}

N_BLOCKS, DIM = 4, 32
CALLS: dict[str, int] = {}

# Every attention op these models can dispatch, by the name OpOverload.name()
# reports. A backend added without a matching pattern fails here rather than
# silently costing full recomputation, which is the bug this file guards.
KNOWN_SAVE_OPS = [
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "fastvideo::_flash_attn_default_forward",
    "fastvideo::_flash_attn_cute_forward",
    "fastvideo::_flash_attn_cute_varlen_forward",
    "fastvideo::_flash_attn_no_pad_forward",
    "fastvideo::video_sparse_attn",
    "fastvideo::moba_attn_varlen",
    "fastvideo::sage_attn",
    "_c10d_functional::reduce_scatter_tensor",
    "_c10d_functional::all_gather_into_tensor",
]

# aten::mm is the one op the replaced identity set named that is deliberately
# dropped: every Linear here carries a bias so F.linear emits addmm and the
# entry never fired, and retaining matmuls costs far more than it saves at
# video sequence lengths.
KNOWN_RECOMPUTE_OPS = ["aten::mm", "aten::addmm", "aten::native_layer_norm", "aten::silu"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ac_under_test_{name}", _MODULES[name])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LOADED = {name: _load(name) for name in _MODULES}


def _make_attn_op(opname: str):
    """Custom op mirroring flash_attn_cute.py: custom_op + register_autograd."""

    @torch.library.custom_op(f"actest::{opname}", mutates_args=())
    def _fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        CALLS[opname] = CALLS.get(opname, 0) + 1
        weights = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1]**0.5, dim=-1)
        return weights @ v

    @_fwd.register_fake
    def _(q, k, v):
        return torch.empty_like(q)

    def _bwd(ctx, grad):
        q, k, v = ctx.saved
        with torch.enable_grad():
            qq, kk, vv = (t.detach().requires_grad_(True) for t in (q, k, v))
            weights = torch.softmax(qq @ kk.transpose(-1, -2) / qq.shape[-1]**0.5, dim=-1)
            return torch.autograd.grad(weights @ vv, (qq, kk, vv), grad)

    torch.library.register_autograd(
        f"actest::{opname}",
        _bwd,
        setup_context=lambda ctx, inputs, output: setattr(ctx, "saved", inputs),
    )
    return getattr(torch.ops.actest, opname)


# "flash_attn_stub" carries the "flash_attn" fragment and must match;
# "mystery_kernel" carries none of them and must not.
MATCHING = _make_attn_op("flash_attn_stub")
UNMATCHED = _make_attn_op("mystery_kernel")


class Block(nn.Module):

    def __init__(self, dim: int, op) -> None:
        super().__init__()
        # bias=True, like every Linear in a Wan block -- this is what made the
        # replaced policy's aten.mm.default entry never fire.
        self.qkv, self.o, self.op = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim), op

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        return self.o(self.op(q, k, v))


class MiniDiT(nn.Module):

    def __init__(self, dim: int, num_blocks: int, op) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, op) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.fixture(params=sorted(_MODULES))
def ac(request):
    return _LOADED[request.param]


def _attention_calls(ac_module, op, opname: str, **kwargs) -> int:
    torch.manual_seed(0)
    model = ac_module.apply_activation_checkpointing(MiniDiT(DIM, N_BLOCKS, op), **kwargs)
    model(torch.randn(1, 16, DIM, requires_grad=True)).sum().backward()
    return CALLS.get(opname, 0)


def _replaced_identity_wrap(module):
    """The exact pre-fix `ops` implementation, as a controlled comparison."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.utils.checkpoint import (CheckpointPolicy, create_selective_checkpoint_contexts)

    ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    }

    def ctx_fn():
        meta: dict[str, int] = collections.defaultdict(int)

        def policy(ctx, func, *args, **kwargs):
            key = f"{'recompute' if ctx.is_recompute else 'forward'}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[key] += 1
            save = func in ops and not (func == torch.ops.aten.mm.default and meta[key] % 2 == 0)
            return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(policy)

    return checkpoint_wrapper(module, context_fn=ctx_fn, preserve_rng_state=False)


@pytest.mark.parametrize("op_name", KNOWN_SAVE_OPS)
def test_every_known_attention_and_collective_op_is_saved(ac, op_name: str) -> None:
    assert any(pattern in op_name for pattern in ac._SAVE_OP_PATTERNS)


@pytest.mark.parametrize("op_name", KNOWN_RECOMPUTE_OPS)
def test_ordinary_compute_is_recomputed(ac, op_name: str) -> None:
    assert not any(pattern in op_name for pattern in ac._SAVE_OP_PATTERNS)


def test_full_recomputes_attention(ac) -> None:
    assert _attention_calls(ac, MATCHING, "flash_attn_stub", checkpointing_type="full") == 2 * N_BLOCKS


def test_replaced_identity_op_set_missed_custom_ops() -> None:
    torch.manual_seed(0)
    model = _replaced_identity_wrap(MiniDiT(DIM, N_BLOCKS, MATCHING))
    model(torch.randn(1, 16, DIM, requires_grad=True)).sum().backward()
    assert CALLS.get("flash_attn_stub", 0) == 2 * N_BLOCKS


def test_ops_saves_attention(ac) -> None:
    assert _attention_calls(ac, MATCHING, "flash_attn_stub", checkpointing_type="ops") == N_BLOCKS


def test_unlisted_op_is_recomputed_but_still_correct(ac) -> None:
    assert _attention_calls(ac, UNMATCHED, "mystery_kernel", checkpointing_type="ops") == 2 * N_BLOCKS


@pytest.mark.parametrize(("n_layer", "expected"), [
    (1, 2 * N_BLOCKS),
    (2, int(1.5 * N_BLOCKS)),
])
def test_block_skip_n_layer_controls_coverage(ac, n_layer: int, expected: int) -> None:
    assert _attention_calls(ac, MATCHING, "flash_attn_stub", checkpointing_type="block_skip",
                            n_layer=n_layer) == expected


@pytest.mark.parametrize("mode", ["full", "ops", "block_skip"])
def test_checkpointing_does_not_change_gradients(ac, mode: str) -> None:

    def grads(checkpointing_type: str | None) -> list[torch.Tensor]:
        torch.manual_seed(0)
        model = MiniDiT(DIM, N_BLOCKS, MATCHING)
        if checkpointing_type is not None:
            model = ac.apply_activation_checkpointing(model, checkpointing_type=checkpointing_type, n_layer=2)
        torch.manual_seed(1)
        model(torch.randn(1, 16, DIM)).square().sum().backward()
        return [p.grad.detach().clone() for p in model.parameters()]

    reference = grads(None)
    for expected, actual in zip(reference, grads(mode), strict=True):
        assert torch.equal(expected, actual)
