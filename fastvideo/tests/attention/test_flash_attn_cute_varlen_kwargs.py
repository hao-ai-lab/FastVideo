# SPDX-License-Identifier: Apache-2.0
"""FA4 varlen must not forward ``max_seqlen_q/k`` to ``_flash_attn_fwd``.

``FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN=1`` routes H3 inference through the
varlen custom op. Current ``flash_attn.cute`` builds derive the maximum
sequence lengths from ``cu_seqlens`` and reject ``max_seqlen_q``/``max_seqlen_k``
as keyword arguments, so every packed-varlen forward crashed with
``TypeError: _flash_attn_fwd() got an unexpected keyword argument
'max_seqlen_q'``.

The test installs a fake ``flash_attn.cute.interface`` whose ``_flash_attn_fwd``
has the current (max_seqlen-free) signature, then calls the varlen forward
directly. The old code fails with that TypeError; the fixed code forwards only
the arguments the kernel accepts. It runs in a subprocess so the fake modules
and the imported kernel references cannot leak into other tests.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

_SKIP_EXIT_CODE = 77

_SUBPROCESS_SCRIPT = r'''
import importlib
import importlib.machinery
import json
import sys
import types

import torch

_SKIP_EXIT_CODE = 77

import fastvideo  # noqa: F401  Import diffusers and the rest of the package before faking flash_attn.

calls = []


def fake_flash_attn_fwd(
    q,
    k,
    v,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    softmax_scale=None,
    causal=False,
    window_size_left=None,
    window_size_right=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    return_lse=True,
    deterministic=False,
    sm_margin=0,
):
    calls.append({
        "cu_seqlens_q_is_input": cu_seqlens_q is cu_seqlens,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "return_lse": return_lse,
    })
    return torch.empty_like(q), torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32)


def fake_flash_attn_bwd(*args, **kwargs):
    raise NotImplementedError("the test fake only covers the forward")


flash_attn = types.ModuleType("flash_attn")
cute = types.ModuleType("flash_attn.cute")
interface = types.ModuleType("flash_attn.cute.interface")
flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
cute.__spec__ = importlib.machinery.ModuleSpec("flash_attn.cute", None)
interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.cute.interface", None)
interface._flash_attn_fwd = fake_flash_attn_fwd
interface._flash_attn_bwd = fake_flash_attn_bwd
cute.interface = interface
flash_attn.cute = cute
sys.modules["flash_attn"] = flash_attn
sys.modules["flash_attn.cute"] = cute
sys.modules["flash_attn.cute.interface"] = interface

try:
    module = importlib.import_module("fastvideo.attention.utils.flash_attn_cute")
except ImportError as exc:
    print(exc, file=sys.stderr)
    sys.exit(_SKIP_EXIT_CODE)
module._flash_attn_fwd = fake_flash_attn_fwd


q = torch.randn(7, 2, 8, device="cuda")
cu_seqlens = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
with torch.no_grad():
    out, lse = torch.ops.fastvideo._flash_attn_cute_varlen_forward(
        q,
        q,
        q,
        cu_seqlens,
        cu_seqlens,
        7,
        7,
        0.125,
        False,
        False,
    )

with open(sys.argv[1], "w") as handle:
    json.dump({"out_shape": list(out.shape), "lse_shape": list(lse.shape), "calls": calls}, handle)
'''


def test_varlen_forward_omits_max_seqlen_kwargs(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("flash_attn.cute is only imported on CUDA builds")

    result_path = tmp_path / "varlen_forward.json"
    completed = subprocess.run([sys.executable, "-c", _SUBPROCESS_SCRIPT, str(result_path)],
                               capture_output=True,
                               text=True,
                               check=False)
    if completed.returncode == _SKIP_EXIT_CODE:
        pytest.skip(f"flash_attn.cute is not importable: {completed.stderr.strip()}")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result_path.read_text())

    assert payload["out_shape"] == [7, 2, 8]
    assert payload["lse_shape"] == [2, 7]
    assert len(payload["calls"]) == 1
    forwarded = payload["calls"][0]
    assert "max_seqlen_q" not in forwarded
    assert "max_seqlen_k" not in forwarded
    assert forwarded["cu_seqlens_q_is_input"]
    assert forwarded["softmax_scale"] == 0.125
