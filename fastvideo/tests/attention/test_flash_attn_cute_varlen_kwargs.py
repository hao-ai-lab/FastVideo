# SPDX-License-Identifier: Apache-2.0
"""FA4 varlen forwards optional kwargs only when the installed kernel declares them.

``FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN=1`` routes H3 inference through the
varlen custom ops. Some ``flash_attn.cute`` builds reject ``max_seqlen_q`` /
``max_seqlen_k`` (and, in backward, ``window_size_*`` / ``deterministic``) with
``TypeError: ... got an unexpected keyword argument``; the pinned build accepts
them and sizes its grid from the max lengths, so they must still reach it.

Each case installs a fake ``flash_attn.cute.interface`` with one of the two
signatures, runs a packed-varlen forward and backward, and runs in a subprocess
so the fake modules and the imported kernel references cannot leak into other
tests.
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

calls = {"fwd": [], "bwd": []}


def _outputs(q):
    return torch.empty_like(q), torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32)


def legacy_fwd(q, k, v, *, cu_seqlens_q=None, cu_seqlens_k=None, softmax_scale=None, causal=False,
               window_size_left=None, window_size_right=None, softcap=0.0, num_splits=1, pack_gqa=None,
               return_lse=True):
    calls["fwd"].append({"cu_seqlens_q_is_input": cu_seqlens_q is cu_seqlens, "softmax_scale": softmax_scale})
    return _outputs(q)


def legacy_bwd(q, k, v, out, dout, lse, softmax_scale=None, causal=False, softcap=0.0, cu_seqlens_q=None,
               cu_seqlens_k=None, seqused_q=None, seqused_k=None):
    calls["bwd"].append({"cu_seqlens_q_is_input": cu_seqlens_q is cu_seqlens, "softmax_scale": softmax_scale})
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def pinned_fwd(q, k, v, *, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None,
               softmax_scale=None, causal=False, window_size_left=None, window_size_right=None, softcap=0.0,
               num_splits=1, pack_gqa=None, return_lse=True):
    calls["fwd"].append({"cu_seqlens_q_is_input": cu_seqlens_q is cu_seqlens, "softmax_scale": softmax_scale,
                         "max_seqlen_q": max_seqlen_q, "max_seqlen_k": max_seqlen_k})
    return _outputs(q)


def pinned_bwd(q, k, v, out, dout, lse, softmax_scale=None, causal=False, softcap=0.0, window_size_left=None,
               window_size_right=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
               max_seqlen_k=None, deterministic=False):
    calls["bwd"].append({"cu_seqlens_q_is_input": cu_seqlens_q is cu_seqlens, "softmax_scale": softmax_scale,
                         "max_seqlen_q": max_seqlen_q, "max_seqlen_k": max_seqlen_k,
                         "deterministic": deterministic})
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


fake_fwd, fake_bwd = (pinned_fwd, pinned_bwd) if sys.argv[2] == "pinned" else (legacy_fwd, legacy_bwd)
flash_attn = types.ModuleType("flash_attn")
cute = types.ModuleType("flash_attn.cute")
interface = types.ModuleType("flash_attn.cute.interface")
for module, name in ((flash_attn, "flash_attn"), (cute, "flash_attn.cute"), (interface, "flash_attn.cute.interface")):
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
interface._flash_attn_fwd = fake_fwd
interface._flash_attn_bwd = fake_bwd
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
module._flash_attn_fwd = fake_fwd
module._flash_attn_bwd = fake_bwd

q = torch.randn(7, 2, 8, device="cuda", requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu_seqlens = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
out, lse = torch.ops.fastvideo._flash_attn_cute_varlen_forward(q, k, v, cu_seqlens, cu_seqlens, 7, 7, 0.125, False,
                                                              False)
out.sum().backward()

with open(sys.argv[1], "w") as handle:
    json.dump({"out_shape": list(out.shape), "lse_shape": list(lse.shape), "q_grad": q.grad is not None,
               "calls": calls}, handle)
'''


def _run(tmp_path, signature: str) -> dict:
    if not torch.cuda.is_available():
        pytest.skip("flash_attn.cute is only imported on CUDA builds")
    result_path = tmp_path / f"varlen_{signature}.json"
    completed = subprocess.run([sys.executable, "-c", _SUBPROCESS_SCRIPT,
                                str(result_path), signature],
                               capture_output=True,
                               text=True,
                               check=False)
    if completed.returncode == _SKIP_EXIT_CODE:
        pytest.skip(f"flash_attn.cute is not importable: {completed.stderr.strip()}")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result_path.read_text())
    assert payload["out_shape"] == [7, 2, 8]
    assert payload["lse_shape"] == [2, 7]
    assert payload["q_grad"]
    assert len(payload["calls"]["fwd"]) == 1
    assert len(payload["calls"]["bwd"]) == 1
    for call in (payload["calls"]["fwd"][0], payload["calls"]["bwd"][0]):
        assert call["cu_seqlens_q_is_input"]
        assert call["softmax_scale"] == 0.125
    return payload["calls"]


def test_varlen_omits_kwargs_the_kernel_rejects(tmp_path) -> None:
    calls = _run(tmp_path, "legacy")
    assert "max_seqlen_q" not in calls["fwd"][0]
    assert "max_seqlen_q" not in calls["bwd"][0]


def test_varlen_forwards_max_seqlen_when_the_kernel_accepts_it(tmp_path) -> None:
    calls = _run(tmp_path, "pinned")
    assert calls["fwd"][0]["max_seqlen_q"] == 7
    assert calls["fwd"][0]["max_seqlen_k"] == 7
    assert calls["bwd"][0]["max_seqlen_q"] == 7
    assert calls["bwd"][0]["max_seqlen_k"] == 7
    assert calls["bwd"][0]["deterministic"] is False
