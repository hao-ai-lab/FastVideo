# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Muon optimizer (CPU; no distributed required).

Covers the three correctness surfaces of ``fastvideo/train/utils/muon.py``:
the Newton-Schulz orthogonalization, the muon-vs-aux parameter split, and a
single-optimizer step that reduces a quadratic loss with finite updates. The
FSDP2 / DTensor gather-per-matrix path is exercised separately on multi-GPU
hardware (not in CPU CI).
"""

from __future__ import annotations

import torch

from fastvideo.train.utils.muon import (
    MuonWithAuxAdam,
    split_params_for_muon,
    zeropower_via_newtonschulz5,
)


def test_newton_schulz_orthogonalizes() -> None:
    # The 5-step quintic pushes singular values into a band around 1 (it is
    # deliberately approximate, not exact orthogonalization).
    torch.manual_seed(0)
    for shape in [(64, 128), (128, 64), (96, 96)]:
        g = torch.randn(*shape)
        o = zeropower_via_newtonschulz5(g, steps=5).float()
        assert o.shape == g.shape
        s = torch.linalg.svdvals(o)
        assert torch.isfinite(s).all()
        # The 5-step quintic lands singular values in a band around 1 (wider
        # for square inputs); it is approximate by design, not exact.
        assert 0.3 < s.min() and s.max() < 1.5


def _named(specs: list[tuple[str, tuple[int, ...]]]):
    return [(n, torch.nn.Parameter(torch.randn(*shp))) for n, shp in specs]


def test_split_keeps_hidden_matmuls_excludes_embed_head_and_1d() -> None:
    named = _named([
        ("patch_embed.weight", (16, 3, 2, 2, 2)),       # conv (>2D) -> aux
        ("blocks.0.attn.to_q.weight", (16, 16)),         # hidden 2D -> muon
        ("blocks.0.attn.to_q.bias", (16, )),             # 1D -> aux
        ("blocks.0.attn.to_out.weight", (16, 16)),       # attn out -> muon
        ("blocks.0.mlp.fc1.weight", (64, 16)),           # hidden 2D -> muon
        ("blocks.0.norm.weight", (16, )),                # 1D -> aux
        ("text_embedder.weight", (16, 32)),              # embed -> aux
        ("proj_out.weight", (3, 16)),                    # output head -> aux
    ])
    muon, aux = split_params_for_muon(named)
    muon_names = {n for n, p in named if any(p is q for q in muon)}
    aux_names = {n for n, p in named if any(p is q for q in aux)}
    assert muon_names == {
        "blocks.0.attn.to_q.weight",
        "blocks.0.attn.to_out.weight",
        "blocks.0.mlp.fc1.weight",
    }
    for excluded in ("patch_embed.weight", "text_embedder.weight",
                     "proj_out.weight", "blocks.0.norm.weight",
                     "blocks.0.attn.to_q.bias"):
        assert excluded in aux_names


def test_split_skips_non_trainable() -> None:
    p = torch.nn.Parameter(torch.randn(8, 8))
    p.requires_grad_(False)
    muon, aux = split_params_for_muon([("blocks.0.w.weight", p)])
    assert muon == [] and aux == []


def test_muon_with_aux_adam_step_reduces_loss_and_is_finite() -> None:
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 32))   # muon group
    b = torch.nn.Parameter(torch.zeros(32))       # aux (adam) group
    opt = MuonWithAuxAdam([w], [b], lr=0.05, momentum=0.95, ns_steps=5,
                          aux_lr=0.02)
    tw, tb = torch.randn(32, 32), torch.randn(32)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = ((w - tw)**2).mean() + ((b - tb)**2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert torch.isfinite(w).all() and torch.isfinite(b).all()
    # Both groups must make progress (Muon orthogonalizes the update, so it is
    # steadier than GD but still monotonically descends this convex problem).
    assert losses[-1] < 0.5 * losses[0]


def test_muon_requires_some_params() -> None:
    try:
        MuonWithAuxAdam([], [], lr=0.01)
    except ValueError:
        return
    raise AssertionError("MuonWithAuxAdam([], []) should raise ValueError")
