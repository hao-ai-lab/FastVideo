"""GPU torch backend — the CPU-verifiable tier (design_v3 §17).

The torch adapters/kernels themselves need a GPU box (see GPU_BRINGUP.md). What IS verifiable on a
torchless laptop is the wiring that must be right for the GPU path to even resolve correctly: the
cells are registered with honest sources, importing the backends never imports torch, a cuda
platform resolves the REAL cells (not a silent toy fallback), and building without torch fails loudly
rather than quietly returning a toy. Those are exactly the bugs that would otherwise surface only on
the box — so they're worth pinning here.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from v2.card import load_card
from v2.recipes.wan21 import build_wan21_card
from v2.platform import (
    COMPONENTS,
    FLOW_MATCH_STEP,
    FLOW_SDE_STEP,
    KERNELS,
    Platform,
    component_matrix,
    ensure_backends_loaded,
    kernel_matrix,
)


def _make_cuda_available(monkeypatch):
    """Flip every registered cuda cell to 'available' (auto-restored), so resolution can be tested
    without torch actually present."""
    ensure_backends_loaded()
    for reg in (*COMPONENTS._reg.values(), *KERNELS._reg.values()):
        if reg.key[1] == "cuda":
            monkeypatch.setattr(reg, "available", lambda: True)


# --------------------------------------------------------------------------- #
# Registration + honest sources                                               #
# --------------------------------------------------------------------------- #
def test_cuda_has_all_components_and_two_solver_ops():
    comps = sorted(r["kind"] for r in component_matrix() if r["device"] == "cuda")
    # + upsampler (LTX-2 spatial), audio_vae + vocoder (LTX-2.3 T2VS audio branch)
    assert comps == ["audio_vae", "dit", "text_encoder", "upsampler", "vae", "vocoder"]
    ops = sorted(r["op"] for r in kernel_matrix() if r["device"] == "cuda")
    assert ops == ["flow_match_step", "flow_sde_step"]


def test_cuda_solver_source_is_honest_no_fake_kernel():
    """fastvideo-kernel ships NO fused solver kernel — the cuda solver is plain torch. The source
    must say so, not claim a 'fastvideo-kernel:flow_*' that does not exist."""
    for r in kernel_matrix():
        if r["device"] == "cuda":
            assert "torch elementwise" in r["source"]
            assert "fastvideo-kernel" not in r["source"]
            assert r["arch"] == "generic"                 # elementwise: arch-agnostic, no static scratch


def test_cuda_cells_unavailable_on_this_box():
    assert all(not r["available"] for r in component_matrix() if r["device"] == "cuda")
    assert all(not r["available"] for r in kernel_matrix() if r["device"] == "cuda")


# --------------------------------------------------------------------------- #
# The load-bearing guard: importing the backends never imports torch          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch installed (GPU box): the cuda-availability probe imports torch by design to call "
           "torch.cuda.is_available(); this no-torch-import invariant is only verifiable without torch.")
def test_loading_backends_does_not_import_torch():
    ensure_backends_loaded()
    component_matrix(); kernel_matrix()
    assert "torch" not in sys.modules
    assert "v2.platform.backends.torch_adapters" not in sys.modules   # imported only inside builders
    assert "v2.platform.backends.torch_kernels" not in sys.modules


# --------------------------------------------------------------------------- #
# Resolution: when cuda IS available, the REAL cells win (no silent toy)       #
# --------------------------------------------------------------------------- #
def test_cuda_platform_resolves_real_cells_not_toy(monkeypatch):
    _make_cuda_available(monkeypatch)
    p = Platform.cuda("sm90")
    for kind in ("dit", "vae", "text_encoder"):
        reg = COMPONENTS.resolve_first(p._component_keys(kind, "default"))
        assert reg is not None and reg.key[1] == "cuda", kind     # the cuda adapter, not the cpu toy
    reg = p.resolve_kernel(FLOW_MATCH_STEP)
    assert reg.key[1] == "cuda" and "torch elementwise" in reg.source
    assert p.resolve_kernel(FLOW_SDE_STEP).key[1] == "cuda"


def test_cuda_build_fails_loudly_without_torch(monkeypatch):
    """With cuda 'available' but torch absent, building a component must RAISE (the lazy import of the
    torch adapter fails) — never silently hand back a numpy toy decoding real latents."""
    _make_cuda_available(monkeypatch)
    inst = load_card(build_wan21_card(), cache_manager=None, platform=Platform.cuda("sm90"))
    with pytest.raises((ImportError, ModuleNotFoundError, RuntimeError)):
        inst.component("transformer")


# --------------------------------------------------------------------------- #
# The checkpoint seam (risk A) is additive — toy cards are unaffected         #
# --------------------------------------------------------------------------- #
def test_checkpoint_field_defaults_empty_on_toy_cards():
    card = build_wan21_card()
    assert all(spec.checkpoint == "" for spec in card.components.values())   # CPU toys need no weights
