# SPDX-License-Identifier: Apache-2.0
"""Multi-frame KV-cache parity for Waypoint.

The single-frame transformer parity test cannot catch a stateful (cross-frame)
cache bug — the original flat KV ring matched the official on frame 0 but
diverged from frame 2 because it ignored the official per-layer local/global +
dilated windowing. This test drives the production
``GatedSelfAttention._upsert_kv_cache`` (a pure ring-update; needs no model) and
the official ``LayerKVCache.upsert`` with identical writes for many frames and
asserts the stored K/V buffers + the validity mask stay identical for both a
local layer and a (dilated) global layer.

Requires the official Overworld remote code (set ``WAYPOINT_OVERWORLD_PATH`` or
download ``Overworld/Waypoint-1-Small`` to ``models/Waypoint-1-Small``) and
``tensordict``; skips otherwise.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

OVERWORLD = os.environ.get("WAYPOINT_OVERWORLD_PATH", "models/Waypoint-1-Small")

TPF = 256
HEAD_DIM = 64
N_KV_HEADS = 20
N_FRAMES = 40  # exercises eviction (local L=16) and dilation wrap (global)


def _load_official_cache_classes():
    """Import the official LayerKVCache from the Overworld remote code."""
    bd = os.path.join(OVERWORLD, "before_denoise.py")
    if not os.path.isfile(bd):
        pytest.skip(f"Overworld official code not found at {bd}")
    try:
        import tensordict  # noqa: F401
    except ImportError:
        pytest.skip("tensordict required for the official Overworld cache")
    pkg = "wp_official_cache_test"
    mod = types.ModuleType(pkg)
    mod.__path__ = [OVERWORLD]
    mod.__package__ = pkg
    sys.modules[pkg] = mod
    spec = importlib.util.spec_from_file_location(f"{pkg}.before_denoise", bd)
    sub = importlib.util.module_from_spec(spec)
    sub.__package__ = pkg
    sys.modules[f"{pkg}.before_denoise"] = sub
    spec.loader.exec_module(sub)
    from tensordict import TensorDict
    return sub.LayerKVCache, TensorDict


def _fv_layer_cache(L, pinned_dilation, device, dtype):
    """Per-layer cache dict matching production _create_waypoint_kv_cache."""
    capacity = L + TPF
    written = torch.zeros(capacity, dtype=torch.bool, device=device)
    written[L:] = True
    offsets = torch.arange(TPF, dtype=torch.long, device=device)
    return {
        "k": torch.zeros(1, N_KV_HEADS, capacity, HEAD_DIM, device=device, dtype=dtype),
        "v": torch.zeros(1, N_KV_HEADS, capacity, HEAD_DIM, device=device, dtype=dtype),
        "written": written,
        "L": L,
        "tpf": TPF,
        "pinned_dilation": pinned_dilation,
        "num_buckets": (L // TPF) // pinned_dilation,
        "frame_offsets": offsets,
        "current_idx": offsets + L,
        "frozen_ref": [False],
    }


@pytest.mark.parametrize("kind,local_window,global_window,dilation", [
    ("local", 16, 128, 1),
    ("global", 16, 128, 8),
])
def test_waypoint_kv_cache_matches_official(kind, local_window, global_window, dilation):
    from fastvideo.models.dits.waypoint_transformer import GatedSelfAttention

    LayerKVCache, TensorDict = _load_official_cache_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    L = (global_window if kind == "global" else local_window) * TPF

    fv = _fv_layer_cache(L, dilation, device, dtype)
    official = LayerKVCache(1, N_KV_HEADS, L, HEAD_DIM, dtype, TPF, dilation).to(device)

    torch.manual_seed(0)
    for frame_t in range(N_FRAMES):
        k = torch.randn(1, N_KV_HEADS, TPF, HEAD_DIM, device=device, dtype=dtype)
        v = torch.randn(1, N_KV_HEADS, TPF, HEAD_DIM, device=device, dtype=dtype)

        # production ring update (pure method; self unused)
        GatedSelfAttention._upsert_kv_cache(None, fv, k, v, frame_t, update_cache=True)

        # official ring update
        t_pos = torch.full((1, TPF), frame_t, device=device, dtype=torch.long)
        pos_ids = TensorDict({"t_pos": t_pos}, batch_size=[1, TPF])
        off_k, off_v, _ = official.upsert(torch.stack([k, v], 0), pos_ids, is_frozen=False)

        assert torch.equal(fv["written"], official.written), (
            f"{kind} layer: written-slot mismatch at frame {frame_t}")
        assert torch.allclose(fv["k"], off_k), f"{kind}: K buffer mismatch at frame {frame_t}"
        assert torch.allclose(fv["v"], off_v), f"{kind}: V buffer mismatch at frame {frame_t}"
