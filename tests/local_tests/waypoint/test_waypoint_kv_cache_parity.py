# SPDX-License-Identifier: Apache-2.0
"""Waypoint KV-ring parity against the published implementation."""

import importlib.util
import os
import sys
import types

import pytest
import torch

OVERWORLD = os.environ.get("WAYPOINT_OVERWORLD_PATH", "models/Waypoint-1-Small")
TOKENS_PER_FRAME = 256
HEAD_DIM = 64
KV_HEADS = 20


def _official_cache():
    source = os.path.join(OVERWORLD, "before_denoise.py")
    if not os.path.isfile(source):
        pytest.skip(f"Official Waypoint source not found at {source}")
    pytest.importorskip("tensordict")
    package = "waypoint_official_cache_parity"
    module = types.ModuleType(package)
    module.__path__ = [OVERWORLD]
    module.__package__ = package
    sys.modules[package] = module
    spec = importlib.util.spec_from_file_location(f"{package}.before_denoise", source)
    implementation = importlib.util.module_from_spec(spec)
    implementation.__package__ = package
    sys.modules[spec.name] = implementation
    spec.loader.exec_module(implementation)
    implementation.make_block_mask = lambda *_args, **_kwargs: None
    return implementation.LayerKVCache


@pytest.mark.parametrize(
    "history_frames,dilation",
    [(16, 1), (128, 8)],
)
def test_waypoint_kv_cache_matches_official(history_frames, dilation):
    from tensordict import TensorDict

    from fastvideo.models.dits.waypoint_transformer import WaypointLayerKVCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = history_frames * TOKENS_PER_FRAME
    native = WaypointLayerKVCache(
        1,
        KV_HEADS,
        history,
        HEAD_DIM,
        torch.float32,
        TOKENS_PER_FRAME,
        dilation,
    ).to(device)
    official = _official_cache()(
        1,
        KV_HEADS,
        history,
        HEAD_DIM,
        torch.float32,
        TOKENS_PER_FRAME,
        dilation,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(0)
    for frame_index in range(40):
        key = torch.randn(
            1,
            KV_HEADS,
            TOKENS_PER_FRAME,
            HEAD_DIM,
            generator=generator,
            device=device,
        )
        value = torch.randn_like(key)
        native.upsert(key, value, frame_index, is_frozen=False)
        positions = TensorDict(
            {
                "t_pos": torch.full(
                    (1, TOKENS_PER_FRAME),
                    frame_index,
                    device=device,
                    dtype=torch.long,
                )
            },
            batch_size=[1, TOKENS_PER_FRAME],
        )
        official.upsert(torch.stack((key, value)), positions, is_frozen=False)

        assert torch.equal(native.written, official.written)
        assert torch.equal(native.kv, official.kv)
