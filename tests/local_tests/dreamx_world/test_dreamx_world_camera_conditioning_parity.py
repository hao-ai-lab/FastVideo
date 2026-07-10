# SPDX-License-Identifier: Apache-2.0
"""DreamX-World camera-conditioning parity against the official reference.

Coverage scope: implementation_subcomponent. This verifies the weightless
action-sequence to PRoPE camera tensor path used by DreamX-World-5B-Cam.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
from torch.testing import assert_close

from fastvideo.pipelines.basic.dreamx_world.camera_conditioning import build_dreamx_camera_condition

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.getenv("DREAMX_WORLD_OFFICIAL_REF_DIR", REPO_ROOT / "DreamX-World"))
PARITY_SCOPE = "implementation_subcomponent"


def _load_official_functions():
    if not OFFICIAL_REF_DIR.exists():
        pytest.skip(f"Official reference missing: {OFFICIAL_REF_DIR}")
    try:
        import importlib.util

        pose_path = OFFICIAL_REF_DIR / "utils" / "pose_utils.py"
        pose_spec = importlib.util.spec_from_file_location(
            "dreamx_world_pose_utils", pose_path)
        if pose_spec is None or pose_spec.loader is None:
            raise RuntimeError(f"Cannot load DreamX pose_utils: {pose_path}")
        pose_module = importlib.util.module_from_spec(pose_spec)
        pose_spec.loader.exec_module(pose_module)

        source = (OFFICIAL_REF_DIR / "utils" / "inference_utils.py").read_text()
        source = source.replace(
            "from .pose_utils import interpolate_camera_poses\n", "")
        namespace = {"interpolate_camera_poses": pose_module.interpolate_camera_poses}
        exec(compile(source, str(OFFICIAL_REF_DIR / "utils" / "inference_utils.py"), "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - local parity should skip missing reference deps.
        pytest.skip(f"Cannot load DreamX camera reference: {exc}")
    return namespace["ActionToPoseFromID"], namespace["GetPoseEmbedsFromPosesPrope"]


def _official_camera_condition(
    action_seq: list[str],
    action_speed_list: list[float],
    *,
    num_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype,
):
    action_to_pose, get_pose_embeds = _load_official_functions()
    duration = -(-num_frames // len(action_seq))
    poses = action_to_pose(action_seq, action_speed_list, duration=duration)[:num_frames]
    condition, _ = get_pose_embeds(poses, height, width, len(poses), False, 0, dtype=dtype, device="cpu")
    return condition


@pytest.mark.parametrize(
    ("action_seq", "action_speed_list", "num_frames"),
    [
        (["w"], [4], 81),
        (["wj", "d"], [4, 6], 121),
        (["i", "k", "l"], [3, 5, 2], 85),
    ],
)
def test_dreamx_world_camera_conditioning_matches_official(action_seq, action_speed_list, num_frames):
    dtype = torch.float32
    official = _official_camera_condition(
        action_seq,
        action_speed_list,
        num_frames=num_frames,
        height=704,
        width=1280,
        dtype=dtype,
    )
    fastvideo = build_dreamx_camera_condition(
        action_seq,
        action_speed_list,
        num_frames=num_frames,
        height=704,
        width=1280,
        dtype=dtype,
        device="cpu",
    )

    assert official.keys() == fastvideo.keys() == {"viewmats", "K"}
    for key in ("viewmats", "K"):
        assert official[key].shape == fastvideo[key].shape
        diff = (official[key] - fastvideo[key]).abs()
        print(f"{key}: shape={tuple(fastvideo[key].shape)} diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
        assert_close(fastvideo[key], official[key], atol=1e-5, rtol=1e-5)
