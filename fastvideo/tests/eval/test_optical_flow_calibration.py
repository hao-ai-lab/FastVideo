# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

from fastvideo.eval.metrics.optical_flow.synthetic_optical_flow._thirdperson import (
    load_calibration,
)


def test_load_legacy_synthetic_flow_calibration_defaults_pivot(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps({
            "alpha_yaw": 0.1,
            "alpha_pitch": 0.2,
            "alpha_turn": 0.3,
            "beta_fwd": 0.4,
            "beta_strafe": 0.5,
            "focal_length": 457.0,
        }),
        encoding="utf-8",
    )

    calibration = load_calibration(calibration_path)

    assert calibration.r_z == 0.0
    assert calibration.r_y == 0.0
    assert calibration.init_pitch == 0.0
