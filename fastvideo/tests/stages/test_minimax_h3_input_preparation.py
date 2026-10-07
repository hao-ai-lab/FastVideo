# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import resolve_target_num_frames


@pytest.mark.parametrize("requested, expected", [(120, 124), (345, 345), (346, 362), (360, 362), (362, 362)])
def test_target_frames_allow_padding_at_duration_limit(requested: int, expected: int) -> None:
    assert resolve_target_num_frames(requested) == expected
    assert resolve_target_num_frames(expected) == expected


@pytest.mark.parametrize("requested", [107, 363, 379])
def test_target_frames_reject_buckets_outside_duration_limits(requested: int) -> None:
    with pytest.raises(ValueError, match="MiniMax-H3 generates"):
        resolve_target_num_frames(requested)


@pytest.mark.parametrize("requested", [360.0, "360", None])
def test_target_frames_require_integer(requested: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        resolve_target_num_frames(requested)
