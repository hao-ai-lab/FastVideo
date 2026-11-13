# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex bucket geometry contract."""

import pytest

from fastvideo.pipelines.basic.pika_h3_flex import buckets


def test_released_bucket_tables_match_pika() -> None:
    assert buckets.bucket_table("p480") == (
        (960, 416),
        (832, 480),
        (704, 480),
        (640, 480),
        (608, 480),
        (480, 480),
        (480, 608),
        (480, 640),
        (480, 704),
        (480, 832),
        (416, 960),
    )
    assert buckets.bucket_table("p768") == (
        (1536, 672),
        (1344, 768),
        (1152, 768),
        (1024, 768),
        (960, 768),
        (768, 768),
        (768, 960),
        (768, 1024),
        (768, 1152),
        (768, 1344),
        (672, 1536),
    )


def test_unset_canvas_uses_default_tier_and_aspect() -> None:
    assert buckets.resolve_canvas(None, None) == (768, 1344, "p768", "16:9")


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (720, 1280, (768, 1344, "p768", "16:9")),
        (1080, 1920, (768, 1344, "p768", "16:9")),
        (480, 832, (480, 832, "p480", "16:9")),
        (1024, 768, (1024, 768, "p768", "3:4")),
        (960, 416, (960, 416, "p480", "9:21")),
    ],
)
def test_requested_canvas_snaps_to_bucket(height: int, width: int, expected: tuple[int, int, str, str]) -> None:
    assert buckets.resolve_canvas(height, width) == expected


def test_one_sided_canvas_is_rejected() -> None:
    with pytest.raises(ValueError, match="together"):
        buckets.resolve_canvas(720, None)


def test_tier_reference_short_edge_follows_the_output_canvas() -> None:
    assert buckets.reference_short_edge_for_canvas(480, 832) == 1024
    assert buckets.reference_short_edge_for_canvas(768, 1344) == 2048


def test_resolve_frames_aligns_to_the_17n_plus_5_grid() -> None:
    assert buckets.resolve_frames(120) == 124
    assert buckets.resolve_frames(124) == 124
    assert buckets.resolve_frames(345) == 345
    assert buckets.resolve_frames(360) == 362
    with pytest.raises(ValueError, match="Pika H3 Flex generates"):
        buckets.resolve_frames(107)


def test_unknown_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="released 480p and 768p"):
        buckets.resolve_tier("1080p")
