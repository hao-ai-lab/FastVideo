# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex resolution buckets and frame grid.

Pika H3 Flex keeps MiniMax-H3's canvas rule: sides sit on a 32-pixel lattice,
a fixed short edge bounds the bucket, and an area cap keeps the token count
finite.  The open-weight release samples at two of the released tiers, 480p and
768p; the hosted API's larger tiers are intentionally not exposed here because
the local checkpoints were not trained for them.

A request selects a bucket, not a free-form canvas.  ``height`` and ``width``
are therefore aspect hints: an exact bucket is honoured, anything else is
snapped to the nearest aspect family of the tier implied by the requested
short edge.  Keeping this arithmetic in a torch-free module lets the geometry
contract be tested on a CPU-only host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable

PIKA_FPS = 24
PIKA_CANVAS_MULTIPLE = 32
PIKA_MIN_ASPECT_RATIO = 0.25
PIKA_MAX_ASPECT_RATIO = 4.0

# Landscape families first, then square, then the mirrored portraits.
PIKA_ASPECTS: tuple[str, ...] = (
    "21:9",
    "16:9",
    "3:2",
    "4:3",
    "5:4",
    "1:1",
    "4:5",
    "3:4",
    "2:3",
    "9:16",
    "9:21",
)

# 17n + 5 from 2.3 s to 15.1 s at 24 fps.  Open-weight requests are limited to
# the documented 5-15 s range; the two smaller grid points remain part of the
# released grid but are rejected by the request validator.
PIKA_FRAME_GRID: tuple[int, ...] = tuple(17 * n + 5 for n in range(3, 22))
PIKA_MIN_DURATION = 5.0
PIKA_MAX_DURATION = 15.0
PIKA_MIN_ALIGNED_FRAMES = 120
PIKA_MAX_ALIGNED_FRAMES = 362


@dataclass(frozen=True)
class PikaTier:
    """One released resolution tier."""

    name: str
    short_edge: int
    max_pixels: int
    reference_short_edge: int


PIKA_TIERS: dict[str, PikaTier] = {
    "p480": PikaTier("p480", short_edge=480, max_pixels=832 * 480, reference_short_edge=1024),
    "p768": PikaTier("p768", short_edge=768, max_pixels=1344 * 768, reference_short_edge=2048),
}
DEFAULT_TIER = "p768"
DEFAULT_ASPECT: tuple[int, int] = (16, 9)

_TIER_ALIASES = {
    "480p": "p480",
    "p480": "p480",
    "480": "p480",
    "768p": "p768",
    "p768": "p768",
    "768": "p768",
}


def resolve_tier(value: str | None) -> str:
    """Normalize ``480p`` / ``p480`` / ``768p`` / ``p768`` to a released tier name."""
    if value is None:
        return DEFAULT_TIER
    key = str(value).strip().lower()
    tier = _TIER_ALIASES.get(key)
    if tier is None:
        raise ValueError(f"Pika H3 Flex only samples the released 480p and 768p tiers; got {value!r}.")
    return tier


def tier_rule(tier: str | None) -> PikaTier:
    name = resolve_tier(tier)
    return PIKA_TIERS[name]


def _parse_aspect(aspect: str | tuple[int, int] | tuple[float, float]) -> tuple[float, float]:
    if isinstance(aspect, str):
        parts = aspect.split(":")
        if len(parts) != 2:
            raise ValueError(f"An aspect ratio must be W:H, got {aspect!r}.")
        width, height = float(parts[0]), float(parts[1])
    elif isinstance(aspect, tuple) and len(aspect) == 2:
        width, height = float(aspect[0]), float(aspect[1])
    else:
        raise TypeError(f"An aspect ratio must be a 'W:H' string or a (width, height) pair, got {aspect!r}.")
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        raise ValueError(f"An aspect ratio must be finite and positive, got {aspect!r}.")
    return width, height


def aspect_ratio(width: float, height: float) -> float:
    width, height = _parse_aspect((width, height))
    return width / height


def canvas_for_aspect(aspect: str | tuple[float, float], tier: str | None = None) -> tuple[int, int]:
    """``(height, width)`` of one tier's bucket for an aspect family."""
    rule = tier_rule(tier)
    aspect_width, aspect_height = _parse_aspect(aspect)
    ratio = aspect_width / aspect_height
    if not PIKA_MIN_ASPECT_RATIO <= ratio <= PIKA_MAX_ASPECT_RATIO:
        raise ValueError(f"Pika H3 Flex supports aspect ratios from 1:4 to 4:1, got {aspect!r}.")

    if ratio >= 1:
        width, height = rule.short_edge * ratio, float(rule.short_edge)
    else:
        width, height = float(rule.short_edge), rule.short_edge / ratio
    area = width * height
    if area > rule.max_pixels:
        scale = (rule.max_pixels / area)**0.5
        width, height = width * scale, height * scale
    multiple = PIKA_CANVAS_MULTIPLE
    return (
        max(multiple,
            round(height / multiple) * multiple),
        max(multiple,
            round(width / multiple) * multiple),
    )


def aspect_families(tier: str | None = None) -> tuple[str, ...]:
    del tier
    return PIKA_ASPECTS


def bucket_table(tier: str | None = None) -> tuple[tuple[int, int], ...]:
    """All ``(width, height)`` buckets of one tier, in :data:`PIKA_ASPECTS` order."""
    return tuple((canvas_for_aspect(aspect, tier)[1], canvas_for_aspect(aspect, tier)[0]) for aspect in PIKA_ASPECTS)


def iter_buckets() -> Iterable[tuple[str, str, int, int]]:
    """Yield ``(tier, aspect, width, height)`` for every released bucket."""
    for tier in PIKA_TIERS:
        for aspect in PIKA_ASPECTS:
            height, width = canvas_for_aspect(aspect, tier)
            yield tier, aspect, width, height


def exact_bucket(height: int, width: int) -> tuple[str, str] | None:
    """The ``(tier, aspect)`` whose bucket is exactly ``height x width``, if any."""
    for tier, aspect, bucket_width, bucket_height in iter_buckets():
        if (bucket_height, bucket_width) == (int(height), int(width)):
            return tier, aspect
    return None


def tier_for_canvas(height: int, width: int) -> str:
    """The smallest released tier whose short edge covers the canvas."""
    if height <= 0 or width <= 0:
        raise ValueError(f"A canvas must be positive, got {height}x{width}.")
    short = min(int(height), int(width))
    for tier in sorted(PIKA_TIERS.values(), key=lambda rule: rule.short_edge):
        if short <= tier.short_edge:
            return tier.name
    return max(PIKA_TIERS.values(), key=lambda rule: rule.short_edge).name


def reference_short_edge_for_canvas(height: int, width: int) -> int:
    """Reference-image short edge the released tier rule uses for an output canvas."""
    return PIKA_TIERS[tier_for_canvas(height, width)].reference_short_edge


def _nearest_aspect(ratio: float, tier: str) -> tuple[int, int, str]:
    best: tuple[float, str, tuple[int, int]] | None = None
    for aspect in PIKA_ASPECTS:
        family_ratio = aspect_ratio(*_parse_aspect(aspect))
        distance = abs(math.log(ratio / family_ratio))
        if best is None or distance < best[0]:
            best = (distance, aspect, canvas_for_aspect(aspect, tier))
        # PIKA_ASPECTS is ordered by family; equal distances keep the earlier one.
    assert best is not None
    height, width = best[2]
    return height, width, best[1]


def nearest_bucket(height: int, width: int, tier: str) -> tuple[int, int, str]:
    """Snap ``height x width`` to the closest aspect family of ``tier``."""
    if height <= 0 or width <= 0:
        raise ValueError(f"A canvas must be positive, got {height}x{width}.")
    ratio = width / height
    if not PIKA_MIN_ASPECT_RATIO - 1e-9 <= ratio <= PIKA_MAX_ASPECT_RATIO + 1e-9:
        raise ValueError(f"Pika H3 Flex supports aspect ratios from 1:4 to 4:1, got {width}x{height}.")
    return _nearest_aspect(ratio, tier)


def resolve_canvas(
    height: int | None,
    width: int | None,
    *,
    default_aspect: str | tuple[float, float] = DEFAULT_ASPECT,
    default_tier: str | None = DEFAULT_TIER,
) -> tuple[int, int, str, str]:
    """Resolve a request canvas to ``(height, width, tier, aspect)``.

    With neither side set, the default tier and aspect are used.  With both
    sides set, an exact released bucket is honoured; otherwise the tier follows
    the requested short edge (``<= 480`` is 480p, everything else 768p) and the
    canvas snaps to the nearest aspect family.  One-sided requests are rejected.
    """
    if (height is None) != (width is None):
        raise ValueError("Pika H3 Flex `height` and `width` must be passed together, or neither.")
    if height is None:
        tier = resolve_tier(default_tier)
        aspect_width, aspect_height = _parse_aspect(default_aspect)
        bucket_height, bucket_width = canvas_for_aspect((aspect_width, aspect_height), tier)
        aspect = PIKA_ASPECTS[0]
        for family in PIKA_ASPECTS:
            if canvas_for_aspect(family, tier) == (bucket_height, bucket_width):
                aspect = family
                break
        return bucket_height, bucket_width, tier, aspect

    if not isinstance(height, int) or isinstance(height, bool) or not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("Pika H3 Flex `height` and `width` must be integers.")
    if height <= 0 or width <= 0:
        raise ValueError(f"Pika H3 Flex `height` and `width` must be positive, got {height}x{width}.")

    exact = exact_bucket(height, width)
    if exact is not None:
        tier, aspect = exact
        return height, width, tier, aspect

    tier = "p480" if min(height, width) <= PIKA_TIERS["p480"].short_edge else "p768"
    bucket_height, bucket_width, aspect = nearest_bucket(height, width, tier)
    return bucket_height, bucket_width, tier, aspect


def resolve_frames(num_frames: int) -> int:
    """Round a request up to Pika's ``17n + 5`` grid inside the 5-15 s range."""
    if not isinstance(num_frames, int) or isinstance(num_frames, bool):
        raise TypeError(f"`num_frames` must be an integer, got {type(num_frames).__name__}.")
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    aligned = num_frames
    while aligned % 17 != 5:
        aligned += 1
    if not PIKA_MIN_ALIGNED_FRAMES <= aligned <= PIKA_MAX_ALIGNED_FRAMES:
        raise ValueError(f"Pika H3 Flex generates {PIKA_MIN_DURATION:g}-{PIKA_MAX_DURATION:g} seconds at "
                         f"{PIKA_FPS} fps; aligned num_frames={aligned} is outside the accepted "
                         f"{PIKA_MIN_ALIGNED_FRAMES}-{PIKA_MAX_ALIGNED_FRAMES} frame range.")
    return aligned


def duration_seconds(num_frames: int) -> float:
    return int(num_frames) / PIKA_FPS


__all__ = [
    "DEFAULT_ASPECT",
    "DEFAULT_TIER",
    "PIKA_ASPECTS",
    "PIKA_CANVAS_MULTIPLE",
    "PIKA_FPS",
    "PIKA_FRAME_GRID",
    "PIKA_MAX_ALIGNED_FRAMES",
    "PIKA_MAX_DURATION",
    "PIKA_MIN_ALIGNED_FRAMES",
    "PIKA_MIN_DURATION",
    "PIKA_TIERS",
    "PikaTier",
    "aspect_ratio",
    "bucket_table",
    "canvas_for_aspect",
    "duration_seconds",
    "exact_bucket",
    "iter_buckets",
    "nearest_bucket",
    "reference_short_edge_for_canvas",
    "resolve_canvas",
    "resolve_frames",
    "resolve_tier",
    "tier_for_canvas",
    "tier_rule",
]
