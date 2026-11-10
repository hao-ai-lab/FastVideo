# SPDX-License-Identifier: Apache-2.0
"""Output geometry advertised by a loaded pipeline, independent of device capacity."""
from typing import Any


def video_output_capabilities(args: Any) -> dict[str, Any] | None:
    """Return known output constraints, or None for an unregistered family.

    Recommended sizes are valid request examples, not an exhaustive canvas list
    or a claim that the deployment has enough memory for every combination.
    """
    from fastvideo.registry import get_preset_selection

    _, family = get_preset_selection(args.model_path)
    if family not in {"minimax_h3", "pika_h3_flex"}:
        return None

    from fastvideo.pipelines.basic.minimax_h3.packing import (
        MINIMAX_H3_CANVAS_MULTIPLE,
        MINIMAX_H3_FPS,
        MINIMAX_H3_MAX_DURATION,
        MINIMAX_H3_MAX_PIXELS,
        MINIMAX_H3_MIN_DURATION,
        align_num_frames,
        resolve_canvas_size,
    )
    from fastvideo.pipelines.basic.pika_h3_flex.buckets import PIKA_ASPECTS, canvas_for_aspect

    aspects = ("16:9", *(aspect for aspect in PIKA_ASPECTS if aspect != "16:9"))
    tiers = {"480p": (480, 832 * 480), "768p": (768, MINIMAX_H3_MAX_PIXELS)}
    sizes: dict[str, dict[str, list[int]]] = {}
    for aspect in aspects:
        aw, ah = (int(value) for value in aspect.split(":"))
        sizes[aspect] = {}
        for resolution, (short_edge, max_pixels) in tiers.items():
            if family == "pika_h3_flex":
                height, width = canvas_for_aspect(aspect, resolution)
            else:
                height, width = resolve_canvas_size(aw, ah, short_edge=short_edge, max_pixels=max_pixels)
                # Nearest-lattice rounding can exceed the area cap. These are recommendations for the
                # explicit width/height path, whose adapter enforces the cap before admission.
                while width * height > max_pixels:
                    if width >= height:
                        width -= MINIMAX_H3_CANVAS_MULTIPLE
                    else:
                        height -= MINIMAX_H3_CANVAS_MULTIPLE
            sizes[aspect][resolution] = [width, height]
    return {
        "schema_version": 1,
        "fps": MINIMAX_H3_FPS,
        "duration_seconds": {
            "min": int(MINIMAX_H3_MIN_DURATION),
            "max": int(MINIMAX_H3_MAX_DURATION)
        },
        "num_frames_by_duration_seconds": {
            str(seconds): align_num_frames(seconds * MINIMAX_H3_FPS)
            for seconds in range(int(MINIMAX_H3_MIN_DURATION),
                                 int(MINIMAX_H3_MAX_DURATION) + 1)
        },
        "recommended_frame_sizes": sizes,
        "canvas_multiple": MINIMAX_H3_CANVAS_MULTIPLE,
        "hardware_capacity_verified": False,
    }
