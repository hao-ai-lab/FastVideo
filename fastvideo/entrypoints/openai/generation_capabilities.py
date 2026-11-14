# SPDX-License-Identifier: Apache-2.0
"""Output geometry advertised by a loaded pipeline, independent of device capacity."""
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def video_output_capabilities(args: Any) -> dict[str, Any] | None:
    """Return known output constraints, or None for an unregistered family.

    Recommended sizes are valid request examples, not an exhaustive canvas list
    or a claim that the deployment has enough memory for every combination.
    """
    from fastvideo.registry import get_preset_selection

    try:
        # Resolve locally: model discovery must never gate /v1/model_info on a Hub lookup.
        _, family = get_preset_selection(args.model_path, allow_remote=False)
    except Exception as exc:  # discovery is best-effort; report unknown instead of failing the request
        logger.warning("Could not resolve output capabilities for %r: %s", args.model_path, exc)
        return None
    if family not in {"minimax_h3", "pika_h3_flex"}:
        return None

    from fastvideo.pipelines.basic.minimax_h3.packing import (
        MINIMAX_H3_CANVAS_MULTIPLE,
        MINIMAX_H3_FPS,
        MINIMAX_H3_MAX_DURATION,
        MINIMAX_H3_MIN_DURATION,
        MINIMAX_H3_RESOLUTION_TIERS,
        align_num_frames,
        resolve_canvas_size,
    )
    from fastvideo.pipelines.basic.pika_h3_flex.buckets import PIKA_ASPECTS, canvas_for_aspect

    aspects = ("16:9", *(aspect for aspect in PIKA_ASPECTS if aspect != "16:9"))
    tiers = MINIMAX_H3_RESOLUTION_TIERS
    sizes: dict[str, dict[str, list[int]]] = {}
    for aspect in aspects:
        aw, ah = (int(value) for value in aspect.split(":"))
        sizes[aspect] = {}
        for resolution, (short_edge, max_pixels) in tiers.items():
            if family == "pika_h3_flex":
                height, width = canvas_for_aspect(aspect, resolution)
            else:
                height, width = resolve_canvas_size(aw, ah, short_edge=short_edge, max_pixels=max_pixels)
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
        **_generation_tasks(args, family),
        "hardware_capacity_verified": False,
    }


def _generation_tasks(args: Any, family: str) -> dict[str, Any]:
    """The request tasks this server accepts (``task`` values and the media each one takes)."""
    from fastvideo.entrypoints.openai.request_adapter import is_ref2va_server

    if not is_ref2va_server(args, family):
        return {
            "tasks": [
                {
                    "task": "t2va",
                    "image_references": 0
                },
                {
                    "task": "i2va",
                    "image_references": 1,
                    "anchors": ["first"]
                },
                {
                    "task": "fl2va",
                    "image_references": 2,
                    "anchors": ["first", "last"]
                },
            ],
        }
    from fastvideo.pipelines.basic.minimax_h3.reference import (
        MINIMAX_H3_MAX_REFERENCE_AUDIOS,
        MINIMAX_H3_MAX_REFERENCE_IMAGES,
        MINIMAX_H3_MAX_REFERENCE_VIDEOS,
        MINIMAX_H3_MAX_REFERENCES,
    )

    tasks: list[dict[str, Any]] = [{"task": "t2va", "image_references": 0}] if family == "minimax_h3" else []
    tasks.append({"task": "ref2va"})
    return {
        "tasks": tasks,
        "reference_limits": {
            "images": MINIMAX_H3_MAX_REFERENCE_IMAGES,
            "videos": MINIMAX_H3_MAX_REFERENCE_VIDEOS,
            "audios": MINIMAX_H3_MAX_REFERENCE_AUDIOS,
            "total": MINIMAX_H3_MAX_REFERENCES,
            "continuation_video": family == "minimax_h3",
        },
    }
