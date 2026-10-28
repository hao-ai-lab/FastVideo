# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/streamv2/capabilities``: what the served model can generate."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from fastvideo.entrypoints.streaming_v2.lora import lora_capabilities
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
)

if TYPE_CHECKING:
    from fastvideo.entrypoints.streaming_v2.server import ServerState

router = APIRouter()

# Hardcoded for MiniMax-H3 Ref2VA until FastVideo can derive these facts from
# the loaded model. Frame counts follow H3's causal-VAE grid (17 * n + 5) at
# 24 fps; `frame_sizes` is [width, height] by aspect ratio, then resolution.
# `max_references` caps the references across media types; `reference_media_types` gives each type's own cap.
CAPABILITIES: dict[str, Any] = {
    "model_id": "h3-ref2va",
    "name": "H3 Ref2AV",
    "min_segment_duration_sec": 5,
    "max_segment_duration_sec": 15,
    "max_references": MINIMAX_H3_MAX_REFERENCES,
    "reference_media_types": {
        "image": {
            "max_count": MINIMAX_H3_MAX_REFERENCE_IMAGES
        },
        "video": {
            "max_count": MINIMAX_H3_MAX_REFERENCE_VIDEOS
        },
        "audio": {
            "max_count": MINIMAX_H3_MAX_REFERENCE_AUDIOS
        },
    },
    "max_reference_aspect_ratio": 4.0,
    "frame_sizes": {
        "16:9": {
            "720p": [1344, 768]
        }
    },
    "num_frames_by_duration_sec": {
        "5": 124,
        "6": 158,
        "7": 175,
        "8": 192,
        "9": 226,
        "10": 243,
        "11": 277,
        "12": 294,
        "13": 328,
        "14": 345,
        "15": 362,
    },
}


@router.get("/v1/streamv2/capabilities")
async def get_capabilities(raw_request: Request) -> dict[str, Any]:
    """The served model's facts plus the ``lora`` block; a server with no adapter list reports it unsupported."""
    state: ServerState = raw_request.app.state.streaming_v2
    return {**CAPABILITIES, "lora": lora_capabilities(state.lora)}
