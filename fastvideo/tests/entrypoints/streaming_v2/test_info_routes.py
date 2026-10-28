# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/streamv2/capabilities`` and ``GET /v1/streamv2/health``."""
from __future__ import annotations

from starlette.testclient import TestClient

from fastvideo.entrypoints.streaming_v2.server import create_app


class _UnusedGenerator:

    def generate(self, request: object) -> object:
        raise AssertionError("info routes must not generate")


def test_capabilities_reports_hardcoded_h3_ref2va_facts() -> None:
    with TestClient(create_app(_UnusedGenerator())) as client:
        response = client.get("/v1/streamv2/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "h3-ref2va",
        "name": "H3 Ref2AV",
        "min_segment_duration_sec": 5,
        "max_segment_duration_sec": 15,
        "max_references": 12,
        "reference_media_types": {"image": {"max_count": 9}, "video": {"max_count": 3}, "audio": {"max_count": 3}},
        "max_reference_aspect_ratio": 4.0,
        "frame_sizes": {"16:9": {"720p": [1344, 768]}},
        "num_frames_by_duration_sec": {
            "5": 124, "6": 158, "7": 175, "8": 192, "9": 226, "10": 243,
            "11": 277, "12": 294, "13": 328, "14": 345, "15": 362,
        },
    }


def test_health_reports_ready() -> None:
    with TestClient(create_app(_UnusedGenerator())) as client:
        response = client.get("/v1/streamv2/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
