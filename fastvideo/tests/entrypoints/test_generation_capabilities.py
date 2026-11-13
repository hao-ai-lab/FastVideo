# SPDX-License-Identifier: Apache-2.0
"""Discovery follows H3 geometry instead of treating serve defaults as limits."""
from types import SimpleNamespace

import pytest

from fastvideo.entrypoints.openai.generation_capabilities import video_output_capabilities
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_MAX_PIXELS
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import resolve_target_num_frames
from fastvideo.pipelines.basic.pika_h3_flex.buckets import (PIKA_FPS, PIKA_MAX_DURATION, PIKA_MIN_DURATION,
                                                            canvas_for_aspect, resolve_frames)


def _family(monkeypatch, family: str | None) -> None:
    monkeypatch.setattr("fastvideo.registry.get_preset_selection", lambda *_args, **_kwargs: (None, family))


@pytest.mark.parametrize("family", ["minimax_h3", "pika_h3_flex"])
def test_h3_discovery_includes_longer_clips_and_768p(monkeypatch, family):
    _family(monkeypatch, family)
    result = video_output_capabilities(SimpleNamespace(model_path="local-model"))
    assert result["duration_seconds"] == {"min": 5, "max": 15}
    assert result["recommended_frame_sizes"]["16:9"] == {"480p": [832, 480], "768p": [1344, 768]}
    assert result["num_frames_by_duration_seconds"]["10"] == 243
    assert result["num_frames_by_duration_seconds"]["15"] == 362
    for frames in result["num_frames_by_duration_seconds"].values():
        assert resolve_target_num_frames(frames) == frames
    for aspect, tiers in result["recommended_frame_sizes"].items():
        for tier, (width, height) in tiers.items():
            assert width % 32 == height % 32 == 0
            # Both families share the released tier geometry, so neither may advertise a
            # bucket the other would reject for the same aspect and resolution.
            assert (height, width) == canvas_for_aspect(aspect, tier)
            assert width * height <= MINIMAX_H3_MAX_PIXELS
    assert result["hardware_capacity_verified"] is False


def test_duration_grid_reuses_pika_frame_rules(monkeypatch):
    _family(monkeypatch, "pika_h3_flex")
    result = video_output_capabilities(SimpleNamespace(model_path="local-model"))
    assert result["fps"] == PIKA_FPS
    assert result["duration_seconds"] == {"min": int(PIKA_MIN_DURATION), "max": int(PIKA_MAX_DURATION)}
    for seconds, frames in result["num_frames_by_duration_seconds"].items():
        assert frames == resolve_frames(int(seconds) * PIKA_FPS)


def test_unknown_family_does_not_advertise_h3_options(monkeypatch):
    _family(monkeypatch, "wan")
    assert video_output_capabilities(SimpleNamespace(model_path="local-model")) is None


def test_unresolved_model_path_stays_offline(monkeypatch):
    downloads: list[tuple] = []

    def _record_download(*args, **kwargs):
        downloads.append((args, kwargs))
        return {}

    monkeypatch.setattr("fastvideo.registry.maybe_download_model_index", _record_download)
    assert video_output_capabilities(SimpleNamespace(model_path="unregistered-org/unregistered-model")) is None
    assert downloads == []


def test_model_info_exposes_output_constraints_without_generation(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastvideo.entrypoints.openai import common_api

    _family(monkeypatch, "minimax_h3")
    monkeypatch.setattr(common_api, "get_server_args", lambda: SimpleNamespace(model_path="local-model", lora_path=None))
    monkeypatch.setattr(common_api, "get_serving_engine", lambda: SimpleNamespace(lora_runtime=None))
    monkeypatch.setattr(common_api, "get_served_model_name", lambda: "custom-omni-alias")
    app = FastAPI()
    app.include_router(common_api.router)
    with TestClient(app) as client:
        response = client.get("/v1/model_info")
    assert response.status_code == 200
    assert response.json()["served_model_name"] == "custom-omni-alias"
    capabilities = response.json()["video_output_capabilities"]
    assert capabilities["duration_seconds"]["max"] == 15
    assert capabilities["recommended_frame_sizes"]["16:9"]["768p"] == [1344, 768]
