# SPDX-License-Identifier: Apache-2.0
"""The OpenAI request adapter's task selector, Ref2VA routing and geometry for FastH3 and Pika H3 Flex."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import RequestAdaptationError, build_generation_request
from fastvideo.pipelines.basic.minimax_h3.packing import resolve_canvas_size
from fastvideo.pipelines.basic.pika_h3_flex.buckets import canvas_for_aspect

FASTH3 = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
PIKA = "pika-labs/pika-h3-flex"
H3_REF2VA = "MiniMaxH3Ref2VAModularPipeline"
PIKA_REF2VA = "PikaH3FlexRef2VAModularPipeline"


def _args(model_path: str, pipeline: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_path=model_path, lora_path=None, lora_nickname="default", lora_strength=1.0,
                           override_pipeline_cls_name=pipeline)


def _adapt(tmp_path: Path, model_path: str, pipeline: str | None = None, **fields):
    fields.setdefault("aspect_ratio", "16:9")
    fields.setdefault("num_frames", 124)
    request = VideoGenerationRequest(prompt="a fox", **fields)
    return build_generation_request("video_gen_test", request, _args(model_path, pipeline), served_model_name="model",
                                    output_dir=str(tmp_path))


def _image(tmp_path: Path, name: str = "frame.png") -> str:
    path = tmp_path / name
    Image.new("RGB", (64, 36)).save(path)
    return str(path)


@pytest.mark.parametrize("task", ["i2va", "i2v"])
@pytest.mark.parametrize("model_path", [FASTH3, PIKA])
def test_i2va_takes_one_first_frame(tmp_path, model_path, task):
    adapted = _adapt(tmp_path, model_path, task=task, image_reference=[{"image_url": _image(tmp_path)}])
    assert adapted.inputs.image_path == str(tmp_path / "frame.png")
    assert adapted.inputs.last_image is None


@pytest.mark.parametrize("model_path", [FASTH3, PIKA])
def test_i2va_rejects_a_second_frame(tmp_path, model_path):
    images = [{"image_url": _image(tmp_path, "a.png")}, {"image_url": _image(tmp_path, "b.png")}]
    with pytest.raises(RequestAdaptationError, match="exactly one image reference"):
        _adapt(tmp_path, model_path, task="i2va", image_reference=images)


@pytest.mark.parametrize("model_path", [FASTH3, PIKA])
def test_ref2va_needs_the_ref2va_pipeline(tmp_path, model_path):
    with pytest.raises(RequestAdaptationError, match="requires its Ref2VA pipeline"):
        _adapt(tmp_path, model_path, task="ref2va", image_reference=[{"image_url": _image(tmp_path)}])


@pytest.mark.parametrize("task", [None, "t2va", "t2v"])
def test_omni_serves_text_only_requests_as_t2va(tmp_path, task):
    adapted = _adapt(tmp_path, FASTH3, H3_REF2VA, task=task)
    assert not adapted.inputs.references
    assert adapted.inputs.image_path is None


@pytest.mark.parametrize("task", ["i2va", "fl2va"])
def test_omni_rejects_keyframe_tasks(tmp_path, task):
    with pytest.raises(RequestAdaptationError, match="configured for MiniMax-H3 Ref2VA"):
        _adapt(tmp_path, FASTH3, H3_REF2VA, task=task, image_reference=[{"image_url": _image(tmp_path)}])


def test_pika_ref2va_routes_media_into_ordered_references(tmp_path):
    adapted = _adapt(tmp_path, PIKA, PIKA_REF2VA, task="ref2va", image_reference=[{"image_url": "/tmp/a.png"}],
                     video_reference=[{"video_url": "/tmp/b.mp4"}])
    assert [(ref.media_type, ref.source) for ref in adapted.inputs.references] == [("image", "/tmp/a.png"),
                                                                                   ("video", "/tmp/b.mp4")]


@pytest.mark.parametrize(("fields", "message"), [
    ({}, "requires at least one image or video reference"),
    ({"task": "t2va"}, "configured for Pika H3 Flex Ref2VA"),
    ({"image_reference": [{"image_url": "/tmp/a.png"}], "continuation_video": "/tmp/c.mkv"}, "continuation_video"),
])
def test_pika_ref2va_rejections(tmp_path, fields, message):
    with pytest.raises(RequestAdaptationError, match=message):
        _adapt(tmp_path, PIKA, PIKA_REF2VA, **fields)


def test_pika_fl2va_server_rejects_video_references(tmp_path):
    with pytest.raises(RequestAdaptationError, match="Pika H3 Flex video/audio references"):
        _adapt(tmp_path, PIKA, video_reference=[{"video_url": "/tmp/b.mp4"}])


@pytest.mark.parametrize(("fields", "message"), [
    ({"num_frames": 120}, "causal-VAE grid"),
    ({"num_frames": 73}, "generates 5-15 seconds"),
    ({"fps": 30}, "requires fps=24"),
])
def test_pika_geometry_fails_at_admission(tmp_path, fields, message):
    with pytest.raises(RequestAdaptationError, match=message):
        _adapt(tmp_path, PIKA, **fields)


def test_pika_reports_the_bucket_it_renders(tmp_path):
    adapted = _adapt(tmp_path, PIKA, aspect_ratio=None, size="1000x600")
    assert (adapted.sampling.height, adapted.sampling.width) == canvas_for_aspect("16:9", "p768")


@pytest.mark.parametrize(("short_edge", "tier"), [(480, "480p"), (768, "768p"), (None, "768p")])
def test_aspect_ratio_selects_the_480p_or_768p_canvas(tmp_path, short_edge, tier):
    h3 = _adapt(tmp_path, FASTH3, short_edge=short_edge)
    pika = _adapt(tmp_path, PIKA, short_edge=short_edge)
    edge, pixels = {"480p": (480, 832 * 480), "768p": (768, 768 * 1344)}[tier]
    assert (h3.sampling.height, h3.sampling.width) == resolve_canvas_size(16, 9, short_edge=edge, max_pixels=pixels)
    assert (pika.sampling.height, pika.sampling.width) == canvas_for_aspect("16:9", tier)


def test_other_short_edges_are_rejected(tmp_path):
    with pytest.raises(RequestAdaptationError, match="short_edge 480 or 768"):
        _adapt(tmp_path, FASTH3, short_edge=600)


def test_task_selector_is_limited_to_h3_families(tmp_path):
    with pytest.raises(RequestAdaptationError, match="MiniMax-H3 and Pika H3 Flex"):
        _adapt(tmp_path, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", task="t2va", aspect_ratio=None, num_frames=None)
