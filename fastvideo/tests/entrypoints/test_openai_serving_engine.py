# SPDX-License-Identifier: Apache-2.0
"""CPU-light contracts for the shared OpenAI serving engine and adapter."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastvideo.api.parser import parse_config
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.openai.media_policy import MediaPolicy
from fastvideo.entrypoints.openai.protocol import UrlAudioReference, VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import (
    RequestAdaptationError,
    build_generation_request,
    discard_materialized_media,
    prepare_reference_media,
    validate_served_model_name,
)
from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine
from fastvideo.entrypoints.openai.stores import VIDEO_STORE
from fastvideo.entrypoints.openai.video_api import _media_type_for_path


class _BlockingGenerator:

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.shutdown_called = False

    def generate(self, request):
        self.started.set()
        self.release.wait(timeout=5)
        return request

    def shutdown(self) -> None:
        self.shutdown_called = True


class _WaveGenerator:

    def __init__(self) -> None:
        self.last_output: Path | None = None

    def generate(self, request):
        output = Path(request.output.output_path)
        self.last_output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fastvideo-test")
        return SimpleNamespace(
            video_path=str(output),
            generation_time=0.01,
            peak_memory_mb=12.5,
            logging_info=None,
        )

    def shutdown(self) -> None:
        return None


class _FileGenerator:

    def __init__(self) -> None:
        self.last_output: Path | None = None

    def generate(self, request):
        output = Path(request.output.output_path)
        self.last_output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x00\x00\x00\x18ftypmp42fastvideo-test")
        return SimpleNamespace(
            video_path=str(output),
            generation_time=0.01,
            peak_memory_mb=12.5,
            logging_info=None,
        )

    def shutdown(self) -> None:
        return None


def _args(model_path: str, **overrides):
    values = {
        "model_path": model_path,
        "lora_path": None,
        "lora_nickname": "default",
        "lora_strength": 1.0,
        "override_pipeline_cls_name": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_engine_cancellation_keeps_pipeline_locked() -> None:
    asyncio.run(_assert_engine_cancellation_keeps_pipeline_locked())


async def _assert_engine_cancellation_keeps_pipeline_locked() -> None:
    generator = _BlockingGenerator()
    engine = OpenAIServingEngine(generator)  # type: ignore[arg-type]

    first = asyncio.create_task(engine.generate("first"))  # type: ignore[arg-type]
    await asyncio.to_thread(generator.started.wait, 2)
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()

    second_started = threading.Event()

    def second_call():
        second_started.set()
        return "second"

    second = asyncio.create_task(engine.run_serialized(second_call))
    await asyncio.sleep(0.05)
    assert not second_started.is_set()

    generator.release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == "second"

    await engine.shutdown()
    assert generator.shutdown_called


def test_protocol_rejects_unknown_and_client_output_fields() -> None:
    with pytest.raises(Exception, match="prompt_path"):
        VideoGenerationRequest(prompt="a fox", prompt_path="/srv/prompts.txt")
    with pytest.raises(Exception, match="output_path"):
        VideoGenerationRequest(prompt="a fox", output_path="/srv/keep.mp4")


def test_validate_served_model_name_accepts_omitted_or_matching_name() -> None:
    args = _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    validate_served_model_name(None, args, "wan")
    validate_served_model_name("wan", args, "wan")


def test_validate_served_model_name_rejects_mismatch() -> None:
    with pytest.raises(RequestAdaptationError, match="this server provides wan"):
        validate_served_model_name("other-model", _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"), "wan")


def test_validate_served_model_name_uses_lora_nickname() -> None:
    args = _args(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        lora_path="/models/startup.safetensors",
        lora_nickname="fast",
    )
    validate_served_model_name("fast", args, "wan")
    with pytest.raises(RequestAdaptationError, match="this server provides fast"):
        validate_served_model_name("wan", args, "wan")


def test_request_adapter_restricts_extra_params(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="a fox", extra_params={"prompt_path": "/srv/prompts.txt"})
    with pytest.raises(RequestAdaptationError, match="Unsupported extra_params fields: prompt_path"):
        build_generation_request(
            "video_gen_test",
            request,
            _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            served_model_name="wan",
            output_dir=str(tmp_path),
        )


def test_request_adapter_owns_output_path(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="a fox", size="64x64", num_frames=1)
    default_request = parse_config(GenerationRequest, {"output": {"output_path": "/srv/untrusted-default.mp4"}})
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        served_model_name="wan",
        output_dir=str(tmp_path),
        default_request=default_request,
    )
    assert adapted.output.output_path == str(tmp_path / "videos" / "video_gen_test.mp4")


def test_audio_request_maps_seconds_to_duration_and_wav_output(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="rain on a metal roof", seconds=4, audio_start_in_s=0)
    adapted = build_generation_request(
        "audio_gen_test",
        request,
        _args("FastVideo/stable-audio-open-1.0-Diffusers", workload_type=SimpleNamespace(value="t2a")),
        served_model_name="stable-audio",
        output_dir=str(tmp_path),
    )
    assert adapted.sampling.audio_start_in_s == 0.0
    assert adapted.sampling.audio_end_in_s == 4.0
    assert adapted.output.output_path == str(tmp_path / "audio" / "audio_gen_test.wav")


def test_audio_request_rejects_video_geometry(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="rain", size="512x512")
    with pytest.raises(RequestAdaptationError, match="does not accept size"):
        build_generation_request(
            "audio_gen_test",
            request,
            _args("FastVideo/stable-audio-open-1.0-Diffusers", workload_type=SimpleNamespace(value="t2a")),
            served_model_name="stable-audio",
            output_dir=str(tmp_path),
        )


def test_audio_seconds_override_operator_duration(tmp_path: Path) -> None:
    default_request = parse_config(
        GenerationRequest,
        {"sampling": {
            "audio_start_in_s": 1.0,
            "audio_end_in_s": 10.0,
        }},
    )
    request = VideoGenerationRequest(prompt="rain", seconds=4)
    adapted = build_generation_request(
        "audio_gen_test",
        request,
        _args("FastVideo/stable-audio-open-1.0-Diffusers", workload_type=SimpleNamespace(value="t2a")),
        served_model_name="stable-audio",
        output_dir=str(tmp_path),
        default_request=default_request,
    )
    assert adapted.sampling.audio_start_in_s == 1.0
    assert adapted.sampling.audio_end_in_s == 5.0


def test_wav_media_type_is_detected_from_output_path() -> None:
    assert _media_type_for_path("/tmp/request.wav") == "audio/wav"
    assert _media_type_for_path("/tmp/request.MP4") == "video/mp4"


def test_missing_video_reference_fails_during_admission(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="waves", video_reference={"video_url": str(tmp_path / "missing.mp4")})
    with pytest.raises(RequestAdaptationError, match="does not exist"):
        asyncio.run(
            prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy(roots=(str(tmp_path), ))))


def test_local_media_paths_need_a_configured_root(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    request = VideoGenerationRequest(prompt="waves", video_reference={"video_url": str(clip)})
    with pytest.raises(RequestAdaptationError, match="server.media_roots"):
        asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))


def _data_url(media_type: str, payload: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"


def _ref(kind: str, url: str) -> dict[str, str]:
    return {f"{kind}_url": url}


def _uploads(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "uploads").glob("*"))


def test_inline_video_and_audio_references_are_decoded_in_order(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="waves",
        video_reference=[
            _ref("video", _data_url("video/mp4", b"first-clip")),
            _ref("video", _data_url("video/quicktime", b"second-clip")),
        ],
        audio_reference=[
            _ref("audio", _data_url("audio/mpeg", b"first-track")),
            _ref("audio", _data_url("audio/wav", b"second-track")),
        ],
    )
    created = asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
    assert isinstance(request.video_reference, list) and isinstance(request.audio_reference, list)
    videos = [reference.video_url for reference in request.video_reference]
    audios = [reference.audio_url for reference in request.audio_reference]
    assert created == videos + audios
    contents = [Path(path).read_bytes() for path in created]
    assert contents == [b"first-clip", b"second-clip", b"first-track", b"second-track"]
    assert [Path(path).suffix for path in created] == [".mp4", ".mov", ".mp3", ".wav"]
    assert {Path(path).parent for path in created} == {tmp_path / "uploads"}
    discard_materialized_media(created)
    assert not _uploads(tmp_path)


def test_inline_reference_payload_may_be_wrapped_and_percent_escaped(tmp_path: Path) -> None:
    track = b"wrapped-track" * 8
    encoded = base64.b64encode(track).decode().replace("=", "%3D")
    wrapped = "\n".join(encoded[start:start + 16] for start in range(0, len(encoded), 16))
    request = VideoGenerationRequest(prompt="waves", audio_reference={"audio_url": f"data:audio/flac;base64,{wrapped}"})
    created = asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
    assert Path(created[0]).read_bytes() == track
    assert created[0].endswith(".flac")


@pytest.mark.parametrize(
    ("kind", "url", "message"),
    [
        ("video", "data:video/mp4;base64,not*base64", "not valid base64"),
        ("audio", "data:audio/mpeg;base64,", "payload is empty"),
        ("video", "data:video/mp4,raw-bytes", "must be base64-encoded"),
        ("video", _data_url("audio/mpeg", b"track"), "need a data:video/"),
        ("audio", _data_url("video/mp4", b"clip"), "need a data:audio/"),
        ("audio", _data_url("application/octet-stream", b"track"), "need a data:audio/"),
        ("image", _data_url("video/mp4", b"clip"), "need a data:image/"),
    ],
)
def test_invalid_inline_references_are_refused(tmp_path: Path, kind: str, url: str, message: str) -> None:
    request = VideoGenerationRequest(prompt="waves", **{f"{kind}_reference": _ref(kind, url)})
    with pytest.raises(RequestAdaptationError, match=message):
        asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
    assert not _uploads(tmp_path)


def test_inline_reference_over_the_media_limit_is_refused(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="waves", audio_reference={"audio_url": _data_url("audio/wav", b"12345")})
    with pytest.raises(RequestAdaptationError, match="4-byte limit"):
        asyncio.run(prepare_reference_media("request", request, str(tmp_path), max_bytes=4, policy=MediaPolicy()))
    assert not _uploads(tmp_path)
    at_limit = VideoGenerationRequest(prompt="waves", video_reference={"video_url": _data_url("video/mp4", b"1234")})
    created = asyncio.run(prepare_reference_media("request", at_limit, str(tmp_path), max_bytes=4,
                                                  policy=MediaPolicy()))
    assert Path(created[0]).read_bytes() == b"1234"


def test_decoded_references_are_removed_when_a_later_reference_fails(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="waves",
        video_reference={"video_url": _data_url("video/mp4", b"clip")},
        audio_reference={"audio_url": "data:audio/mpeg;base64,%%%"},
    )
    with pytest.raises(RequestAdaptationError, match="not valid base64"):
        asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
    assert not _uploads(tmp_path)


def test_local_audio_paths_need_a_configured_root(tmp_path: Path) -> None:
    track = tmp_path / "track.wav"
    track.write_bytes(b"x")
    request = VideoGenerationRequest(prompt="waves", audio_reference={"audio_url": str(track)})
    with pytest.raises(RequestAdaptationError, match="server.media_roots"):
        asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
    created = asyncio.run(
        prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy(roots=(str(tmp_path), ))))
    assert created == []
    assert isinstance(request.audio_reference, UrlAudioReference)
    assert request.audio_reference.audio_url == os.path.realpath(track)


def test_mmaudio_v2a_requires_video_reference(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="waves")
    with pytest.raises(RequestAdaptationError, match="requires video_reference"):
        build_generation_request(
            "audio_gen_test",
            request,
            _args("FastVideo/MMAudio-large-44k-v2-Diffusers", workload_type=SimpleNamespace(value="v2a")),
            served_model_name="mmaudio",
            output_dir=str(tmp_path),
        )


def test_missing_reference_fails_during_admission(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="a fox", input_reference=str(tmp_path / "missing.png"))
    with pytest.raises(RequestAdaptationError, match="does not exist"):
        asyncio.run(
            prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy(roots=(str(tmp_path), ))))


def test_request_adapter_resolves_vllm_nested_params(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="a fox",
        video_params={
            "width": 832,
            "height": 480,
            "fps": 30
        },
        seconds="2",
        seed=7,
    )
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        served_model_name="wan",
        output_dir=str(tmp_path),
    )

    assert adapted.sampling.width == 832
    assert adapted.sampling.height == 480
    assert adapted.sampling.fps == 30
    assert adapted.sampling.num_frames == 60
    assert adapted.sampling.seed == 7
    assert adapted.output.output_path.endswith("video_gen_test.mp4")


def test_request_adapter_accepts_matching_startup_lora(tmp_path: Path) -> None:
    adapter = str(tmp_path / "adapter.safetensors")
    args = _args(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        lora_path=adapter,
        lora_nickname="fast",
        lora_strength=0.75,
    )
    request = VideoGenerationRequest(
        prompt="a fox",
        model="fast",
        lora={
            "name": "fast",
            "path": adapter,
            "scale": 0.75
        },
    )

    build_generation_request(
        "video_gen_test",
        request,
        args,
        served_model_name="wan",
        output_dir=str(tmp_path),
    )


def test_request_adapter_rejects_runtime_lora_swap(tmp_path: Path) -> None:
    args = _args(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        lora_path="/models/startup.safetensors",
        lora_nickname="fast",
    )
    request = VideoGenerationRequest(prompt="a fox", lora={"path": "/models/other.safetensors"})

    with pytest.raises(RequestAdaptationError, match="does not match the startup adapter"):
        build_generation_request(
            "video_gen_test",
            request,
            args,
            served_model_name="wan",
            output_dir=str(tmp_path),
        )


def test_fasth3_request_uses_the_general_adapter(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="a fox",
        task="t2va",
        aspect_ratio="16:9",
        num_frames=124,
        num_inference_steps=5,
        guidance_scale=1.0,
    )
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"),
        served_model_name="fasth3",
        output_dir=str(tmp_path),
    )

    assert adapted.sampling.width == 1344
    assert adapted.sampling.height == 768
    assert adapted.sampling.num_frames == 124


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({
            "aspect_ratio": "16:9",
            "num_frames": 22
        }, "generates 5-15 seconds"),
        ({
            "size": "100x100",
            "num_frames": 124
        }, "positive multiples of 32"),
        ({
            "aspect_ratio": "1:9",
            "num_frames": 124
        }, "supports aspect ratios"),
        ({
            "aspect_ratio": "16:9",
            "num_frames": 120
        }, "causal-VAE grid"),
    ],
)
def test_fasth3_invalid_geometry_fails_at_admission(tmp_path: Path, overrides, message) -> None:
    request = VideoGenerationRequest(prompt="a fox", **overrides)
    with pytest.raises(RequestAdaptationError, match=message):
        build_generation_request(
            "video_gen_test",
            request,
            _args("FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"),
            served_model_name="fasth3",
            output_dir=str(tmp_path),
        )


@pytest.mark.parametrize(("seconds", "expected_frames"), [("5", 124), ("15", 362)])
def test_fasth3_seconds_align_to_causal_vae_grid(tmp_path: Path, seconds: str, expected_frames: int) -> None:
    request = VideoGenerationRequest(prompt="a fox", aspect_ratio="16:9", seconds=seconds)
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"),
        served_model_name="fasth3",
        output_dir=str(tmp_path),
    )
    assert adapted.sampling.num_frames == expected_frames


def test_fasth3_ref2va_limits_references_at_admission(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="a fox",
        task="ref2va",
        aspect_ratio="16:9",
        num_frames=124,
        image_reference=[{
            "image_url": f"/tmp/reference-{index}.png"
        } for index in range(10)],
    )
    with pytest.raises(RequestAdaptationError, match="at most 9 image references"):
        build_generation_request(
            "video_gen_test",
            request,
            _args(
                "MiniMaxAI/MiniMax-H3",
                override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline",
            ),
            served_model_name="minimax-h3",
            output_dir=str(tmp_path),
        )


def test_aspect_ratio_overrides_operator_dimensions(tmp_path: Path) -> None:
    default_request = parse_config(
        GenerationRequest,
        {"sampling": {
            "width": 1344,
            "height": 768,
            "num_frames": 124,
        }},
    )
    request = VideoGenerationRequest(prompt="a fox", aspect_ratio="9:16")
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"),
        served_model_name="fasth3",
        output_dir=str(tmp_path),
        default_request=default_request,
    )
    assert adapted.sampling.width == 768
    assert adapted.sampling.height == 1344


def test_explicit_null_uses_seconds_and_operator_fps(tmp_path: Path) -> None:
    default_request = parse_config(
        GenerationRequest,
        {"sampling": {
            "fps": 30,
            "num_frames": 99,
        }},
    )
    request = VideoGenerationRequest(prompt="a fox", seconds=2, fps=None, num_frames=None)
    adapted = build_generation_request(
        "video_gen_test",
        request,
        _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        served_model_name="wan",
        output_dir=str(tmp_path),
        default_request=default_request,
    )
    assert adapted.sampling.fps == 30
    assert adapted.sampling.num_frames == 60


def test_unsupported_vllm_postprocessing_fails_at_admission(tmp_path: Path) -> None:
    request = VideoGenerationRequest(prompt="a fox", enable_frame_interpolation=True)

    with pytest.raises(RequestAdaptationError, match="frame_interpolation"):
        build_generation_request(
            "video_gen_test",
            request,
            _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            served_model_name="wan",
            output_dir=str(tmp_path),
        )


def test_audio_jobs_publish_wav_media_type_and_seconds(tmp_path: Path) -> None:
    from fastvideo.entrypoints.openai import state
    from fastvideo.entrypoints.openai.video_api import router

    generator = _WaveGenerator()
    engine = OpenAIServingEngine(generator)  # type: ignore[arg-type,unused-ignore]
    args = _args("FastVideo/stable-audio-open-1.0-Diffusers", workload_type=SimpleNamespace(value="t2a"))
    state.set_state(
        generator,  # type: ignore[arg-type]
        engine,
        args,  # type: ignore[arg-type]
        str(tmp_path),
        served_model_name="stable-audio",
    )
    asyncio.run(VIDEO_STORE.clear())
    app = FastAPI()
    app.include_router(router)

    try:
        with TestClient(app) as client:
            created = client.post(
                "/v1/videos",
                json={
                    "model": "stable-audio",
                    "prompt": "rain",
                    "seconds": 3
                },
            )
            assert created.status_code == 200
            audio_id = created.json()["id"]

            for _ in range(50):
                detail = client.get(f"/v1/videos/{audio_id}")
                if detail.json()["status"] == "completed":
                    break
                time.sleep(0.01)
            body = detail.json()
            assert body["status"] == "completed"
            assert body["media_type"] == "audio/wav"
            assert body["seconds"] == "3"
            assert body["size"] is None
            assert body["file_name"].endswith(".wav")

            content = client.get(f"/v1/videos/{audio_id}/content")
            assert content.status_code == 200
            assert content.headers["content-type"] == "audio/wav"
            assert content.content.startswith(b"RIFF")

            sync = client.post("/v1/videos/sync", json={"prompt": "rain", "seconds": 2})
            assert sync.status_code == 200
            assert sync.headers["content-type"] == "audio/wav"
            assert sync.content.startswith(b"RIFF")
            assert generator.last_output is not None
            assert not generator.last_output.exists()
    finally:
        asyncio.run(VIDEO_STORE.clear())
        state.clear_state()


class _ReferenceGenerator(_FileGenerator):

    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, bytes]] = []

    def generate(self, request):
        self.references = [(reference.media_type, Path(reference.source).read_bytes())
                           for reference in request.inputs.references]
        return super().generate(request)


def test_ref2va_route_accepts_inline_references_and_removes_them(tmp_path: Path) -> None:
    from fastvideo.entrypoints.openai import state
    from fastvideo.entrypoints.openai.video_api import router

    generator = _ReferenceGenerator()
    engine = OpenAIServingEngine(generator)  # type: ignore[arg-type]
    args = _args("MiniMaxAI/MiniMax-H3", override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline")
    state.set_state(
        generator,  # type: ignore[arg-type]
        engine,
        args,  # type: ignore[arg-type]
        str(tmp_path),
        served_model_name="minimax-h3",
    )
    asyncio.run(VIDEO_STORE.clear())
    app = FastAPI()
    app.include_router(router)
    body = dict(
        model="minimax-h3",
        prompt="a fox sings along",
        task="ref2va",
        aspect_ratio="16:9",
        num_frames=124,
        video_reference=[_ref("video", _data_url("video/mp4", b"clip-with-sound"))],
        audio_reference=[
            _ref("audio", _data_url("audio/mpeg", b"first-voice")),
            _ref("audio", _data_url("audio/wav", b"second-voice")),
        ],
    )

    try:
        with TestClient(app) as client:
            sync = client.post("/v1/videos/sync", json=body)
            assert sync.status_code == 200, sync.text
            assert generator.references == [
                ("video", b"clip-with-sound"),
                ("audio", b"first-voice"),
                ("audio", b"second-voice"),
            ]
            assert not _uploads(tmp_path)

            mismatched = {**body, "audio_reference": [_ref("audio", _data_url("video/mp4", b"clip"))]}
            refused = client.post("/v1/videos", json=mismatched)
            assert refused.status_code == 400
            assert "need a data:audio/" in refused.text
            assert not _uploads(tmp_path)
    finally:
        asyncio.run(VIDEO_STORE.clear())
        state.clear_state()


def test_video_routes_cover_async_sync_list_content_and_delete(tmp_path: Path) -> None:
    from fastvideo.entrypoints.openai import state
    from fastvideo.entrypoints.openai.video_api import router

    generator = _FileGenerator()
    engine = OpenAIServingEngine(generator)  # type: ignore[arg-type]
    args = _args("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    state.set_state(
        generator,  # type: ignore[arg-type]
        engine,
        args,  # type: ignore[arg-type]
        str(tmp_path),
        served_model_name="wan-test",
    )
    asyncio.run(VIDEO_STORE.clear())
    app = FastAPI()
    app.include_router(router)

    try:
        with TestClient(app) as client:
            created = client.post(
                "/v1/videos",
                json={
                    "model": "wan-test",
                    "prompt": "a fox",
                    "size": "64x64",
                    "num_frames": 1,
                },
            )
            assert created.status_code == 200
            video_id = created.json()["id"]

            for _ in range(50):
                detail = client.get(f"/v1/videos/{video_id}")
                if detail.json()["status"] == "completed":
                    break
                time.sleep(0.01)
            assert detail.status_code == 200
            assert detail.json()["status"] == "completed"

            listing = client.get("/v1/videos", params={"limit": 1})
            assert listing.status_code == 200
            assert listing.json()["data"][0]["id"] == video_id

            content = client.get(f"/v1/videos/{video_id}/content")
            assert content.status_code == 200
            assert content.content[4:8] == b"ftyp"

            deleted = client.delete(f"/v1/videos/{video_id}")
            assert deleted.status_code == 200
            assert deleted.json() == {
                "id": video_id,
                "deleted": True,
                "object": "video.deleted",
            }

            sync = client.post(
                "/v1/videos/sync",
                json={
                    "prompt": "a fox",
                    "size": "64x64",
                    "num_frames": 1,
                },
            )
            assert sync.status_code == 200
            assert sync.headers["x-model"] == "wan-test"
            assert sync.content[4:8] == b"ftyp"
            assert generator.last_output is not None
            assert not generator.last_output.exists()
    finally:
        asyncio.run(VIDEO_STORE.clear())
        state.clear_state()
