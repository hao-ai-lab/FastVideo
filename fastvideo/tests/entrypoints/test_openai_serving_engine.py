# SPDX-License-Identifier: Apache-2.0
"""CPU-light contracts for the shared OpenAI serving engine and adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import (
    RequestAdaptationError,
    build_generation_request,
)
from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine


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


@pytest.mark.asyncio
async def test_engine_cancellation_keeps_pipeline_locked() -> None:
    generator = _BlockingGenerator()
    engine = OpenAIServingEngine(generator)  # type: ignore[arg-type]

    first = asyncio.create_task(engine.generate("first"))  # type: ignore[arg-type]
    await asyncio.to_thread(generator.started.wait, 2)
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


def test_request_adapter_resolves_vllm_nested_params(tmp_path: Path) -> None:
    request = VideoGenerationRequest(
        prompt="a fox",
        video_params={"width": 832, "height": 480, "fps": 30},
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
        lora={"name": "fast", "path": adapter, "scale": 0.75},
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
