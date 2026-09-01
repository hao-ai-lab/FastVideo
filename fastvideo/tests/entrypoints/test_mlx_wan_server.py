# SPDX-License-Identifier: Apache-2.0
"""Wan MLX serving config, request validation, and generator dispatch.

Mirrors fastvideo/tests/entrypoints/test_openai_video_client.py's MLX H3
coverage pattern. MLXWanGenerator only imports the real `mlx` package inside
_load, which runs on a worker thread and is stubbed out here -- nothing in
this file needs Apple Silicon.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fastvideo.entrypoints.openai.mlx_wan_server import (
    MLXWanGenerator,
    create_mlx_wan_app,
    load_config,
    validate_wan_video_request,
)
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "examples/serving/mlx_wan21_1_3b.yaml"


def test_load_config_parses_the_checked_in_yaml() -> None:
    config = load_config(str(CONFIG_PATH))
    assert config.runtime == "mlx"
    assert config.generator.model_root == "./FastMetal-1.3B-QAD"
    assert config.generator.mlx_checkpoint == "./FastMetal-1.3B-QAD"
    assert config.server.served_model_name == "fastwan21-1.3b-mlx"


def test_generator_config_rejects_unknown_fields() -> None:
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    with pytest.raises(ValidationError):
        MLXWanGeneratorConfig(model_root="x", mlx_checkpoint="y", extra_field="not allowed")


@pytest.mark.parametrize("field,value", [
    ("negative_prompt", "bad"),
    ("task", "t2va"),
    ("guidance_scale", 2.0),
    ("num_inference_steps", 10),
    ("seed", -1),
])
def test_validate_rejects_unsupported_request_fields(field, value) -> None:
    with pytest.raises(ValueError):
        validate_wan_video_request(VideoGenerationRequest(prompt="a cat", **{field: value}))


def test_validate_accepts_the_recipe_shape() -> None:
    validate_wan_video_request(VideoGenerationRequest(prompt="a cat", num_inference_steps=3, guidance_scale=1.0))


def test_create_app_builds_without_loading_the_mlx_pipeline() -> None:
    """generator_factory is lazy -- building the app must not import mlx.core."""
    config = load_config(str(CONFIG_PATH))
    app = create_mlx_wan_app(config)
    assert app.state.served_model_name == "fastwan21-1.3b-mlx"


def test_create_app_rejects_unsupported_default_request_fields(tmp_path) -> None:
    bad_config_path = tmp_path / "bad.yaml"
    bad_config_path.write_text("""
runtime: mlx
generator:
  model_root: ./FastMetal-1.3B-QAD
  mlx_checkpoint: ./FastMetal-1.3B-QAD
server:
  served_model_name: bad
default_request:
  negative_prompt: not-supported
  sampling:
    height: 480
    width: 832
    num_frames: 81
    fps: 16
""")
    with pytest.raises(ValueError, match="unsupported fields"):
        create_mlx_wan_app(load_config(str(bad_config_path)))


def test_generator_dispatches_request_to_pipeline_on_worker_thread() -> None:
    """MLXWanGenerator must call MLXWanPipeline.generate with the request's
    sampling fields, entirely off the calling thread."""
    calls: dict[str, object] = {}

    class _StubPipeline:

        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def generate(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls["generate_kwargs"] = kwargs
            return SimpleNamespace(video_path="outputs/stub.mp4")

    stub_pipeline_module = SimpleNamespace(MLXWanPipeline=_StubPipeline)
    stub_memory_module = SimpleNamespace(cleanup_mlx=lambda: calls.setdefault("cleaned_up", True))
    generator_config = SimpleNamespace(model_root="./FastMetal-1.3B-QAD", mlx_checkpoint="./FastMetal-1.3B-QAD")

    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict("sys.modules", {
             "fastvideo.mlx_runtime.wan_pipeline": stub_pipeline_module,
             "fastvideo.mlx_runtime.memory": stub_memory_module,
         }):
        generator = MLXWanGenerator(generator_config)
        try:
            request = SimpleNamespace(
                prompt="a red panda",
                output=SimpleNamespace(output_path="outputs/out.mp4"),
                sampling=SimpleNamespace(width=832, height=480, num_frames=81, seed=1024, fps=16),
            )
            result = generator.generate(request)
        finally:
            generator.shutdown()

    assert result["video_path"] == "outputs/stub.mp4"
    assert calls["prompt"] == "a red panda"
    assert calls["generate_kwargs"] == {"output_path": "outputs/out.mp4", "width": 832, "height": 480,
                                        "num_frames": 81, "seed": 1024, "fps": 16}
    assert calls["cleaned_up"] is True


def test_generator_rejects_non_apple_silicon() -> None:
    with patch("platform.system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            MLXWanGenerator(SimpleNamespace(model_root="x", mlx_checkpoint="y"))
