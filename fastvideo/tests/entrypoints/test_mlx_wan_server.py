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
CONFIG_PATH_14B = ROOT / "examples/serving/mlx_wan21_14b.yaml"
CONFIG_PATH_5B = ROOT / "examples/serving/mlx_wan22_5b.yaml"


def test_load_config_parses_the_checked_in_yaml() -> None:
    config = load_config(str(CONFIG_PATH))
    assert config.runtime == "mlx"
    assert config.generator.model_path == "FastVideo/FastMetal-1.3B-QAD"
    assert config.generator.model_root == "./FastMetal-1.3B-QAD"
    assert config.generator.mlx_checkpoint == "./FastMetal-1.3B-QAD"
    assert config.server.served_model_name == "fastwan21-1.3b-mlx"


def test_load_config_parses_the_14b_yaml() -> None:
    config = load_config(str(CONFIG_PATH_14B))
    assert config.generator.model_path == "FastVideo/FastMetal-14B-QAD"
    assert config.generator.model_root == "./FastMetal-14B-QAD"
    assert config.server.served_model_name == "fastwan21-14b-mlx"


def test_create_app_uses_the_14b_config_model_path_not_the_1_3b_default() -> None:
    """Regression guard: the served model must come from the config, not a
    module-level constant left over from when only 1.3B was supported."""
    config = load_config(str(CONFIG_PATH_14B))
    app = create_mlx_wan_app(config)
    assert app.state.fastvideo_args.model_path == "FastVideo/FastMetal-14B-QAD"


def test_load_config_parses_the_5b_yaml() -> None:
    config = load_config(str(CONFIG_PATH_5B))
    assert config.generator.model_path == "FastVideo/FastMetal-5B-QAD"
    assert config.server.served_model_name == "fastwan22-5b-mlx"
    assert config.default_request["sampling"]["height"] == 704
    assert config.default_request["sampling"]["width"] == 1280


def test_generator_config_rejects_unknown_fields() -> None:
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    with pytest.raises(ValidationError):
        MLXWanGeneratorConfig(model_path="FastVideo/FastMetal-1.3B-QAD", model_root="x", mlx_checkpoint="y",
                              extra_field="not allowed")


def test_generator_config_requires_a_model_path() -> None:
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    with pytest.raises(ValidationError):
        MLXWanGeneratorConfig(model_root="x", mlx_checkpoint="y")


@pytest.mark.parametrize("model_path", ["FastVideo/FastMetal-1.3B-QAD", "FastVideo/FastMetal-14B-QAD"])
def test_generator_config_accepts_the_two_wan21_sizes(model_path) -> None:
    """1.3B and 14B share MLXWanPipeline's Wan2.1 architecture."""
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    config = MLXWanGeneratorConfig(model_path=model_path, model_root="x", mlx_checkpoint="y")
    assert config.model_path == model_path


def test_generator_config_accepts_the_wan22_5b_model() -> None:
    """FastMetal 5B is Wan2.2-TI2V; MLXWanGenerator routes it to MLXWan22Pipeline
    (see test_generator_loads_the_pipeline_class_matching_model_path)."""
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    config = MLXWanGeneratorConfig(model_path="FastVideo/FastMetal-5B-QAD", model_root="x", mlx_checkpoint="y")
    assert config.model_path == "FastVideo/FastMetal-5B-QAD"


def test_generator_config_rejects_an_unregistered_model() -> None:
    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanGeneratorConfig
    with pytest.raises(ValidationError):
        MLXWanGeneratorConfig(model_path="FastVideo/SomeOtherModel", model_root="x", mlx_checkpoint="y")


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


def test_validate_accepts_explicit_task_t2v() -> None:
    validate_wan_video_request(VideoGenerationRequest(prompt="a cat", task="t2v"))


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_validate_accepts_seed_boundary_values(seed) -> None:
    validate_wan_video_request(VideoGenerationRequest(prompt="a cat", seed=seed))


def test_validate_rejects_seed_above_the_uint32_range() -> None:
    with pytest.raises(ValueError):
        validate_wan_video_request(VideoGenerationRequest(prompt="a cat", seed=2**32))


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
  model_path: FastVideo/FastMetal-1.3B-QAD
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


def test_create_app_rejects_a_default_request_missing_required_sampling_fields(tmp_path) -> None:
    incomplete_config_path = tmp_path / "incomplete.yaml"
    incomplete_config_path.write_text("""
runtime: mlx
generator:
  model_path: FastVideo/FastMetal-1.3B-QAD
  model_root: ./FastMetal-1.3B-QAD
  mlx_checkpoint: ./FastMetal-1.3B-QAD
server:
  served_model_name: incomplete
default_request:
  sampling:
    height: 480
    width: 832
""")
    with pytest.raises(ValueError, match="must set"):
        create_mlx_wan_app(load_config(str(incomplete_config_path)))


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

    generator_config = SimpleNamespace(model_path="FastVideo/FastMetal-1.3B-QAD", model_root="./FastMetal-1.3B-QAD",
                                       mlx_checkpoint="./FastMetal-1.3B-QAD")

    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("fastvideo.mlx_runtime.wan_pipeline.MLXWanPipeline", _StubPipeline), \
         patch("fastvideo.mlx_runtime.memory.cleanup_mlx", lambda: calls.setdefault("cleaned_up", True)):
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


@pytest.mark.parametrize("model_path,expected_class_name", [
    ("FastVideo/FastMetal-1.3B-QAD", "MLXWanPipeline"),
    ("FastVideo/FastMetal-14B-QAD", "MLXWanPipeline"),
    ("FastVideo/FastMetal-5B-QAD", "MLXWan22Pipeline"),
])
def test_generator_loads_the_pipeline_class_matching_model_path(model_path, expected_class_name) -> None:
    """1.3B/14B must load MLXWanPipeline (Wan2.1); 5B must load MLXWan22Pipeline
    (Wan2.2-TI2V) -- these are different architectures, not interchangeable."""
    loaded: dict[str, object] = {}

    class _StubWan21Pipeline:

        def __init__(self, **kwargs):
            loaded["class_name"] = "MLXWanPipeline"

    class _StubWan22Pipeline:

        def __init__(self, **kwargs):
            loaded["class_name"] = "MLXWan22Pipeline"

    generator_config = SimpleNamespace(model_path=model_path, model_root="x", mlx_checkpoint="y")

    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("fastvideo.mlx_runtime.wan_pipeline.MLXWanPipeline", _StubWan21Pipeline), \
         patch("fastvideo.mlx_runtime.wan_pipeline.MLXWan22Pipeline", _StubWan22Pipeline), \
         patch("fastvideo.mlx_runtime.memory.cleanup_mlx", lambda: None):
        generator = MLXWanGenerator(generator_config)
        generator.shutdown()

    assert loaded["class_name"] == expected_class_name


def test_generator_rejects_non_apple_silicon() -> None:
    with patch("platform.system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            MLXWanGenerator(SimpleNamespace(model_root="x", mlx_checkpoint="y"))


def test_generator_rejects_intel_mac() -> None:
    """Darwin alone isn't enough -- x86_64 Macs have no Metal MLX support."""
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="x86_64"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            MLXWanGenerator(SimpleNamespace(model_root="x", mlx_checkpoint="y"))


def test_generator_rejects_missing_ffmpeg() -> None:
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            MLXWanGenerator(SimpleNamespace(model_root="x", mlx_checkpoint="y"))


def test_generator_shuts_down_its_worker_thread_when_load_fails() -> None:
    """__init__ must not leak a running thread pool if the pipeline fails to load."""
    import threading

    class _BrokenPipeline:

        def __init__(self, **_kwargs):
            raise RuntimeError("checkpoint is corrupt")

    threads_before = {thread.name for thread in threading.enumerate()}

    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("fastvideo.mlx_runtime.wan_pipeline.MLXWanPipeline", _BrokenPipeline):
        with pytest.raises(RuntimeError, match="checkpoint is corrupt"):
            MLXWanGenerator(SimpleNamespace(model_path="FastVideo/FastMetal-1.3B-QAD", model_root="x",
                                            mlx_checkpoint="y"))

    # The worker thread pool must not outlive the failed __init__ call.
    threads_after = {thread.name for thread in threading.enumerate()}
    assert not any("wan-mlx" in name for name in threads_after - threads_before)
