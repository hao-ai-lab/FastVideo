# SPDX-License-Identifier: Apache-2.0
"""Declaration checks for family serving baselines; no model weights are loaded."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
import yaml

from fastvideo.api.compat import request_to_sampling_param
from fastvideo.api.flux import FluxSamplingParam
from fastvideo.api.matrixgame2 import MatrixGame2SamplingParam
from fastvideo.api.parser import parse_config
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import GenerationRequest, ServeConfig
from fastvideo.entrypoints.openai.media_policy import MediaPolicy
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import (
    RequestAdaptationError,
    build_generation_request,
    discard_materialized_media,
    prepare_reference_media,
)
from fastvideo.pipelines.basic.hunyuan15.presets import HUNYUAN15_I2V_480P_DISTILLED
from fastvideo.pipelines.basic.pan2.presets import PAN2_T2V

ROOT = Path(__file__).resolve().parents[3]

BASELINES = {
    "openai_wan21_t2v_14b.yaml": "t2v",
    "openai_wan22_i2v_a14b.yaml": "i2v",
    "openai_hunyuan15_t2v_480p.yaml": "t2v",
    "openai_kandinsky5_t2v.yaml": "t2v",
    "openai_hunyuan15_i2v_480p.yaml": "i2v",
    "openai_kandinsky6_ti2va.yaml": "t2v",
    "openai_kandinsky6_ti2va_distilled.yaml": "t2v",
    "openai_ltx2_distilled.yaml": "t2v",
    "openai_pan2.yaml": "t2v",
    "openai_flux1_dev.yaml": "t2i",
    "openai_flux2_dev.yaml": "t2i",
    "openai_flux2_klein.yaml": "t2i",
    "openai_sd35.yaml": "t2i",
    "openai_stable_audio.yaml": "t2a",
    "openai_mmaudio.yaml": "v2a",
    "openai_fasth3_omniref_rtx5090.yaml": "i2v",
    "openai_fasth3_omniref_rtx_pro6000.yaml": "i2v",
    "openai_fasth3_omniref_spark.yaml": "i2v",
}


def _load(name: str) -> ServeConfig:
    path = ROOT / "examples" / "serving" / name
    return parse_config(ServeConfig, yaml.safe_load(path.read_text()))


def _args(config: ServeConfig, workload: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=config.generator.model_path,
        lora_path=None,
        lora_nickname="default",
        lora_strength=1.0,
        override_pipeline_cls_name=None,
        workload_type=SimpleNamespace(value=workload),
    )


@pytest.mark.parametrize(("name", "workload"), BASELINES.items())
def test_family_baseline_declares_workload_and_parses(name: str, workload: str) -> None:
    config = _load(name)
    assert config.generator.pipeline.workload_type == workload
    assert config.server.served_model_name


@pytest.mark.parametrize("device", ["rtx5090", "rtx_pro6000", "spark"])
def test_omniref_serving_matches_its_maintained_inference_recipe(device: str) -> None:
    served = yaml.safe_load((ROOT / "examples" / "serving" / f"openai_fasth3_omniref_{device}.yaml").read_text())
    recipe = yaml.safe_load(
        (ROOT / "examples" / "inference" / "basic" / f"basic_fasth3_omniref_{device}.yaml").read_text())
    assert served["generator"] == recipe["generator"]
    assert served["generator"]["pipeline"]["components"]["override_pipeline_cls_name"] == (
        "MiniMaxH3Ref2VAModularPipeline")
    for key in ("height", "width", "num_frames", "fps", "num_inference_steps", "guidance_scale"):
        assert served["default_request"]["sampling"][key] == recipe["request"]["sampling"][key], key


def test_hunyuan_i2v_480p_keeps_the_registered_checkpoint_semantics(tmp_path: Path) -> None:
    """The checkpoint is named 480p, but its registered preset renders 720x1280."""
    config = _load("openai_hunyuan15_i2v_480p.yaml")
    defaults = HUNYUAN15_I2V_480P_DISTILLED.defaults
    sampling = config.default_request.sampling

    assert config.generator.model_path.endswith("480p_i2v_step_distilled")
    for field in ("height", "width", "num_frames", "fps", "num_inference_steps", "guidance_scale"):
        assert getattr(sampling, field) == defaults[field]
    assert config.default_request.negative_prompt == defaults["negative_prompt"]

    image = tmp_path / "first-frame.png"
    image.write_bytes(b"not-decoded-at-admission")
    request = VideoGenerationRequest(prompt="a red fox", image_reference={"image_url": str(image)})
    adapted = build_generation_request(
        "i2v_test",
        request,
        _args(config, "i2v"),
        served_model_name=config.server.served_model_name or "",
        output_dir=str(tmp_path),
        default_request=config.default_request,
    )
    assert adapted.sampling.height == 720
    assert adapted.sampling.width == 1280
    assert adapted.inputs.image_path == str(image)
    assert adapted.output.output_path.endswith(".mp4")


def test_pan2_serving_matches_its_preset_and_runs_image_to_video_from_a_reference(tmp_path: Path) -> None:
    config = _load("openai_pan2.yaml")
    sampling = config.default_request.sampling
    assert config.generator.model_path == "IFM/PAN2"
    assert config.generator.engine.offload.dit is False
    assert config.generator.engine.offload.dit_layerwise is False
    for field in ("height", "width", "num_frames", "fps", "num_inference_steps", "guidance_scale"):
        assert getattr(sampling, field) == PAN2_T2V.defaults[field], field
    assert config.default_request.negative_prompt == PAN2_T2V.defaults["negative_prompt"]

    image = tmp_path / "first-frame.png"
    image.write_bytes(b"not-decoded-at-admission")
    for reference in (None, {"image_url": str(image)}):
        request = VideoGenerationRequest(prompt="a red fox", image_reference=reference)
        adapted = build_generation_request(
            "pan2_test",
            request,
            _args(config, "t2v"),
            served_model_name=config.server.served_model_name or "",
            output_dir=str(tmp_path),
            default_request=config.default_request,
        )
        assert adapted.sampling.num_frames == 313
        assert adapted.inputs.image_path == (str(image) if reference else None)
        assert adapted.output.output_path.endswith(".mp4")


def test_stable_audio_seconds_override_produces_wav(tmp_path: Path) -> None:
    config = _load("openai_stable_audio.yaml")
    request = VideoGenerationRequest(prompt="rain on a metal roof", seconds=6)
    adapted = build_generation_request(
        "audio_test",
        request,
        _args(config, "t2a"),
        served_model_name=config.server.served_model_name or "",
        output_dir=str(tmp_path),
        default_request=config.default_request,
    )
    assert adapted.sampling.audio_start_in_s == 0.0
    assert adapted.sampling.audio_end_in_s == 6.0
    assert adapted.output.output_path == str(tmp_path / "audio" / "audio_test.wav")


def test_mmaudio_uses_operator_duration_and_video_reference(tmp_path: Path) -> None:
    config = _load("openai_mmaudio.yaml")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-decoded-at-admission")
    request = VideoGenerationRequest(prompt="waves", seconds=4, video_reference={"video_url": str(video)})
    adapted = build_generation_request(
        "audio_test",
        request,
        _args(config, "v2a"),
        served_model_name=config.server.served_model_name or "",
        output_dir=str(tmp_path),
        default_request=config.default_request,
    )
    assert adapted.sampling.audio_end_in_s == 4.0
    assert adapted.inputs.video_path == str(video)
    assert adapted.output.output_path == str(tmp_path / "audio" / "audio_test.wav")


def test_flux_request_uses_flux_sampling_defaults() -> None:
    request = parse_config(GenerationRequest, {"prompt": "a red fox"})
    sampling = request_to_sampling_param(request, model_path="black-forest-labs/FLUX.1-dev")
    assert isinstance(sampling, FluxSamplingParam)
    assert sampling.num_inference_steps == 28
    assert sampling.guidance_scale == 3.5
    assert sampling.use_embedded_guidance is True


def test_matrixgame2_request_uses_its_registered_sampling_defaults() -> None:
    # The registered sampling-param class now applies to every family that declares one, not only FLUX.
    request = parse_config(GenerationRequest, {"prompt": "a road"})
    sampling = request_to_sampling_param(request, model_path="FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers")
    assert isinstance(sampling, MatrixGame2SamplingParam)
    assert (sampling.num_inference_steps, sampling.height, sampling.width, sampling.num_frames) == (3, 352, 640, 57)


def test_local_directory_without_model_index_uses_base_sampling_defaults(tmp_path: Path) -> None:
    request = parse_config(GenerationRequest, {"prompt": "a road"})
    sampling = request_to_sampling_param(request, model_path=str(tmp_path))
    assert type(sampling) is SamplingParam


def test_flux_explicit_sampling_overrides_win_over_flux_defaults() -> None:
    request = parse_config(
        GenerationRequest,
        {
            "prompt": "a red fox",
            "sampling": {
                "num_inference_steps": 4,
                "guidance_scale": 2.0,
                "use_embedded_guidance": False,
            },
        },
    )
    sampling = request_to_sampling_param(request, model_path="black-forest-labs/FLUX.1-dev")
    assert sampling.num_inference_steps == 4
    assert sampling.guidance_scale == 2.0
    assert sampling.use_embedded_guidance is False


def _serve_clip(payload: bytes) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/clip.mp4"


def test_local_media_url_is_downloaded_and_tracked_for_cleanup(tmp_path: Path) -> None:
    server, url = _serve_clip(b"local-clip")
    try:
        request = VideoGenerationRequest(prompt="waves", video_reference={"video_url": url})
        created = asyncio.run(
            prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy(allow_private_urls=True)))
        assert len(created) == 1
        assert Path(created[0]).read_bytes() == b"local-clip"
        discard_materialized_media(created)
        assert not Path(created[0]).exists()
    finally:
        server.shutdown()
        server.server_close()


def test_server_media_reference_limit_is_a_validated_serve_field() -> None:
    config = parse_config(
        ServeConfig,
        {
            "generator": {"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
            "server": {"media_reference_max_bytes": 4096},
        },
    )
    assert config.server.media_reference_max_bytes == 4096
    with pytest.raises(ValueError, match="must be positive"):
        parse_config(
            ServeConfig,
            {
                "generator": {"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
                "server": {"media_reference_max_bytes": 0},
            },
        )


def test_media_url_limit_is_configurable_and_removes_partial_file(tmp_path: Path) -> None:
    server, url = _serve_clip(b"larger-than-four")
    try:
        request = VideoGenerationRequest(prompt="waves", video_reference={"video_url": url})
        with pytest.raises(RequestAdaptationError, match="4-byte limit"):
            asyncio.run(
                prepare_reference_media("request",
                                        request,
                                        str(tmp_path),
                                        max_bytes=4,
                                        policy=MediaPolicy(allow_private_urls=True)))
        assert not list((tmp_path / "uploads").glob("*"))
    finally:
        server.shutdown()
        server.server_close()


def test_media_policy_fields_are_validated_serve_fields() -> None:
    config = parse_config(
        ServeConfig,
        {
            "generator": {"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
            "server": {"media_roots": ["/data/media"], "allow_private_media_urls": True},
        },
    )
    assert config.server.media_roots == ["/data/media"]
    assert config.server.allow_private_media_urls is True
    default = parse_config(ServeConfig, {"generator": {"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}})
    assert default.server.media_roots == []
    assert default.server.allow_private_media_urls is False


def test_loopback_media_urls_are_rejected_by_default(tmp_path: Path) -> None:
    server, url = _serve_clip(b"local-clip")
    try:
        request = VideoGenerationRequest(prompt="waves", video_reference={"video_url": url})
        with pytest.raises(RequestAdaptationError, match="private or local address"):
            asyncio.run(prepare_reference_media("request", request, str(tmp_path), policy=MediaPolicy()))
        assert not list((tmp_path / "uploads").glob("*"))
    finally:
        server.shutdown()
        server.server_close()


def test_discard_materialized_media_removes_only_downloaded_files(tmp_path: Path) -> None:
    downloaded = tmp_path / "request_video.mp4"
    downloaded.write_bytes(b"temporary")
    caller_owned = tmp_path / "caller.mp4"
    caller_owned.write_bytes(b"keep")
    discard_materialized_media([str(downloaded)])
    assert not downloaded.exists()
    assert caller_owned.exists()
