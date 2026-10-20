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
from fastvideo.api.parser import parse_config
from fastvideo.api.schema import GenerationRequest, ServeConfig
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import (
    RequestAdaptationError,
    build_generation_request,
    discard_materialized_media,
    prepare_reference_media,
)

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
    "openai_flux1_dev.yaml": "t2i",
    "openai_flux2_dev.yaml": "t2i",
    "openai_flux2_klein.yaml": "t2i",
    "openai_sd35.yaml": "t2i",
    "openai_stable_audio.yaml": "t2a",
    "openai_mmaudio.yaml": "v2a",
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
        created = asyncio.run(prepare_reference_media("request", request, str(tmp_path)))
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
            asyncio.run(prepare_reference_media("request", request, str(tmp_path), max_bytes=4))
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
