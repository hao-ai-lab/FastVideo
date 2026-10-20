# SPDX-License-Identifier: Apache-2.0
"""Declaration checks for family serving baselines; no model weights are loaded."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fastvideo.api.parser import parse_config
from fastvideo.api.schema import ServeConfig
from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
from fastvideo.entrypoints.openai.request_adapter import build_generation_request

ROOT = Path(__file__).resolve().parents[3]

BASELINES = {
    "openai_wan22_i2v_a14b.yaml": "i2v",
    "openai_hunyuan15_t2v_480p.yaml": "t2v",
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
