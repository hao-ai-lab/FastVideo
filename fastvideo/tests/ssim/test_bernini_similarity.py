# SPDX-License-Identifier: Apache-2.0
"""Bernini-R V2V quality gate: one CUDA GPU, pinned public weights and input.

Uses SDPA and device/backend/runtime-matched references. The source video,
checkpoint revision and sampling recipe are pinned below.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import requests
import torch
from huggingface_hub import snapshot_download

import fastvideo.envs as envs
from fastvideo import VideoGenerator
from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig, InputConfig, OffloadConfig,
                           OutputConfig, ParallelismConfig, SamplingConfig)
from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import _assert_similarity, attention_backend
from fastvideo.tests.ssim.reference_utils import (build_generated_output_dir, build_reference_folder_path,
                                                 get_cuda_device_name, resolve_device_reference_folder,
                                                 select_ssim_params)


def _checkpoint_converter():
    # `pytest`'s console entrypoint need not put the repository root on
    # sys.path. Load the repository-owned converter without a scripts import
    # or a process-wide sys.path mutation.
    path = Path(__file__).resolve().parents[3] / "scripts/checkpoint_conversion/bernini_r_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("bernini_ssim_converter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _checkpoint_converter()
SOURCE_REPO = converter.SOURCE_REPO
SOURCE_REVISION = converter.SOURCE_REVISION
prepare_checkpoint = converter.prepare_checkpoint

logger = init_logger(__name__)
REQUIRED_GPUS = 1
MIN_SSIM = 0.98
OFFICIAL_CODE_REVISION = "e6c2cf11843470b08491745947abcba9c3c6b079"
SOURCE_VIDEO_URL = (f"https://raw.githubusercontent.com/bytedance/Bernini/{OFFICIAL_CODE_REVISION}/"
                    "assets/testcases/v2v/source_case1.mp4")
SOURCE_VIDEO_SHA256 = "a1bf05d790e0b17caa82a0d179e383cf53c35a3fb7e0bbd7960f08d6e1fd4c85"

BERNINI_PARAMS = {
    "height": 480,
    "width": 848,
    "num_frames": 81,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "fps": 16,
    "seed": 42,
}
BERNINI_MODEL_TO_PARAMS = {
    "Bernini-R-1.3B-Diffusers": BERNINI_PARAMS,
}
BERNINI_FULL_QUALITY_MODEL_TO_PARAMS = {
    "Bernini-R-1.3B-Diffusers": dict(BERNINI_PARAMS),
}
TEST_PROMPTS = [
    "Add a realistic snowman on the right side of the snowy path, positioned in the mid-right ground so it sits "
    "naturally beside the trail without blocking the black-and-white dog. Make it a classic three-snowball snowman, "
    "about medium height relative to the dog, with a carrot nose, small coal eyes, a few coal buttons, thin twig arms, "
    "and a dark knit hat with a muted scarf. Match the overcast winter lighting and snowy atmosphere, "
    "with soft diffuse "
    "shadows, slight snow accumulation on the hat and shoulders, and its base partially embedded in the snow. Ensure "
    "stable placement and scale across the video, with accurate perspective, consistent contact with the ground, and "
    "no flicker or drifting. Keep the dog, the snow-covered road, the snowy trees and bushes, the distant green post, "
    "and the overall winter scene unchanged.",
]


def _validate_source_revision(source: Path) -> None:
    metadata = source / ".cache/huggingface/download/config.json.metadata"
    if metadata.is_file():
        revision = metadata.read_text().splitlines()[0]
    elif source.parent.name == "snapshots":
        revision = source.name
    else:
        raise ValueError("Bernini SSIM requires a pinned Hub snapshot or hf download --local-dir metadata")
    if revision != SOURCE_REVISION:
        raise ValueError(f"Bernini SSIM requires checkpoint revision {SOURCE_REVISION}, got {revision}")


def _download_source_video(destination: Path) -> Path:
    with requests.get(SOURCE_VIDEO_URL, timeout=120) as response:
        response.raise_for_status()
        data = response.content
    if hashlib.sha256(data).hexdigest() != SOURCE_VIDEO_SHA256:
        raise ValueError("Bernini SSIM source video failed its SHA-256 check")
    destination.write_bytes(data)
    return destination


@pytest.fixture(scope="session")
def bernini_device_reference_folder():
    if not torch.cuda.is_available():
        pytest.skip("Bernini video-editing SSIM requires one CUDA GPU")
    device_name = get_cuda_device_name()
    return resolve_device_reference_folder(
        (("GB200", "GB200"), ("B200", "B200"), ("H200", "H200"), ("H100", "H100"),
         ("L40S", "L40S"), ("A40", "A40")),
        device_name=device_name,
        unknown_device_prefix="".join(character for character in device_name.removeprefix("NVIDIA ")
                                       if character.isalnum()),
    )


@pytest.fixture(scope="session")
def bernini_checkpoint(tmp_path_factory, bernini_device_reference_folder):
    source_dir = envs.FASTVIDEO_TEST_BERNINI_MODEL_DIR.get()
    if source_dir:
        source = Path(source_dir)
        if not source.is_dir():
            pytest.skip(f"Bernini checkpoint directory does not exist: {source}")
    else:
        source = Path(snapshot_download(repo_id=SOURCE_REPO, revision=SOURCE_REVISION,
                                       allow_patterns=["config.json", "transformer/*", "text_encoder/*",
                                                       "tokenizer/*", "vae/*", "scheduler/*"]))
    _validate_source_revision(source)
    return prepare_checkpoint(source, tmp_path_factory.mktemp("bernini-model") / "checkpoint")


@pytest.fixture(scope="session")
def bernini_source_video(tmp_path_factory, bernini_device_reference_folder):
    return _download_source_video(tmp_path_factory.mktemp("bernini-input") / "source.mp4")


@pytest.mark.gpu
@pytest.mark.parametrize("prompt", TEST_PROMPTS, ids=["snowman_edit"])
@pytest.mark.parametrize("attention_backend_name", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(BERNINI_MODEL_TO_PARAMS))
def test_bernini_video_editing_similarity(prompt, attention_backend_name, model_id, bernini_checkpoint,
                                         bernini_source_video, bernini_device_reference_folder):
    script_dir = str(Path(__file__).resolve().parent)
    output_dir = build_generated_output_dir(script_dir, bernini_device_reference_folder, model_id,
                                            attention_backend_name)
    params = select_ssim_params(BERNINI_MODEL_TO_PARAMS, BERNINI_FULL_QUALITY_MODEL_TO_PARAMS)[model_id]
    with attention_backend(attention_backend_name):
        generator = VideoGenerator.from_config(GeneratorConfig(
            model_path=str(bernini_checkpoint),
            engine=EngineConfig(num_gpus=1, use_fsdp_inference=False,
                                parallelism=ParallelismConfig(sp_size=1, tp_size=1),
                                offload=OffloadConfig(dit=True, dit_layerwise=False, text_encoder=True, vae=True)),
        ))
        try:
            result = generator.generate(request=GenerationRequest(
                prompt=prompt, negative_prompt="",
                inputs=InputConfig(video_path=str(bernini_source_video)),
                sampling=SamplingConfig(**params),
                output=OutputConfig(output_path=output_dir, save_video=True, return_frames=False),
            ))
        finally:
            generator.shutdown()

    expected_size = (params["height"], params["width"], params["num_frames"])
    assert result.size == expected_size, "Unexpected Bernini output geometry"
    assert result.video_path and Path(result.video_path).is_file(), "Bernini did not return a generated video"
    generated_video = Path(result.video_path)
    # Use this run's returned filename, never an older prompt-matching file.
    _assert_similarity(
        logger=logger, output_dir=str(generated_video.parent), output_media_name=generated_video.name,
        reference_folder=build_reference_folder_path(script_dir, bernini_device_reference_folder, model_id,
                                                      attention_backend_name),
        prompt=prompt, num_inference_steps=params["num_inference_steps"], min_acceptable_ssim=MIN_SSIM,
        model_id=model_id, attention_backend_name=attention_backend_name, media_extension=".mp4",
    )
