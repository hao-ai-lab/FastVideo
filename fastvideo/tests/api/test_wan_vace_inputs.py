# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for Wan-VACE conditioning inputs."""

import pytest
import torch
from PIL import Image

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.wan.stages.vace_input import preprocess_vace_reference_images
from fastvideo.pipelines.basic.wan.stages.denoising import WanDenoisingStage, WanDenoisingState
from fastvideo.pipelines.basic.wan.stages.vace_denoising import WanVACEDenoisingStage
from fastvideo.pipelines.basic.wan.stages.vace_input import WanVACEInputStage
from fastvideo.pipelines.basic.wan.stages.vace_latent_preparation import WanVACELatentPreparationStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage


@pytest.fixture(params=("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "Wan-AI/Wan2.1-VACE-14B-diffusers"))
def vace_args(request):
    return FastVideoArgs.from_kwargs(model_path=request.param)


def _batch(**kwargs):
    return ForwardBatch(data_type="video", height=32, width=32, num_frames=5, fps=16, **kwargs)


@pytest.mark.parametrize("inputs", ({}, {"mask_path": "mask.mp4"}, {"references": ["reference.png"]}))
def test_vace_without_source_video_synthesizes_zero_pixels(monkeypatch, vace_args, inputs):
    monkeypatch.setattr(
        "fastvideo.pipelines.basic.wan.stages.vace_input.load_video_path_to_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 5, 32, 32),
    )
    monkeypatch.setattr("fastvideo.pipelines.basic.wan.stages.vace_input.preprocess_vace_reference_images",
                        lambda *args, **kwargs: [torch.zeros(3, 32, 32)])

    batch = WanVACEInputStage().forward(_batch(**inputs), vace_args)

    assert batch.video_latent is not None
    assert batch.video_latent.shape == (1, 3, 5, 32, 32)
    assert torch.count_nonzero(batch.video_latent) == 0


def test_vace_reference_image_preserves_aspect_ratio(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (32, 16), color=(255, 0, 0)).save(path)

    image, = preprocess_vace_reference_images([str(path)], (32, 32), torch.device("cpu"), torch.float32)

    assert image.shape == (3, 32, 32)
    assert torch.all(image[:, 0] == 1)
    assert torch.all(image[:, -1] == 1)
    assert image[1, 16, 16] < 0


def test_vace_mask_resamples_to_video_fps(monkeypatch, vace_args):
    def load_mask_tensor(path, **kwargs):
        assert kwargs["target_fps"] == 16
        values = torch.tensor([i * 40 / 127.5 - 1 for i in range(5)])
        return values.view(1, 1, 5, 1, 1).expand(1, 3, 5, 32, 32).clone()

    monkeypatch.setattr("fastvideo.pipelines.basic.wan.stages.vace_input.load_video_path_to_tensor", load_mask_tensor)
    batch = _batch(video_path="video.mp4", video_latent=torch.zeros(1, 3, 5, 32, 32), mask_path="mask.mp4")

    WanVACEInputStage().forward(batch, vace_args)

    assert batch.mask_video is not None
    assert batch.mask_video.shape == batch.video_latent.shape
    assert torch.allclose(batch.mask_video[0, 0, :, 0, 0], torch.tensor([i * 40 / 127.5 - 1 for i in range(5)]))


def test_vace_video_and_mask_use_same_fps_samples(monkeypatch, vace_args):
    frames = [Image.new("RGB", (32, 32), color=(i * 20, ) * 3) for i in range(10)]

    def load_frames(*args, **kwargs):
        return frames, 32

    from fastvideo.models.vision_utils import normalize, numpy_to_pt, pil_to_numpy, resize

    def load_video_tensor(path, **kwargs):
        resized = [resize(img, kwargs["target_height"], kwargs["target_width"]) for img in frames[::2]]
        video_numpy = normalize(pil_to_numpy(resized))
        return numpy_to_pt(video_numpy).permute(1, 0, 2, 3).unsqueeze(0)

    monkeypatch.setattr("fastvideo.pipelines.stages.video_tensor_utils.load_video_path_to_tensor", load_video_tensor)
    monkeypatch.setattr("fastvideo.pipelines.basic.wan.stages.vace_input.load_video_path_to_tensor", load_video_tensor)
    batch = _batch(video_path="video.mp4", mask_path="mask.mp4", prompt="test", seed=42)

    InputValidationStage().forward(batch, vace_args)
    WanVACEInputStage().forward(batch, vace_args)

    assert batch.video_latent.shape == batch.mask_video.shape == (1, 3, 5, 32, 32)
    assert torch.equal(batch.video_latent, batch.mask_video)
    assert torch.allclose(batch.video_latent[0, 0, :, 0, 0],
                          torch.tensor([i * 40 / 127.5 - 1 for i in range(5)]))


def test_vace_mask_and_video_frame_count_must_match(monkeypatch, vace_args):
    monkeypatch.setattr(
        "fastvideo.pipelines.basic.wan.stages.vace_input.load_video_path_to_tensor",
        lambda *args, **kwargs: torch.zeros(1, 3, 4, 32, 32),
    )
    batch = _batch(video_path="video.mp4", video_latent=torch.zeros(1, 3, 5, 32, 32), mask_path="mask.mp4")

    with pytest.raises(ValueError, match="mask.*frames.*video"):
        WanVACEInputStage().forward(batch, vace_args)


def test_vace_input_stage_does_not_leak_references_between_requests(monkeypatch, vace_args):
    monkeypatch.setattr("fastvideo.pipelines.basic.wan.stages.vace_input.preprocess_vace_reference_images",
                        lambda *args, **kwargs: [torch.zeros(3, 32, 32)])
    stage = WanVACEInputStage()

    first = stage.forward(_batch(references=["reference.png"]), vace_args)
    second = stage.forward(_batch(), vace_args)

    assert first.vace_num_reference_frames == 1
    assert second.vace_num_reference_frames == 0
    assert second.vace_reference_images is None


def test_vace_reference_frame_padding_restores_request_on_failure(monkeypatch, vace_args):
    def fail_after_recording_frames(self, batch, fastvideo_args):
        assert batch.num_frames == 9
        raise RuntimeError("synthetic preparation failure")

    monkeypatch.setattr(LatentPreparationStage, "forward", fail_after_recording_frames)
    batch = _batch(vace_num_reference_frames=1)
    stage = WanVACELatentPreparationStage(scheduler=object(), transformer=object())

    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        stage.forward(batch, vace_args)
    assert batch.num_frames == 5


def test_vace_denoising_preserves_wan_request_state(monkeypatch, vace_args):
    latents = torch.zeros(1, 16, 2, 4, 4)
    control = torch.zeros(1, 96, 2, 4, 4)
    batch = _batch(latents=latents, vace_control_latents=control, conditioning_scale=0.5)
    monkeypatch.setattr(WanDenoisingStage, "prepare_denoising",
                        lambda self, *args: WanDenoisingState(latents=latents, boundary_timestep=42))
    stage = WanVACEDenoisingStage.__new__(WanVACEDenoisingStage)

    state = stage.prepare_denoising(batch, vace_args, torch.float32)

    assert state.boundary_timestep == 42
    assert state.control_hidden_states is control
    assert state.control_hidden_states_scale == 0.5
