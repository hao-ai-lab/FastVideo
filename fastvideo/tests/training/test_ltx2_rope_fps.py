# SPDX-License-Identifier: Apache-2.0
"""The legacy LTX-2 trainer must hand the clip's fps to the DiT.

The DiT divides temporal RoPE positions by ``forward_batch.fps`` from the
forward context. Inference always sets it, so a training forward without it
sees temporal positions in frames instead of seconds.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.configs import VideoLoaderType
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.forward_context import get_forward_context
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.pipelines.preprocess.ltx2.ltx2_preprocess_pipelines import LTX2AudioEncodingStage
from fastvideo.pipelines.preprocess.preprocess_stages import VideoTransformStage
from fastvideo.training import ltx2_training_pipeline as ltx2_module
from fastvideo.training.ltx2_training_pipeline import LTX2TrainingPipeline
from fastvideo.training.trackers import DummyTracker
from fastvideo.workflow.preprocess.preprocess_workflow_ltx2_t2v import LTX2PrecomputedSaver


class _FpsRecordingDiT(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.seen_fps: list[float | None] = []

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        self.seen_fps.append(None if forward_batch is None else forward_batch.fps)
        return hidden_states * self.scale


def _pipeline(monkeypatch: pytest.MonkeyPatch) -> LTX2TrainingPipeline:
    monkeypatch.setattr(ltx2_module, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ltx2_module, "get_world_group",
                        lambda: SimpleNamespace(all_reduce=lambda tensor, op=None: tensor))
    pipeline = object.__new__(LTX2TrainingPipeline)
    pipeline.transformer = _FpsRecordingDiT()
    pipeline.tracker = DummyTracker()
    pipeline.with_audio = False
    pipeline.training_args = SimpleNamespace(num_latent_t=2,
                                             ltx2_first_frame_conditioning_p=0.0,
                                             gradient_accumulation_steps=1)
    pipeline.noise_random_generator = torch.Generator("cpu").manual_seed(0)
    pipeline.noise_gen_cuda = torch.Generator("cpu").manual_seed(0)
    return pipeline


def _pt_batch(fps: torch.Tensor | None) -> dict:
    batch_size = 1 if fps is None else fps.numel()
    latents = {"latents": torch.randn(batch_size, 4, 2, 2, 2)}
    if fps is not None:
        latents["fps"] = fps
    return {
        "latents": latents,
        "conditions": {
            "video_prompt_embeds": torch.randn(batch_size, 3, 8),
            "audio_prompt_embeds": torch.randn(batch_size, 3, 8),
            "prompt_attention_mask": torch.ones(batch_size, 3),
        },
    }


def _train_step_fps(pipeline: LTX2TrainingPipeline, batch: dict) -> float | None:
    training_batch = pipeline._get_next_batch_pt(batch, TrainingBatch())
    training_batch = pipeline._prepare_dit_inputs(training_batch)
    training_batch = pipeline._build_input_kwargs(training_batch)
    training_batch.total_loss = 0.0
    pipeline._transformer_forward_and_compute_loss(training_batch)
    return pipeline.transformer.seen_fps[-1]


def test_training_forward_sees_the_clip_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(monkeypatch)
    assert _train_step_fps(pipeline, _pt_batch(torch.tensor([24.0]))) == 24.0
    assert _train_step_fps(pipeline, _pt_batch(torch.tensor([30.0]))) == 30.0


def test_batch_without_fps_uses_the_preset_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(monkeypatch)
    assert _train_step_fps(pipeline, _pt_batch(None)) == 24.0


def test_mixed_fps_batch_uses_the_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _pipeline(monkeypatch)
    assert _train_step_fps(pipeline, _pt_batch(torch.tensor([25.0, 30.0]))) == 25.0


class _Clip(str):
    """A 60-frame video path whose frames read like a torchcodec decoder's."""

    frames = torch.zeros(60, 3, 16, 16, dtype=torch.uint8)

    def get_frames_at(self, indices) -> SimpleNamespace:
        return SimpleNamespace(data=self.frames[list(indices)])


def test_resampled_clip_keeps_train_fps_for_audio_and_saving(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_audio_seconds: list[float] = []

    def extract_audio(video_path: str, target_duration: float) -> None:
        requested_audio_seconds.append(target_duration)

    monkeypatch.setattr(LTX2AudioEncodingStage, "_extract_audio", staticmethod(extract_audio))
    batch = SimpleNamespace(data_type="video",
                            video_loader=[_Clip("clip.mp4")],
                            video_file_name=["clip.mp4"],
                            fps=[30.0],
                            num_frames=[60],
                            height=[16],
                            width=[16],
                            prompt_embeds=[torch.zeros(1, 3, 8)],
                            prompt_attention_mask=[torch.ones(1, 3)],
                            extra={})
    args = SimpleNamespace(preprocess_config=SimpleNamespace(video_loader_type=VideoLoaderType.TORCHCODEC),
                           workload_type=WorkloadType.T2V)

    VideoTransformStage(train_fps=24, num_frames=9, max_height=16, max_width=16,
                        do_temporal_sample=False).forward(batch, args)
    LTX2AudioEncodingStage(torch.nn.Linear(1, 1), audio_processor=None, fallback_fps=24).forward(batch, args)
    LTX2PrecomputedSaver(tmp_path).save_batch(batch)

    # 9 frames sampled at 24 fps from a 30 fps source cover 9 / 24 s.
    assert requested_audio_seconds == [pytest.approx(9 / 24)]
    assert torch.load(tmp_path / "latents" / "clip.pt")["fps"] == 24.0
    assert batch.num_frames == [9]
    assert batch.fps == [24]
