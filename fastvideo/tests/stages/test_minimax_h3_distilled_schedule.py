# SPDX-License-Identifier: Apache-2.0
"""Checkpoint scheduler shifts and explicit FastH3 DMD inference schedules."""
from __future__ import annotations

from types import SimpleNamespace

import json
import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3ModularPipeline
from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as denoising
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


DMD_STEPS = [999, 874, 749, 624, 500, 375, 250, 125]
CONTRACT = {
    "schema_version": "fasth3-inference-contract-v1",
    "dmd_denoising_steps": DMD_STEPS,
    "num_inference_steps": 9,
    "transformer_forwards": 8,
    "video_scheduler_shift": 10.0,
    "audio_scheduler_shift": 3.0,
}


def _pipeline(tmp_path, *, video_shift=10.0):
    pipeline = object.__new__(MiniMaxH3ModularPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.modules = {"scheduler": MiniMaxH3Scheduler(shift=video_shift),
                        "audio_scheduler": MiniMaxH3Scheduler(shift=3.0)}
    return pipeline


def test_pipeline_preserves_shift10_from_checkpoint(tmp_path):
    # Exercise real initialization with weight loading omitted. The schedulers
    # represent scheduler_config.json from the eight-forward T2AV checkpoint.
    pipeline = object.__new__(MiniMaxH3ModularPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.modules = {
        "scheduler": MiniMaxH3Scheduler(shift=10.0),
        "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
    }
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    pipeline.initialize_pipeline(args)
    assert pipeline.modules["scheduler"].shift == 10.0
    assert pipeline.modules["audio_scheduler"].shift == 3.0


class _TransformerSpy:
    def __init__(self, *, fail_transfer=False):
        self.device = torch.device("cpu")
        self.transfers = []
        self.calls = []
        self.fail_transfer = fail_transfer

    def to(self, device):
        self.device = torch.device(device)
        self.transfers.append(self.device)
        if self.fail_transfer and self.device != torch.device("cpu"):
            raise RuntimeError("injected transfer failure")
        return self

    def __call__(self, **kwargs):
        self.calls.append({key: value.clone() for key, value in kwargs.items()})
        return torch.full_like(kwargs["hidden_states"], 2.0), torch.full_like(kwargs["audio_hidden_states"], -3.0)


def _run_tiny_stage(monkeypatch, steps, grid_points, *, video_shift=10.0, offloaded_transformer=None):
    # cpu:0 is a distinct spy destination from cpu, but all tensors stay on CPU.
    device = torch.device("cpu:0" if offloaded_transformer is not None else "cpu")
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: device)
    monkeypatch.setattr(denoising, "_h3_vsa_metadata_builder", lambda *_: None)
    layout = build_packed_sequence(torch.ones(2, dtype=torch.long), 2, 2, 2, 2, (1, 2, 2))
    transformer = offloaded_transformer if offloaded_transformer is not None else _TransformerSpy()

    stage = denoising.MiniMaxH3DenoisingStage(transformer, MiniMaxH3Scheduler(shift=video_shift),
                                           MiniMaxH3Scheduler(shift=3.0))
    args = SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig(dmd_denoising_steps=steps),
                           dit_cpu_offload=offloaded_transformer is not None,
                           dit_layerwise_offload=False, use_fsdp_inference=False)
    batch = ForwardBatch(
        data_type="video", prompt_embeds=[torch.zeros(1, 2, 8)], latents=torch.ones(2, 96), audio_latents=torch.ones(4, 32),
        num_inference_steps=grid_points, extra={MINIMAX_H3_LAYOUT_KEY: layout})
    result = stage.forward(batch, args)
    return stage, transformer.calls, result


@pytest.mark.parametrize("steps,grid_points,error", [
    (DMD_STEPS, 8, "8 DMD forwards require 9"),
    ([1001], 2, "strictly decreasing integers"),
])
def test_invalid_dmd_schedule_keeps_offloaded_transformer_on_cpu(monkeypatch, steps, grid_points, error):
    transformer = _TransformerSpy()
    with pytest.raises(ValueError, match=error):
        _run_tiny_stage(monkeypatch, steps, grid_points, offloaded_transformer=transformer)
    assert transformer.device == torch.device("cpu")
    assert transformer.transfers == []
    assert transformer.calls == []


def test_offloaded_transformer_is_cleaned_up_after_transfer_failure(monkeypatch):
    transformer = _TransformerSpy(fail_transfer=True)
    with pytest.raises(RuntimeError, match="injected transfer failure"):
        _run_tiny_stage(monkeypatch, DMD_STEPS, 9, offloaded_transformer=transformer)
    assert transformer.device == torch.device("cpu")
    assert transformer.transfers == [torch.device("cpu:0"), torch.device("cpu")]
    assert transformer.calls == []


def test_offloaded_transformer_returns_to_cpu_after_denoising(monkeypatch):
    transformer = _TransformerSpy()
    _run_tiny_stage(monkeypatch, DMD_STEPS, 9, offloaded_transformer=transformer)
    assert transformer.device == torch.device("cpu")
    assert transformer.transfers == [torch.device("cpu:0"), torch.device("cpu")]
    assert len(transformer.calls) == 8


def test_eight_forward_schedule_uses_trained_rungs_for_both_modalities(monkeypatch):
    stage, calls, result = _run_tiny_stage(monkeypatch, DMD_STEPS, 9)
    assert len(calls) == 8
    assert result.step_index == 7
    # Independent equation: the rungs are unshifted noise amounts on the
    # 1000-step training clock, NOT indices into a uniform nine-point grid.
    base = torch.tensor([rung / 1000.0 for rung in DMD_STEPS] + [0.0], dtype=torch.float32)
    for scheduler, shift in ((stage.scheduler, 10.0), (stage.audio_scheduler, 3.0)):
        expected = shift * base / (1 + (shift - 1) * base)
        assert_close(scheduler.sigmas, expected, rtol=0, atol=0)
        assert_close(scheduler.timesteps, 1.0 - expected[:-1], rtol=0, atol=0)
        assert scheduler.step_index == 8

    video_sigmas = 10.0 * base / (1 + 9.0 * base)
    audio_sigmas = 3.0 * base / (1 + 2.0 * base)
    for index, call in enumerate(calls):
        row_times = call["timestep"][call["timestep_indices"]]
        # build_row_timesteps assigns the video clean clock to text too;
        # this T2AV layout has no conditioned video or audio prefix rows.
        for indices, sigmas in (("video_indices", video_sigmas), ("audio_indices", audio_sigmas),
                                ("text_indices", video_sigmas)):
            actual = row_times[call[indices]]
            assert actual.numel() > 0
            assert_close(actual, torch.full_like(actual, 1.0 - sigmas[index]), rtol=0, atol=0)
        # Nonzero, modality-distinct velocities expose skipped or swapped updates.
        assert_close(call["hidden_states"], torch.full_like(call["hidden_states"],
                                                          1.0 + 2.0 * (video_sigmas[0] - video_sigmas[index])))
        assert_close(call["audio_hidden_states"], torch.full_like(call["audio_hidden_states"],
                                                                1.0 - 3.0 * (audio_sigmas[0] - audio_sigmas[index])))
    assert_close(result.latents, torch.full_like(result.latents, 1.0 + 2.0 * video_sigmas[0]))
    assert_close(result.audio_latents, torch.full_like(result.audio_latents, 1.0 - 3.0 * audio_sigmas[0]))


def test_pipeline_loads_the_exported_ladder_and_rejects_a_conflicting_one(tmp_path):
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(CONTRACT))
    pipeline = _pipeline(tmp_path)
    config = MiniMaxH3PipelineConfig()
    pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=config))
    assert config.dmd_denoising_steps == DMD_STEPS

    wrong = MiniMaxH3PipelineConfig(dmd_denoising_steps=[1000, 750, 500, 250])
    with pytest.raises(ValueError, match="checkpoint.*DMD"):
        pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=wrong))


@pytest.mark.parametrize("grid_points", [5, 50])
def test_base_and_four_forward_schedules_are_unchanged(monkeypatch, tmp_path, grid_points):
    config = MiniMaxH3PipelineConfig()
    pipeline = _pipeline(tmp_path, video_shift=12.0)
    pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=config))
    assert pipeline.modules["scheduler"].shift == 12.0
    assert pipeline.modules["audio_scheduler"].shift == 3.0
    assert config.dmd_denoising_steps is None
    assert config.text_encoder_precisions == ("bf16",)
    stage, calls, _ = _run_tiny_stage(monkeypatch, None, grid_points, video_shift=12.0)
    assert len(calls) == grid_points - 1
    base = torch.linspace(1.0, 0.0, grid_points, dtype=torch.float32)
    for scheduler, shift in ((stage.scheduler, 12.0), (stage.audio_scheduler, 3.0)):
        assert_close(scheduler.sigmas, shift * base / (1 + (shift - 1) * base), rtol=0, atol=0)


@pytest.mark.parametrize("rungs", [[], [0], [-1], [1001], [999, 999], [100, 200], [999, 0], [True], [999.0]])
def test_invalid_ladders_are_rejected(monkeypatch, rungs):
    with pytest.raises(ValueError, match="strictly decreasing integers"):
        _run_tiny_stage(monkeypatch, rungs, len(rungs) + 1)


@pytest.mark.parametrize("grid_points", [5, 8, 10, 50])
def test_explicit_eight_forward_ladder_requires_nine_grid_points(monkeypatch, grid_points):
    with pytest.raises(ValueError, match="8 DMD forwards require 9"):
        _run_tiny_stage(monkeypatch, DMD_STEPS, grid_points)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unrecognized"), ("dmd_denoising_steps", None), ("dmd_denoising_steps", []),
    ("dmd_denoising_steps", [999, 999]), ("num_inference_steps", 8), ("transformer_forwards", 9),
    ("video_scheduler_shift", 12.0), ("audio_scheduler_shift", 4.0),
    ("video_scheduler_shift", float("nan")), ("audio_scheduler_shift", True),
])
def test_checkpoint_schedule_metadata_must_be_self_consistent(tmp_path, field, value):
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps({**CONTRACT, field: value}))
    with pytest.raises(ValueError):
        _pipeline(tmp_path).initialize_pipeline(SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig()))


@pytest.mark.parametrize("module_name", ["scheduler", "audio_scheduler"])
@pytest.mark.parametrize("shift", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_checkpoint_shifts_must_be_positive_and_finite(tmp_path, module_name, shift):
    pipeline = _pipeline(tmp_path)
    pipeline.modules[module_name] = SimpleNamespace(shift=shift)
    with pytest.raises(ValueError, match="positive finite shift"):
        pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig()))


def test_sidecar_without_shift_fields_uses_the_checkpoint_schedulers(tmp_path):
    """The published four-step checkpoints predate the shift fields in fasth3_inference_contract_v1
    (they carry only the ladder). Absent keys must fall back to scheduler/*_config.json; a present but
    conflicting key is still an error."""
    four_step = {**CONTRACT, "dmd_denoising_steps": [999, 749, 500, 250], "num_inference_steps": 5,
                 "transformer_forwards": 4}
    four_step.pop("video_scheduler_shift")
    four_step.pop("audio_scheduler_shift")
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(four_step))
    pipeline = _pipeline(tmp_path, video_shift=12.0)
    config = MiniMaxH3PipelineConfig()
    pipeline.initialize_pipeline(SimpleNamespace(pipeline_config=config))
    assert config.dmd_denoising_steps == [999, 749, 500, 250]
    assert pipeline.modules["scheduler"].shift == 12.0
    assert pipeline.modules["audio_scheduler"].shift == 3.0
    # Explicit disagreement is still rejected.
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps({**four_step, "video_scheduler_shift": 10.0}))
    with pytest.raises(ValueError, match="disagrees"):
        _pipeline(tmp_path, video_shift=12.0).initialize_pipeline(SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig()))
