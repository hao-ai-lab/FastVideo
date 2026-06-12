# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for StreamingLongTuningMethod control flow."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.train.methods.distribution_matching.streaming_long_tuning import (
    DistillStage,
    StreamingLongTuningMethod,
    _StreamingChunkInfo,
)


def _method_without_init() -> StreamingLongTuningMethod:
    return StreamingLongTuningMethod.__new__(StreamingLongTuningMethod)


def test_single_train_step_dispatches_short_stage() -> None:
    method = _method_without_init()
    short_stage = DistillStage(
        name="self_forcing",
        start_step=0,
        end_step=10,
        num_latent_t=3,
        streaming_training=False,
    )
    long_stage = DistillStage(
        name="streaming_long",
        start_step=10,
        end_step=None,
        num_latent_t=5,
        streaming_training=True,
        streaming_chunk_size=3,
        streaming_max_length=5,
    )
    method._distill_stages = [short_stage, long_stage]
    method._streaming_state = object()
    method.training_config = SimpleNamespace(data=SimpleNamespace(num_latent_t=5))
    method.cuda_generator = torch.Generator(device="cpu")
    calls: list[tuple[str, Any]] = []

    class Student:

        def prepare_batch(
            self,
            batch: dict[str, Any],
            *,
            generator: torch.Generator,
            latents_source: str,
        ) -> SimpleNamespace:
            del generator
            calls.append(("prepare", latents_source))
            assert batch["raw"] == "batch"
            return SimpleNamespace(latents=torch.zeros(1, 5, 2, 1, 1))

    method.student = Student()
    method._with_padded_first_frame_latent = (
        lambda batch, *, target_latent_frames: batch
    )
    method._truncate_batch_latents = (
        lambda training_batch, num_latent_t: calls.append(("truncate", num_latent_t))
    )
    method._losses_for_batch = (
        lambda training_batch, iteration, *, stage, chunk_mask: (
            {"total_loss": torch.tensor(1.0)},
            {"stage": stage.name, "chunk_mask": chunk_mask},
            {"active_num_latent_t": float(stage.num_latent_t)},
        )
    )

    loss_map, outputs, metrics = method.single_train_step({"raw": "batch"}, iteration=0)

    assert loss_map["total_loss"].item() == 1.0
    assert outputs == {"stage": "self_forcing", "chunk_mask": None}
    assert metrics == {"active_num_latent_t": 3.0}
    assert calls == [("prepare", "zeros"), ("truncate", 3)]
    assert method._streaming_state is None


def test_single_train_step_streaming_stage_returns_metrics() -> None:
    method = _method_without_init()
    stage = DistillStage(
        name="streaming_long",
        start_step=0,
        end_step=None,
        num_latent_t=5,
        streaming_training=True,
        streaming_chunk_size=3,
        streaming_max_length=5,
    )
    method._distill_stages = [stage]
    state_batch = SimpleNamespace(
        latents=torch.zeros(1, 5, 2, 1, 1),
        timesteps=torch.zeros(1, 5),
        attn_metadata_vsa=object(),
    )
    state = SimpleNamespace(batch=state_batch, current_length=3)
    pred_x0 = torch.ones_like(state_batch.latents)
    chunk_mask = torch.ones(1, 5, 1, 1, 1, dtype=torch.bool)
    chunk_info = _StreamingChunkInfo(
        chunk_start=0,
        chunk_end=4,
        train_start=0,
        train_end=4,
        new_frames=5,
        overlap=0,
        current_length=3,
        max_length=5,
    )
    logs: list[_StreamingChunkInfo] = []

    method._should_update_student = lambda iteration: False
    method._ensure_streaming_state = lambda batch, stage: state
    method._generate_streaming_chunk = lambda state, with_grad: (
        pred_x0,
        chunk_mask,
        chunk_info,
    )
    method._critic_flow_matching_loss_for_x0 = (
        lambda generator_pred_x0, batch, *, chunk_mask: (
            torch.tensor(2.0),
            "critic_ctx",
            {"generator_pred_video": generator_pred_x0},
        )
    )
    method._log_train_chunk = (
        lambda *, iteration, stage, update_student, chunk_info: logs.append(chunk_info)
    )
    method._stage_max_length = lambda stage: 5

    loss_map, outputs, metrics = method.single_train_step({}, iteration=0)

    assert loss_map["total_loss"].item() == 2.0
    assert loss_map["generator_loss"].item() == 0.0
    assert loss_map["fake_score_loss"].item() == 2.0
    assert torch.equal(outputs["generator_pred_video"], pred_x0)
    assert outputs["generator_pred_video"].data_ptr() == pred_x0.data_ptr()
    assert outputs["_fv_backward"] == {
        "update_student": False,
        "student_ctx": None,
        "critic_ctx": "critic_ctx",
    }
    assert metrics == {
        "update_student": 0.0,
        "distill_stage_index": 0.0,
        "streaming_current_length": 3.0,
        "streaming_max_length": 5.0,
    }
    assert logs == [chunk_info]
