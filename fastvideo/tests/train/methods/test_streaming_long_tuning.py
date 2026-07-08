# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``WanCausalModel`` + ``StreamingLongTuningMethod``.

Mirrors ``test_wan_causal_cd.py``. LongLive-style streaming long tuning runs a
short self-forcing stage first, then switches to a persistent streaming
sequence trained chunk-by-chunk with the causal student KV cache. The GPU test
walks a single method instance through both stages: the short-stage step, the
first streaming chunk, an overlapped chunk, the chunk that fills the sequence,
and the post-reset chunk, asserting finite losses, student/critic gradient
flow, and the streaming-length bookkeeping at every step.

The pure-tensor helpers (stage selection, schedule validation, first-frame
padding, latent truncation, chunk sizing) are covered by CPU-only tests at the
bottom — they need no GPU, no mocks, and run wherever this file is collected.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29523")

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.methods.distribution_matching.streaming_long_tuning import (
    DistillStage,
    StreamingLongTuningMethod,
    parse_multi_phased_distill_schedule,
    select_distill_stage,
)
from fastvideo.train.models.wan import WanCausalModel, WanModel
from fastvideo.train.utils.config import load_run_config

from .grad_norm_regression import check_grad_norm_regression

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_causal_t2v_streaming_long_min.yaml")


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size, 16, 4096, device=device, dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, 16, device=device),
        "vae_latent":
        torch.randn(batch_size, 16, 6, 8, 8, device=device, dtype=dtype),
    }


def _assert_layer0_grads(model, *, role: str) -> None:
    blocks = model.transformer.blocks
    assert blocks is not None and len(blocks) > 0
    trainable = [p for p in blocks[0].parameters() if p.requires_grad]
    assert len(trainable) > 0, f"{role} layer 0 has no trainable parameters"
    for i, p in enumerate(trainable):
        assert p.grad is not None, f"{role} layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"{role} layer 0 param[{i}] grad contains NaN/Inf")
    assert any(
        p.grad.detach().float().norm().item() > 0.0 for p in trainable), (
            f"all {role} layer-0 grads are exactly zero; the loss did not "
            "reach the first transformer block")


def _assert_finite_losses(loss_map: dict[str, torch.Tensor]) -> None:
    for name in ("total_loss", "generator_loss", "fake_score_loss"):
        loss = loss_map[name]
        assert torch.is_tensor(loss), f"{name} must be a torch.Tensor"
        assert torch.isfinite(loss).item(), (
            f"{name} is not finite: {loss.item()}")


@pytest.mark.usefixtures("distributed_setup")
def test_streaming_long_tuning_multi_stage_train_steps(
        monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_t2v_train_dataloader",
        lambda *args, **kwargs: None,
    )

    student = WanCausalModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    student.transformer = student.transformer.to(device=device, dtype=dtype)

    teacher = WanModel(
        init_from=cfg.models["teacher"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )
    teacher.transformer = teacher.transformer.to(device=device, dtype=dtype)

    critic = WanModel(
        init_from=cfg.models["critic"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    critic.transformer = critic.transformer.to(device=device, dtype=dtype)

    method = StreamingLongTuningMethod(
        cfg=cfg,
        role_models={
            "student": student,
            "teacher": teacher,
            "critic": critic,
        },
    )
    method.on_train_start()

    batch = _build_synthetic_batch(device, dtype)

    # Iteration 0 — short self-forcing stage (stage 0, num_latent_t=3).
    loss_map, outputs, metrics = method.single_train_step(dict(batch), 0)
    _assert_finite_losses(loss_map)
    assert metrics["distill_stage_index"] == 0.0
    assert metrics["active_num_latent_t"] == 3.0
    assert metrics["update_student"] == 1.0
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    _assert_layer0_grads(student, role="student")
    _assert_layer0_grads(critic, role="critic")
    assert all(not p.requires_grad
               for p in teacher.transformer.parameters()), (
                   "teacher must stay frozen")

    # Iteration 1 — first streaming chunk: 3 fresh latents, no overlap.
    method.optimizers_zero_grad(1)
    loss_map, outputs, metrics = method.single_train_step(dict(batch), 1)
    _assert_finite_losses(loss_map)
    assert metrics["distill_stage_index"] == 1.0
    assert metrics["streaming_current_length"] == 3.0
    assert metrics["streaming_max_length"] == 6.0
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    _assert_layer0_grads(student, role="student")
    _assert_layer0_grads(critic, role="critic")
    vis = outputs["dmd_latent_vis_dict"]
    assert int(vis["streaming_chunk_mask"].sum().item()) == 3

    # Iteration 2 — overlapped chunk: 2 new latents + 1 overlap anchor.
    method.optimizers_zero_grad(2)
    loss_map, outputs, metrics = method.single_train_step(dict(batch), 2)
    _assert_finite_losses(loss_map)
    assert metrics["streaming_current_length"] == 5.0
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    mask = outputs["dmd_latent_vis_dict"]["streaming_chunk_mask"]
    assert mask.dtype == torch.bool
    assert int(mask.sum().item()) == 2, (
        "overlap latents must be masked out of the training chunk")

    # Iteration 3 — final chunk fills the sequence (1 new latent).
    method.optimizers_zero_grad(3)
    loss_map, outputs, metrics = method.single_train_step(dict(batch), 3)
    _assert_finite_losses(loss_map)
    assert metrics["streaming_current_length"] == 6.0
    method.backward(loss_map, outputs, grad_accum_rounds=1)

    # Iteration 4 — the exhausted sequence resets and a fresh chunk starts.
    method.optimizers_zero_grad(4)
    loss_map, outputs, metrics = method.single_train_step(dict(batch), 4)
    _assert_finite_losses(loss_map)
    assert metrics["streaming_current_length"] == 3.0, (
        "streaming state must reset once the sequence reaches max length")
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    _assert_layer0_grads(student, role="student")

    # Grad-norm regression on the post-reset chunk (grads zeroed at the top
    # of this iteration, so the reference captures exactly one step). Keep
    # this last: under FASTVIDEO_GRADNORM_UPDATE=1 it records and skips.
    check_grad_norm_regression(
        "test_streaming_long_tuning",
        student.transformer,
    )


# ---------------------------------------------------------------------------
# CPU-only helper tests (no GPU, no mocks)
# ---------------------------------------------------------------------------


def _method_without_init() -> StreamingLongTuningMethod:
    # Bare instance for exercising stateless helpers; no attributes beyond
    # the ones each test sets are touched.
    return StreamingLongTuningMethod.__new__(StreamingLongTuningMethod)


def _make_stage(**overrides) -> DistillStage:
    defaults = dict(
        name="streaming_long",
        start_step=0,
        end_step=None,
        num_latent_t=6,
        streaming_training=True,
        streaming_chunk_size=3,
        streaming_max_length=6,
    )
    defaults.update(overrides)
    return DistillStage(**defaults)


class TestStageSelection:

    def test_boundaries_between_stages(self) -> None:
        short = _make_stage(
            name="self_forcing",
            start_step=0,
            end_step=10,
            streaming_training=False,
            streaming_chunk_size=None,
            streaming_max_length=None,
        )
        long = _make_stage(start_step=10)
        stages = [short, long]

        assert select_distill_stage(stages, 0) is short
        assert select_distill_stage(stages, 9) is short
        assert select_distill_stage(stages, 10) is long
        assert select_distill_stage(stages, 10_000) is long

    def test_past_last_closed_stage_falls_back_to_last(self) -> None:
        short = _make_stage(
            name="self_forcing",
            start_step=0,
            end_step=10,
            streaming_training=False,
            streaming_chunk_size=None,
            streaming_max_length=None,
        )
        assert select_distill_stage([short], 99) is short


class TestScheduleParsing:

    def test_rejects_overlapping_stages(self) -> None:
        with pytest.raises(ValueError, match="ordered and non-overlapping"):
            parse_multi_phased_distill_schedule(
                [
                    {"stage": "self_forcing", "start_step": 0, "end_step": 10},
                    {"stage": "streaming_long", "start_step": 5},
                ],
                default_num_latent_t=6,
                default_streaming_chunk_size=3,
            )

    def test_rejects_fixed_overlap_not_smaller_than_chunk(self) -> None:
        with pytest.raises(ValueError, match="smaller than"):
            parse_multi_phased_distill_schedule(
                [{
                    "stage": "streaming_long",
                    "streaming_chunk_size": 3,
                    "streaming_fixed_overlap_latents": 3,
                }],
                default_num_latent_t=6,
            )


class TestPadFirstFrameLatent:

    def test_pads_short_condition_with_zeros(self) -> None:
        method = _method_without_init()
        first_frame = torch.ones(1, 4, 1, 2, 2)

        batch = method._with_padded_first_frame_latent(
            {"first_frame_latent": first_frame},
            target_latent_frames=3,
        )

        padded = batch["first_frame_latent"]
        assert padded.shape == (1, 4, 3, 2, 2)
        assert torch.equal(padded[:, :, :1], first_frame)
        assert padded[:, :, 1:].abs().sum().item() == 0.0

    def test_passthrough_without_condition_or_when_long_enough(self) -> None:
        method = _method_without_init()
        raw: dict = {"other": 1}
        assert method._with_padded_first_frame_latent(
            raw, target_latent_frames=3) is raw

        full = {"first_frame_latent": torch.zeros(1, 4, 3, 2, 2)}
        assert method._with_padded_first_frame_latent(
            full, target_latent_frames=3) is full

    def test_rejects_non_5d_condition(self) -> None:
        method = _method_without_init()
        with pytest.raises(ValueError, match=r"\[B, C, T, H, W\]"):
            method._with_padded_first_frame_latent(
                {"first_frame_latent": torch.zeros(4, 1, 2, 2)},
                target_latent_frames=3,
            )


class TestTruncateBatchLatents:

    def test_slices_every_temporal_field(self) -> None:
        method = _method_without_init()
        # No _build_attention_metadata on the student → metadata rebuild is
        # a no-op, so only the tensor slicing is exercised.
        method.student = object()
        batch = SimpleNamespace(
            latents=torch.randn(1, 6, 2, 4, 4),
            noise_latents=torch.randn(1, 6, 2, 4, 4),
            noisy_model_input=torch.randn(1, 6, 2, 4, 4),
            noise=torch.randn(1, 6, 2, 4, 4),
            timesteps=torch.zeros(1, 6),
            sigmas=torch.linspace(0, 1, 6),
            raw_latent_shape=(1, 6, 2, 4, 4),
            attn_metadata=None,
            attn_metadata_vsa=None,
        )

        method._truncate_batch_latents(batch, 3)

        assert batch.latents.shape == (1, 3, 2, 4, 4)
        assert batch.noise_latents.shape == (1, 3, 2, 4, 4)
        assert batch.noisy_model_input.shape == (1, 3, 2, 4, 4)
        assert batch.noise.shape == (1, 3, 2, 4, 4)
        assert batch.timesteps.shape == (1, 3)
        assert batch.sigmas.shape == (3, )
        assert batch.raw_latent_shape == (1, 2, 3, 4, 4)


class TestStageChunkSize:

    def test_prefers_stage_then_config_then_rollout_block(self) -> None:
        method = _method_without_init()
        method.method_config = {"streaming_chunk_size": 5}
        method._chunk_size = 7

        assert method._stage_chunk_size(_make_stage()) == 3
        assert method._stage_chunk_size(
            _make_stage(streaming_chunk_size=None)) == 5

        method.method_config = {}
        assert method._stage_chunk_size(
            _make_stage(streaming_chunk_size=None)) == 7

    def test_rejects_non_positive_chunk_size(self) -> None:
        method = _method_without_init()
        method.method_config = {}
        method._chunk_size = 0
        with pytest.raises(ValueError, match="must be positive"):
            method._stage_chunk_size(_make_stage(streaming_chunk_size=None))
