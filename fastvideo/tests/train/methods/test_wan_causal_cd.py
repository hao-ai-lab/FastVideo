# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``WanCausalModel`` + ``CausalConsistencyDistillationMethod``.

Mirrors ``test_wan_causal_dfsft.py``. Causal Consistency Distillation
(Causal-Forcing++ Stage-2b) bootstraps a consistency MSE between the
student's ``x0`` at ``t`` and an EMA copy of the student at ``t_next``,
where ``t_next`` is produced online by a single CFG Euler step of a frozen
teacher (all under clean-history teacher forcing). This test exercises the
full step: finite loss, nonzero student gradients, frozen teacher, and a
post-step EMA update.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.consistency_model.causal_cd import (
    CausalConsistencyDistillationMethod, )
from fastvideo.train.models.wan import WanCausalModel
from fastvideo.train.utils.config import load_run_config


_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_causal_t2v_causal_cd_min.yaml")


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


@pytest.mark.usefixtures("distributed_setup")
def test_wan_causal_cd_single_train_step(
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

    teacher = WanCausalModel(
        init_from=cfg.models["teacher"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )
    teacher.transformer = teacher.transformer.to(device=device, dtype=dtype)

    ema = WanCausalModel(
        init_from=cfg.models["ema"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )
    ema.transformer = ema.transformer.to(device=device, dtype=dtype)

    method = CausalConsistencyDistillationMethod(
        cfg=cfg,
        role_models={"student": student, "teacher": teacher, "ema": ema},
    )
    method.on_train_start()

    batch = _build_synthetic_batch(device, dtype)
    loss_map, outputs, _metrics = method.single_train_step(batch, iteration=0)

    loss = loss_map["total_loss"]
    assert torch.is_tensor(loss), "total_loss must be a torch.Tensor"
    assert torch.isfinite(loss).item(), (
        f"total_loss is not finite: {loss.item()}")

    method.backward(loss_map, outputs, grad_accum_rounds=1)

    blocks = student.transformer.blocks
    assert blocks is not None and len(blocks) > 0
    layer0 = blocks[0]
    trainable = [p for p in layer0.parameters() if p.requires_grad]
    assert len(trainable) > 0, "student layer 0 has no trainable parameters"
    for i, p in enumerate(trainable):
        assert p.grad is not None, f"student layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"student layer 0 param[{i}] grad contains NaN/Inf")
    assert any(p.grad.detach().float().norm().item() > 0.0 for p in trainable), (
        "all student layer-0 grads are exactly zero; consistency loss "
        "did not reach the first transformer block")

    # Teacher must stay frozen.
    assert all(not p.requires_grad for p in teacher.transformer.parameters()), (
        "teacher must be frozen for Causal-CD")

    # The EMA model and student start from the same checkpoint, so the first
    # parameter must match before any update. FSDP fully_shard params are
    # DTensors; compare the local shards (torch.equal is unsupported on
    # DTensor).
    def _local(p: torch.Tensor) -> torch.Tensor:
        return p.to_local() if hasattr(p, "to_local") else p

    ema_param = next(ema.transformer.parameters())
    student_param = next(student.transformer.parameters())
    assert torch.equal(_local(ema_param), _local(student_param)), (
        "EMA model should start identical to the student (same checkpoint)")

    # The EMA update must move EMA toward the student. Apply a visibly large
    # perturbation so the bf16 lerp is well above rounding noise (the real
    # optimizer step at lr=2e-6 would be sub-ULP in bf16).
    with torch.no_grad():
        student_param.add_(1.0)
    before = _local(ema_param).detach().float().clone()
    method._update_ema()
    after = _local(ema_param).detach().float()
    assert not torch.equal(before, after), (
        "EMA weights did not move after _update_ema")
    # EMA = decay*ema + (1-decay)*student moves ~ (1-decay) of the gap.
    expected = before + (1.0 - method._ema_decay) * (
        _local(student_param).detach().float() - before)
    assert torch.allclose(after, expected, atol=1e-2), (
        "EMA update did not follow the expected lerp")
