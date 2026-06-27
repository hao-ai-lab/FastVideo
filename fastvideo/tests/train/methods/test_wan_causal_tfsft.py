# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``WanCausalModel`` + ``TeacherForcingSFTMethod``.

Mirrors ``test_wan_causal_dfsft.py``. Teacher forcing concatenates a clean
context copy of every frame inside the causal transformer (``clean_x``) and
denoises the current block while attending to *clean* history. This test
exercises the ``clean_x`` path end-to-end: forward, finite loss, and nonzero
gradients reaching the first transformer block.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29518")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.tfsft import (
    TeacherForcingSFTMethod, )
from fastvideo.train.models.wan import WanCausalModel
from fastvideo.train.utils.config import load_run_config


_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_causal_t2v_tfsft_min.yaml")


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
def test_wan_causal_tfsft_single_train_step(
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

    model = WanCausalModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    method = TeacherForcingSFTMethod(
        cfg=cfg,
        role_models={"student": model},
    )
    method.on_train_start()

    batch = _build_synthetic_batch(device, dtype)
    loss_map, outputs, _metrics = method.single_train_step(batch, iteration=0)

    loss = loss_map["total_loss"]
    assert torch.is_tensor(loss), "total_loss must be a torch.Tensor"
    assert torch.isfinite(loss).item(), (
        f"total_loss is not finite: {loss.item()}")

    method.backward(loss_map, outputs, grad_accum_rounds=1)

    blocks = getattr(model.transformer, "blocks", None)
    assert blocks is not None and len(blocks) > 0, (
        "CausalWanTransformer is expected to expose ``.blocks``")
    layer0 = blocks[0]

    trainable = [p for p in layer0.parameters() if p.requires_grad]
    assert len(trainable) > 0, "layer 0 has no trainable parameters"

    for i, p in enumerate(trainable):
        assert p.grad is not None, f"layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"layer 0 param[{i}] grad contains NaN/Inf")

    any_nonzero = any(
        p.grad.detach().float().norm().item() > 0.0 for p in trainable)
    assert any_nonzero, (
        "all layer-0 grads are exactly zero; backward did not "
        "reach the first transformer block")

    # Teacher forcing must build its own (concatenated) attention mask and
    # must not have constructed the diffusion-forcing mask.
    assert model.transformer.teacher_forcing_block_mask is not None, (
        "teacher-forcing mask was not constructed")
    assert model.transformer.block_mask is None, (
        "diffusion-forcing mask should not be built on the TF path")
