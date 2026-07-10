# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: frame-wise ``WanCausalModel`` + ``DiffusionForcingSFTMethod``.

Mirrors ``test_wan_causal_dfsft.py`` but with a block size of 1 frame
(``num_frames_per_block=1`` on the model, ``chunk_size=1`` on the method),
so each frame gets its own independent noise level. The test asserts the
override took effect and runs one train step: forward, finite loss, and
nonzero gradients reaching the first transformer block.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.dfsft import (
    DiffusionForcingSFTMethod, )
from fastvideo.train.models.wan import WanCausalModel
from fastvideo.train.utils.config import load_run_config


_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_causal_t2v_dfsft_framewise_min.yaml")


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
def test_wan_causal_dfsft_framewise_single_train_step(
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

    student_cfg = cfg.models["student"]
    model = WanCausalModel(
        init_from=student_cfg["init_from"],
        training_config=cfg.training,
        trainable=True,
        num_frames_per_block=student_cfg.get("num_frames_per_block"),
    )
    assert model.transformer.num_frame_per_block == 1, (
        "frame-wise override did not reach the transformer")
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    method = DiffusionForcingSFTMethod(
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
    assert blocks is not None and len(blocks) > 0
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
