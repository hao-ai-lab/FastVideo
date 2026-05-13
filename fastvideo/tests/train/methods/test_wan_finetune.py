# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``WanModel`` + ``FineTuneMethod``.

Establishes the per-method test pattern for ``fastvideo/train``:

1. Instantiate the model + method via their public constructors
   (no ``Trainer`` setup, no FSDP wrapping).
2. Feed a synthetic ``raw_batch`` dict through
   ``method.single_train_step()`` + ``method.backward()``.
3. Assert that the loss is finite and that the first transformer
   block received a finite, non-zero gradient.

The first block's gradient is the *last* one computed during
backprop, so a healthy grad there implies the full
forward + chain-rule path is intact.  Keeping the assertion to a
single block keeps the reference surface tiny — a later PR layers a
device-keyed grad-norm regression on top of this same harness.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29516")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.finetune import (
    FineTuneMethod, )
from fastvideo.train.models.wan import WanModel
from fastvideo.train.utils.config import load_run_config


_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_t2v_finetune_min.yaml")


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic ``raw_batch`` matching ``WanModel.prepare_batch``.

    Shapes are intentionally small so the 1.3B base model + one
    backward pass fits comfortably on a single L40S.  Wan uses T5
    text embeddings (hidden=4096) and a VAE with ``z_dim=16``.
    """
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size, 16, 4096, device=device, dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, 16, device=device, dtype=dtype),
        # vae_latent in (B, C, T, H, W); ``prepare_batch`` will
        # truncate T to ``training.data.num_latent_t``.
        "vae_latent":
        torch.randn(batch_size, 16, 4, 8, 8, device=device, dtype=dtype),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_wan_finetune_single_train_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    model = WanModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    # Move transformer to device + training dtype.  Real training
    # wraps the transformer with FSDP and shards across ranks; for a
    # single-step smoke we just move it directly.
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    method = FineTuneMethod(
        cfg=cfg.method,
        role_models={"student": model},
    )
    # cuda_generator + RNG seeding (normally done by ``Trainer``).
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
        "Wan transformer is expected to expose ``.blocks``")
    layer0 = blocks[0]

    trainable = [p for p in layer0.parameters() if p.requires_grad]
    assert len(trainable) > 0, "layer 0 has no trainable parameters"

    for i, p in enumerate(trainable):
        assert p.grad is not None, f"layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"layer 0 param[{i}] grad contains NaN/Inf")

    # At least one layer-0 grad must have non-zero L2 norm so we
    # catch the case where backward ran but the first block was
    # detached from the loss (silent connectivity bug).
    any_nonzero = any(
        p.grad.detach().float().norm().item() > 0.0 for p in trainable)
    assert any_nonzero, (
        "all layer-0 grads are exactly zero; backward did not "
        "reach the first transformer block")
