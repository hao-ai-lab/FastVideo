# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``CosmosModel`` + ``FineTuneMethod``.

Mirrors ``test_wan_finetune.py`` for the Cosmos-Predict2.5 plugin.
The harness is intentionally identical so the tests are easy to
compare; the only Cosmos-specific differences are in the synthetic
``raw_batch`` (a single Reason1 text embedding with width
``crossattn_proj_in_channels=100352`` instead of Wan's T5 4096) and
the fixture (``training_cfg_rate=0`` is required, and
``precondition_outputs=false`` for velocity prediction).
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29521")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.finetune import (
    FineTuneMethod, )
from fastvideo.train.models.cosmos import CosmosModel
from fastvideo.train.utils.config import load_run_config

from .grad_norm_regression import (
    check_grad_norm_regression,
    resolve_blocks,
)

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "cosmos_t2w_finetune_min.yaml")

# Cosmos 2.5 Reason1 (Qwen2.5-VL) text embedding width.
_COSMOS_TEXT_DIM = 100352


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic ``raw_batch`` matching ``CosmosModel.prepare_batch``.

    Cosmos uses the Wan-family VAE (``z_dim=16``) but a single Reason1
    text encoder, so ``text_embedding`` is 100352-wide.
    """
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size, 16, _COSMOS_TEXT_DIM, device=device,
                    dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, 16, device=device),
        "vae_latent":
        torch.randn(batch_size, 16, 4, 8, 8, device=device, dtype=dtype),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos_finetune_single_train_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Feed a synthetic ``raw_batch`` straight into ``single_train_step``,
    # so the parquet train dataloader built by ``init_preprocessors`` is
    # never iterated. Stub it out so construction does not require a real
    # ``training.data.data_path``.
    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_t2v_train_dataloader",
        lambda *args, **kwargs: None,
    )

    model = CosmosModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    method = FineTuneMethod(
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

    blocks = resolve_blocks(model.transformer)
    assert blocks is not None and len(blocks) > 0, (
        "transformer is expected to expose a non-empty block list")
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

    # 5a-ii: device-keyed grad-norm regression on top of the same harness.
    # Skips when the current GPU has no seeded reference.
    check_grad_norm_regression("test_cosmos_finetune", model.transformer)
