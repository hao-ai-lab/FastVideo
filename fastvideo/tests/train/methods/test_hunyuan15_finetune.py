# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``Hunyuan15Model`` + ``FineTuneMethod``.

Mirrors ``test_cosmos_finetune.py`` for the HunyuanVideo 1.5 plugin.
The harness is intentionally identical so the tests are easy to compare;
the HY1.5-specific differences are in the synthetic ``raw_batch`` (dual
text embeddings -- Qwen 3584 plus ByT5 1472 -- and 32-channel latents)
and the fixture (``precondition_outputs=true``).

The empty-ByT5 case is covered as a second parametrised run: captions
without glyph text carry a zero-token ByT5 stream, which must survive
``prepare_batch``'s trim and reach the transformer without producing
NaNs.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29523")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.finetune import (
    FineTuneMethod, )
from fastvideo.train.models.hunyuan15 import Hunyuan15Model
from fastvideo.train.utils.config import load_run_config

from .grad_norm_regression import (
    check_grad_norm_regression,
    resolve_blocks,
)

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "hunyuan15_t2v_finetune_min.yaml")

_QWEN_DIM = 3584
_BYT5_DIM = 1472


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
    byt5_tokens: int,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic ``raw_batch`` matching ``Hunyuan15Model.prepare_batch``.

    HY1.5 stores dual text embeddings and 32-channel VAE latents.
    ``byt5_tokens=0`` reproduces a caption with no glyph text.
    """
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size, 20, _QWEN_DIM, device=device, dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, 20, device=device),
        "text_embedding_2":
        torch.randn(batch_size,
                    byt5_tokens,
                    _BYT5_DIM,
                    device=device,
                    dtype=dtype),
        "text_attention_mask_2":
        torch.ones(batch_size, byt5_tokens, device=device),
        "vae_latent":
        torch.randn(batch_size, 32, 4, 16, 16, device=device, dtype=dtype),
    }


@pytest.mark.parametrize("byt5_tokens", [8, 0], ids=["with_byt5", "empty_byt5"])
@pytest.mark.usefixtures("distributed_setup")
def test_hunyuan15_finetune_single_train_step(
    monkeypatch: pytest.MonkeyPatch,
    byt5_tokens: int,
) -> None:
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

    model = Hunyuan15Model(
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

    batch = _build_synthetic_batch(device, dtype, byt5_tokens)
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

    # Device-keyed grad-norm regression, same as the five sibling per-method
    # tests. Skips cleanly on a device with no seeded reference, so it costs
    # nothing until one exists.
    check_grad_norm_regression("test_hunyuan15_finetune", model.transformer)
