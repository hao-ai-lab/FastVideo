# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``MatrixGame2Model`` + ``FineTuneMethod``.

Mirrors ``test_wan_finetune.py`` for the Matrix-Game 2.0 plugin, which
is an action-conditioned I2V world model (no text encoder). The
synthetic ``raw_batch`` therefore supplies ``vae_latent`` +
``clip_feature`` (CLIP image embeds, ``image_dim=1280``) +
``first_frame_latent`` instead of text embeddings, plus all-ones
mouse/keyboard action conditioning so block 0's action modules also
receive gradients. The goal is to validate the training grad path
(forward + chain rule reaches block 0), not output realism.

Matrix-Game-2.0-Base is ~14B at bf16. A single-GPU backward (params +
grads, no FSDP) needs well over the L40S CI runner's 48 GB, so this
test skips on memory-constrained GPUs and only exercises the grad path
on larger dev GPUs (H200 / GB200). LongCat (13.6B) is likewise covered
by a loading-only smoke for the same reason.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29522")

from pathlib import Path

import pytest
import torch

from fastvideo.train.methods.fine_tuning.finetune import (
    FineTuneMethod, )
from fastvideo.train.models.matrixgame2.matrixgame2 import MatrixGame2Model
from fastvideo.train.utils.config import load_run_config

from .grad_norm_regression import (
    check_grad_norm_regression,
    resolve_blocks,
)

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "matrixgame2_finetune_min.yaml")

# Matrix-Game 2.0 CLIP image-embedding width.
_MG2_IMAGE_DIM = 1280

# A 14B bf16 backward (params + grads) needs ~56 GB before activations;
# skip on GPUs that can't hold it (e.g. the 48 GB L40S CI runner).
_MIN_GPU_MEM_BYTES = 60 * (1024**3)


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic ``raw_batch`` matching ``MatrixGame2Model.prepare_batch``.

    Matrix-Game 2.0 conditions on the first-frame latent + CLIP image
    embeds (no text). ``first_frame_latent`` carries 16 VAE channels;
    the model builds the mask/cond concat internally.
    """
    batch_size = 1
    # Action conditioning frames = (num_latent_t - 1) * vae_time_compression
    # + 1 = (4 - 1) * 4 + 1 = 13 (MatrixGame2Model._expected_action_frames).
    # Supplying actions (keyboard_dim_in=6, mouse_dim_in=2 in the loaded
    # checkpoint) exercises the image-action cross-attention so block 0's
    # action-module params receive gradients.
    action_frames = 13
    return {
        "vae_latent":
        torch.randn(batch_size, 16, 4, 8, 8, device=device, dtype=dtype),
        "clip_feature":
        torch.randn(batch_size, 257, _MG2_IMAGE_DIM, device=device,
                    dtype=dtype),
        "first_frame_latent":
        torch.randn(batch_size, 16, 4, 8, 8, device=device, dtype=dtype),
        "keyboard_cond":
        torch.ones(batch_size, action_frames, 6, device=device, dtype=dtype),
        "mouse_cond":
        torch.ones(batch_size, action_frames, 2, device=device, dtype=dtype),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_matrixgame2_finetune_single_train_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    total_mem = torch.cuda.get_device_properties(0).total_memory
    if total_mem < _MIN_GPU_MEM_BYTES:
        pytest.skip(
            f"Matrix-Game 2.0 (~14B) backward needs ~56 GB; this GPU has "
            f"{total_mem / 1024**3:.0f} GB. Loading is covered by "
            "test_load_matrixgame2.py; the grad path runs on larger dev GPUs.")

    cfg = load_run_config(_FIXTURE)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Matrix-Game 2.0 uses its own parquet dataloader; stub it out so
    # construction does not require a real ``training.data.data_path``.
    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_matrixgame2_train_dataloader",
        lambda *args, **kwargs: None,
    )

    model = MatrixGame2Model(
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

    named = [(n, p) for n, p in layer0.named_parameters() if p.requires_grad]
    assert len(named) > 0, "layer 0 has no trainable parameters"

    # Matrix-Game 2.0 is text-free I2V: block 0's text cross-attention
    # (``attn2.*`` and its residual norm) never activates without text
    # conditioning, so those params legitimately get no grad. Assert the
    # *active* path (self-attn, ffn, image cross-attn, action modules) is
    # intact rather than requiring every block-0 param to receive a grad.
    with_grad = [(n, p) for n, p in named if p.grad is not None]
    assert len(with_grad) > 0, (
        "no layer-0 params received a gradient; backward did not reach "
        "the first transformer block")

    for n, p in with_grad:
        assert torch.isfinite(p.grad).all().item(), (
            f"layer 0 param '{n}' grad contains NaN/Inf")

    any_nonzero = any(
        p.grad.detach().float().norm().item() > 0.0 for _, p in with_grad)
    assert any_nonzero, (
        "all layer-0 grads are exactly zero; backward did not "
        "reach the first transformer block")

    # 5a-ii: device-keyed grad-norm regression on top of the same harness.
    # Skips when the current GPU has no seeded reference.
    check_grad_norm_regression("test_matrixgame2_finetune", model.transformer)
