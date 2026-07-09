# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``LTX2Model`` + ``FineTuneMethod``.

Mirrors ``test_wan_finetune.py`` for the LTX-2 plugin, parametrized
over LTX-2.0 and LTX-2.3 checkpoints. LTX2-specific differences: the
synthetic ``raw_batch`` carries a single post-connector Gemma
embedding (3840-d for 2.0, 4096-d for 2.3 which has no in-DiT caption
projection) and 128-channel VAE latents; the DiT is 18.9B params, so
the test skips on GPUs with less than 60GB memory (e.g. the L40S CI
runner).
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
from fastvideo.train.models.ltx2 import LTX2Model
from fastvideo.train.utils.config import load_run_config

from .grad_norm_regression import (
    check_grad_norm_regression,
    resolve_blocks,
)

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# id -> (fixture, post-connector text embedding width, grad-norm ref name)
_CASES = {
    "ltx2": ("ltx2_t2v_finetune_min.yaml", 3840, "test_ltx2_finetune"),
    "ltx2_3": ("ltx2_3_t2v_finetune_min.yaml", 4096, "test_ltx2_3_finetune"),
}

_LTX2_TEXT_LEN = 1024

_MIN_GPU_MEMORY_GB = 60


def _gpu_too_small() -> bool:
    if not torch.cuda.is_available():
        return True
    total = torch.cuda.get_device_properties(0).total_memory
    return total < _MIN_GPU_MEMORY_GB * 1024**3


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
    text_dim: int,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic ``raw_batch`` matching ``LTX2Model.prepare_batch``.

    LTX-2 latents are 128-channel (temporal 8x / spatial 32x
    compression); text embeddings are post-connector Gemma features.
    """
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size,
                    _LTX2_TEXT_LEN,
                    text_dim,
                    device=device,
                    dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, _LTX2_TEXT_LEN, device=device),
        "vae_latent":
        torch.randn(batch_size, 128, 3, 8, 8, device=device, dtype=dtype),
    }


@pytest.mark.usefixtures("distributed_setup")
@pytest.mark.parametrize("case", _CASES.keys())
def test_ltx2_finetune_single_train_step(
        case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if _gpu_too_small():
        pytest.skip(f"requires a CUDA GPU with >= {_MIN_GPU_MEMORY_GB}GB "
                    "memory (LTX-2 DiT is 18.9B params)")

    fixture_name, text_dim, ref_name = _CASES[case]
    cfg = load_run_config(str(_FIXTURE_DIR / fixture_name))

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

    model = LTX2Model(
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

    batch = _build_synthetic_batch(device, dtype, text_dim)
    loss_map, outputs, _metrics = method.single_train_step(batch, iteration=0)

    loss = loss_map["total_loss"]
    assert torch.is_tensor(loss), "total_loss must be a torch.Tensor"
    assert torch.isfinite(loss).item(), (
        f"total_loss is not finite: {loss.item()}")

    method.backward(loss_map, outputs, grad_accum_rounds=1)

    # LTX-2 nests its transformer_blocks under the ``model`` submodule.
    blocks = resolve_blocks(model.transformer.model)
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

    # Audio / cross-modal parameters must be frozen by default.
    audio_trainable = [
        name for name, param in model.transformer.named_parameters()
        if param.requires_grad and any(
            pattern in name for pattern in ("audio", "a2v", "v2a", "av_ca"))
    ]
    assert not audio_trainable, (
        f"audio/cross-modal params unexpectedly trainable: "
        f"{audio_trainable[:5]}")

    # Device-keyed grad-norm regression on top of the same harness.
    # Skips when the current GPU has no seeded reference.
    check_grad_norm_regression(ref_name, model.transformer.model)


def _flashinfer_fp4_available() -> bool:
    try:
        from flashinfer import mm_fp4, nvfp4_quantize  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.usefixtures("distributed_setup")
def test_ltx2_nvfp4_qat_finetune_single_train_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Same harness as the dense test, with NVFP4 QAT enabled via
    ``pipeline.dit_config.quant_config: nvfp4_qat_train``. Additionally
    asserts the QAT quant method is actually attached to the expected
    attention/FFN linears."""
    if _gpu_too_small():
        pytest.skip(f"requires a CUDA GPU with >= {_MIN_GPU_MEMORY_GB}GB "
                    "memory (LTX-2 DiT is 18.9B params)")
    if not _flashinfer_fp4_available():
        pytest.skip("requires flashinfer with FP4 kernels")

    from fastvideo.layers.quantization.nvfp4_qat_train_config import (
        NVFP4QATTrainConfig,
        NVFP4QATTrainQuantizeMethod,
    )

    cfg = load_run_config(str(_FIXTURE_DIR / "ltx2_t2v_qat_finetune_min.yaml"))
    assert isinstance(cfg.training.pipeline_config.dit_config.quant_config,
                      NVFP4QATTrainConfig), (
                          "quant_config string was not resolved to "
                          "NVFP4QATTrainConfig by config parsing")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_t2v_train_dataloader",
        lambda *args, **kwargs: None,
    )

    model = LTX2Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    # 48 audio+video blocks x 28 substring-matched linears (attn1 4 +
    # attn2 4 + ffn 2, video and audio branches, + a2v 4 + v2a 4).
    quantized = sum(
        isinstance(getattr(m, "quant_method", None),
                   NVFP4QATTrainQuantizeMethod)
        for m in model.transformer.modules())
    assert quantized == 1344, (
        f"expected 1344 NVFP4-QAT linears on the LTX-2 DiT, found "
        f"{quantized}")

    method = FineTuneMethod(
        cfg=cfg,
        role_models={"student": model},
    )
    method.on_train_start()

    batch = _build_synthetic_batch(device, dtype, text_dim=3840)
    loss_map, outputs, _metrics = method.single_train_step(batch, iteration=0)

    loss = loss_map["total_loss"]
    assert torch.isfinite(loss).item(), (
        f"total_loss is not finite: {loss.item()}")

    method.backward(loss_map, outputs, grad_accum_rounds=1)

    blocks = resolve_blocks(model.transformer.model)
    assert blocks is not None and len(blocks) > 0
    trainable = [p for p in blocks[0].parameters() if p.requires_grad]
    assert trainable, "layer 0 has no trainable parameters"
    for i, p in enumerate(trainable):
        assert p.grad is not None, f"layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"layer 0 param[{i}] grad contains NaN/Inf")
    assert any(
        p.grad.detach().float().norm().item() > 0.0 for p in trainable), (
            "all layer-0 grads are exactly zero; STE backward did not "
            "reach the first transformer block")

    check_grad_norm_regression("test_ltx2_nvfp4_qat_finetune",
                               model.transformer.model)
