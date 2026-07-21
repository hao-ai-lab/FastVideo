# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``LTX2Model``.

Loads the real LTX-2.0 / LTX-2.3 distilled checkpoints into the 13B
video-only training transformer (skips on GPUs with less than 60GB memory) via
``LTX2Model.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.ltx2.LTX2Model`` and the underlying
``LTX2VideoOnlyTransformer3DModel``.

LTX-2's transformer takes per-token sigma timesteps in [0, 1] shaped
[B, tokens] and a post-connector Gemma text embedding (3840-d for 2.0;
4096-d for 2.3, which has no in-DiT caption projection); it returns
the denoised x0 prediction. This mirrors the kwargs in
``LTX2Model._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import (
    get_forward_context,
    set_forward_context,
)
from fastvideo.models.dits.ltx2 import LTXModelType
from fastvideo.pipelines import ForwardBatch, TrainingBatch
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.ltx2 import LTX2Model
from fastvideo.train.models.ltx2 import ltx2 as ltx2_module
from fastvideo.train.utils.config import load_run_config

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# id -> (fixture, post-connector text embedding width)
_CASES = {
    "ltx2": ("ltx2_t2v_finetune_min.yaml", 3840),
    "ltx2_3": ("ltx2_3_t2v_finetune_min.yaml", 4096),
}

_LTX2_TEXT_LEN = 1024

_MIN_GPU_MEMORY_GB = 60


class _TinyLTX2Transformer(torch.nn.Module):
    """Small stand-in that preserves the LTX-2 wrapper contract."""

    def __init__(self, *, is_ltx2_3: bool) -> None:
        super().__init__()
        self.video_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = type("Config", (), {
            "arch_config": type("Arch", (), {
                "caption_proj_before_connector": is_ltx2_3,
                "cross_attention_dim": 4096,
                "caption_channels": 3840,
            })(),
        })()
        self.forward_fps: float | None = None
        self.forward_timestep: torch.Tensor | None = None
        self.forward_text_dim: int | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        self.forward_fps = get_forward_context().forward_batch.fps
        self.forward_timestep = timestep
        self.forward_text_dim = encoder_hidden_states.shape[-1]
        sigma = timestep[:, 0].view(-1, 1, 1, 1, 1)
        return hidden_states.float() - 2.0 * sigma


def _gpu_too_small() -> bool:
    if not torch.cuda.is_available():
        return True
    total = torch.cuda.get_device_properties(0).total_memory
    return total < _MIN_GPU_MEMORY_GB * 1024**3


def test_ltx2_rejects_audio_training_before_loading(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_run_config(str(_FIXTURE_DIR / _CASES["ltx2"][0]))

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("transformer loading must not start")

    monkeypatch.setattr(LTX2Model, "_load_transformer", fail_if_loaded)
    with pytest.raises(NotImplementedError, match="train_audio=True"):
        LTX2Model(
            init_from=cfg.models["student"]["init_from"],
            training_config=cfg.training,
            train_audio=True,
        )


def test_ltx2_selects_video_only_transformer_and_forwards_backend(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_run_config(str(_FIXTURE_DIR / _CASES["ltx2"][0]))
    transformer = _TinyLTX2Transformer(is_ltx2_3=False)
    captured: dict[str, object] = {}

    def fake_load_module_from_path(**kwargs):
        captured.update(kwargs)
        return transformer

    monkeypatch.setattr(ltx2_module, "load_module_from_path",
                        fake_load_module_from_path)
    model = LTX2Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
        attention_backend="TORCH_SDPA",
    )

    assert model.transformer is transformer
    assert captured["override_transformer_cls_name"] == (
        "LTX2VideoOnlyTransformer3DModel")
    assert captured["attention_backend"] is AttentionBackendEnum.TORCH_SDPA


def test_ltx2_rejects_role_local_sparse_attention_backend() -> None:
    model = object.__new__(LTX2Model)
    model.attention_backend = AttentionBackendEnum.VIDEO_SPARSE_ATTN

    with pytest.raises(NotImplementedError, match="does not support VSA/VMOBA"):
        model._build_attention_metadata(TrainingBatch())


@pytest.mark.parametrize("case", _CASES.keys())
def test_ltx2_wrapper_contract_runs_on_cpu(
        case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_name, text_dim = _CASES[case]
    cfg = load_run_config(str(_FIXTURE_DIR / fixture_name))
    transformer = _TinyLTX2Transformer(is_ltx2_3=case == "ltx2_3")
    monkeypatch.setattr(
        LTX2Model,
        "_load_transformer",
        lambda *_args, **_kwargs: transformer,
    )
    monkeypatch.setattr(
        "fastvideo.train.models.base.get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    model = LTX2Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
        timestep_uniform_prob=0.0,
    )

    assert transformer.video_weight.requires_grad

    model._check_text_embedding_dim(torch.empty(1, 1, text_dim))
    with pytest.raises(ValueError, match="text_embedding width"):
        model._check_text_embedding_dim(torch.empty(1, 1, text_dim + 1))

    raw_latents = torch.randn(1, 128, 2, 2, 2)
    batch = model.prepare_batch(
        {
            "text_embedding": torch.randn(1, 4, text_dim),
            "text_attention_mask": torch.ones(1, 4),
            "vae_latent": raw_latents,
            "info_list": [{"fps": 12.0}],
        },
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert batch.sigmas is not None
    assert batch.timesteps is not None
    assert batch.noisy_model_input is not None
    assert torch.all((batch.sigmas >= 0.0) & (batch.sigmas <= 1.0))
    assert torch.allclose(batch.timesteps, batch.sigmas.flatten() * 1000.0)
    expected_noisy = ((1.0 - batch.sigmas) * raw_latents.bfloat16().float() +
                      batch.sigmas * batch.noise.float()).bfloat16()
    assert torch.equal(batch.noisy_model_input, expected_noisy)

    outer_batch = ForwardBatch(data_type="video", fps=30)
    with set_forward_context(
            current_timestep=torch.tensor([-1.0]),
            attn_metadata=None,
            forward_batch=outer_batch,
    ):
        velocity = model.predict_noise(
            batch.noisy_model_input.permute(0, 2, 1, 3, 4),
            batch.timesteps,
            batch,
            conditional=True,
        )
        assert get_forward_context().forward_batch is outer_batch

    assert velocity.shape == (1, 2, 128, 2, 2)
    assert velocity.device.type == "cpu"
    assert velocity.dtype == torch.float32
    assert torch.allclose(velocity, torch.full_like(velocity, 2.0), atol=1e-3)
    assert transformer.forward_fps == 12.0
    assert transformer.forward_text_dim == text_dim
    assert transformer.forward_timestep is not None
    assert transformer.forward_timestep.shape == (1, 8)
    assert torch.allclose(
        transformer.forward_timestep[:, 0],
        batch.timesteps / 1000.0,
    )

    backward_fps: list[float] = []
    transformer.video_weight.register_hook(
        lambda grad: backward_fps.append(
            get_forward_context().forward_batch.fps) or grad)
    with set_forward_context(
            current_timestep=torch.tensor([-1.0]),
            attn_metadata=None,
            forward_batch=outer_batch,
    ):
        model.backward(
            transformer.video_weight.square(),
            (batch.timesteps, None),
            grad_accum_rounds=2,
        )
        assert get_forward_context().forward_batch is outer_batch
    assert backward_fps == [12.0]
    assert transformer.video_weight.grad.item() == pytest.approx(1.0)


@pytest.mark.usefixtures("distributed_setup")
@pytest.mark.parametrize("case", _CASES.keys())
def test_ltx2_model_loads_and_forwards(case: str):
    if _gpu_too_small():
        pytest.skip(f"requires a CUDA GPU with >= {_MIN_GPU_MEMORY_GB}GB "
                    "memory (LTX-2 video DiT is 13B params)")

    fixture_name, text_dim = _CASES[case]
    cfg = load_run_config(str(_FIXTURE_DIR / fixture_name))
    model = LTX2Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )

    transformer = model.transformer
    assert isinstance(transformer, torch.nn.Module)
    assert sum(p.numel() for p in transformer.parameters()) > 0
    assert transformer.model.model_type is LTXModelType.VideoOnly
    assert not hasattr(transformer.model, "audio_patchify_proj")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    transformer = transformer.to(device=device, dtype=dtype).eval()

    # LTX-2 transformer takes [B, 128, T, H, W] latents, a post-connector
    # Gemma embedding, and PER-TOKEN sigmas in [0, 1] ([B, T*H*W] with
    # patch size 1x1x1). Small spatial + few frames so this fits next to
    # the 13B model.
    b, c, t, h, w = 1, 128, 3, 8, 8
    tokens = t * h * w
    hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(b,
                                        _LTX2_TEXT_LEN,
                                        text_dim,
                                        device=device,
                                        dtype=dtype)
    timestep = torch.full((b, tokens), 0.5, device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast(device.type, dtype=dtype), \
            set_forward_context(
                current_timestep=timestep * 1000.0,
                attn_metadata=None,
                forward_batch=ForwardBatch(data_type="video", fps=24.0),
            ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=None,
            return_dict=False,
        )

    assert torch.is_tensor(out), f"expected a tensor output, got {type(out)}"
    assert out.shape == hidden_states.shape, (
        f"denoised output shape {tuple(out.shape)} != input latent shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "forward output contains NaN/Inf"
