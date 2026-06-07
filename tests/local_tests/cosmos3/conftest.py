# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Cosmos3 native-pipeline local tests.

These fixtures build the FastVideo-native Cosmos3 pipeline
(``fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline.Cosmos3OmniDiffusersPipeline``)
via ``__new__`` and wire it with tiny stub components so the runtime call graph
(sequential CFG, condition-frame masking, mode dispatch) can be exercised on CPU
without real weights or ``cosmos_framework``.

The stub transformer implements the native DiT's packed-input contract
(``{"preds_vision": [[1, C, T, H, W], ...]}``) and records, per call, the first
``text_ids`` token so tests can assert the cond/uncond pass order.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastvideo.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from torch import nn

_LATENT_CHANNEL = 16
_LATENT_PATCH_SIZE = 2
_SPATIAL_FACTOR = 8
_TEMPORAL_FACTOR = 4


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``local`` marker used by sibling test files."""
    config.addinivalue_line(
        "markers",
        "local: marker for local-only parity/scaffold tests (skipped in CI)",
    )


# ---------------------------------------------------------------------------
# Stub transformer: records cond/uncond call order; bounded preds_vision.
# ---------------------------------------------------------------------------
class StubCosmos3Transformer(nn.Module):
    """Records each forward's first ``text_ids`` token + returns preds_vision.

    ``preds_vision`` is keyed by the first text token (so the conditional and
    unconditional passes return different velocities) and is zero on
    conditioning frames, matching the real DiT's unpatchify output.
    """

    def __init__(self, latent_channel: int = _LATENT_CHANNEL) -> None:
        super().__init__()
        self.latent_channel = latent_channel
        self.embed_tokens = nn.Embedding(64, 8)
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        token_ids = kwargs["text_ids"]
        token = int(token_ids.reshape(-1)[0].item()) if token_ids.numel() else 0
        self.calls.append({"token": token, "kwargs": dict(kwargs)})
        scale = 0.01 * (1.0 + (token % 7))
        preds: list[torch.Tensor] = []
        for latent, _shape, nfi in zip(kwargs["vision_tokens"], kwargs["vision_token_shapes"],
                                       kwargs["vision_noisy_frame_indexes"]):
            lat = latent.squeeze(0) if latent.dim() == 5 else latent  # [C, T, H, W]
            out = torch.zeros_like(lat)
            if nfi.numel() > 0:
                out[:, nfi] = scale * torch.tanh(lat[:, nfi])
            preds.append(out.unsqueeze(0))
        return {"preds_vision": preds}


class _StubLatentDist:

    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    """Deterministic VAE shaped by the Wan scale factors."""

    def __init__(self, z_dim: int = _LATENT_CHANNEL) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=_TEMPORAL_FACTOR,
            scale_factor_spatial=_SPATIAL_FACTOR,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        b, _c, t, h, w = video.shape
        lt = (t - 1) // self.config.scale_factor_temporal + 1
        lh = h // self.config.scale_factor_spatial
        lw = w // self.config.scale_factor_spatial
        return _StubLatentDist(torch.ones(b, self.config.z_dim, lt, lh, lw, dtype=video.dtype, device=video.device))

    def decode(self, z: torch.Tensor):
        b, _c, lt, lh, lw = z.shape
        t = (lt - 1) * self.config.scale_factor_temporal + 1
        h = lh * self.config.scale_factor_spatial
        w = lw * self.config.scale_factor_spatial
        sig = torch.nan_to_num(torch.tanh(z[:, :1, :1, :1, :1])).reshape(b, 1, 1, 1, 1)
        return torch.clamp(torch.zeros(b, 3, t, h, w, dtype=z.dtype, device=z.device) + sig, -1.0, 1.0)


class StubQwen2Tokenizer:
    """Qwen2-shaped chat tokenizer stub (special tokens + chat template)."""

    eos_token_id = 62
    _SPECIAL = {"<|vision_start|>": 60, "<|vision_end|>": 61}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL[token]

    def apply_chat_template(self, conversations, *, tokenize=True, add_generation_prompt=True, add_vision_id=False):
        user = next((c["content"] for c in conversations if c["role"] == "user"), "")
        n = max(1, min(8, len(user) % 8 + 1))
        return [10 + (i % 40) for i in range(n)]


def make_scheduler(flow_shift: float = 10.0) -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=flow_shift,
    )


# ---------------------------------------------------------------------------
# Pipeline factory — builds the native pipeline via __new__ + stub modules.
# ---------------------------------------------------------------------------
@pytest.fixture
def make_cosmos3_pipeline():
    """Return a factory building the native Cosmos3 pipeline wired with stubs."""

    def _make(**overrides: Any):
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # noqa: F401
            Cosmos3OmniDiffusersPipeline, )

        pipe = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
        scheduler = make_scheduler()
        pipe.modules = {
            "transformer": StubCosmos3Transformer(),
            "vae": StubCosmos3VAE(),
            "scheduler": scheduler,
            "text_tokenizer": StubQwen2Tokenizer(),
        }
        pipe.scheduler = scheduler
        pipe._base_scheduler_config = scheduler.config
        pipe._current_flow_shift = float(scheduler.config.flow_shift)
        pipe._engine_init_flow_shift = 10.0
        for key, value in overrides.items():
            setattr(pipe, key, value)
        return pipe

    return _make


@pytest.fixture
def make_cosmos3_stage():
    """Return a factory building a ``Cosmos3DenoisingStage`` bound to a pipeline."""

    def _make(pipeline):
        from fastvideo.pipelines.stages.cosmos3_stages import Cosmos3DenoisingStage

        return Cosmos3DenoisingStage(
            transformer=pipeline.modules["transformer"],
            scheduler=pipeline.modules["scheduler"],
            vae=pipeline.modules["vae"],
            tokenizer=pipeline.modules["text_tokenizer"],
            pipeline=pipeline,
        )

    return _make


def make_forward_batch(*, num_frames: int, height: int, width: int, image: Any = None, **overrides: Any):
    """Build a tiny ``ForwardBatch`` for the Cosmos3 stage."""
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    values: dict[str, Any] = dict(
        data_type="video",
        prompt="a calm ocean at sunrise",
        negative_prompt="",
        height=height,
        width=width,
        num_frames=num_frames,
        fps=24,
        num_inference_steps=2,
        guidance_scale=6.0,
        generator=torch.Generator("cpu").manual_seed(0),
        preprocessed_image=image,
    )
    values.update(overrides)
    return ForwardBatch(**values)


def make_fastvideo_args():
    """Build minimal ``fastvideo_args`` (only ``pipeline_config`` is read)."""
    from fastvideo.configs.pipelines.cosmos3 import Cosmos3Config

    cfg = Cosmos3Config()
    arch = cfg.dit_config.arch_config
    arch.latent_channel = _LATENT_CHANNEL
    arch.latent_patch_size = _LATENT_PATCH_SIZE
    arch.temporal_compression_factor = _TEMPORAL_FACTOR
    arch.enable_fps_modulation = False
    return SimpleNamespace(pipeline_config=cfg)
