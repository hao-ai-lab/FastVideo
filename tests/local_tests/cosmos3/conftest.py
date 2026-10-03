# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Cosmos3 local parity tests (Phase 2a Tier A).

Ported from the vllm-omni reference suite so FastVideo's Tier A parity
coverage can land before NVIDIA publishes the real Cosmos3 weights.
The fixtures here are intentionally self-contained: they do NOT import
``vllm_omni`` and do NOT import any FastVideo Cosmos3 modules (those
do not exist yet; they are introduced in Phase 2b).

Reference:
- ``tests/diffusion/models/cosmos3/conftest.py`` lines 1-176 from
  ``vllm-omni`` HEAD ``8536f5b1421f78c7df06af6d96fa195c1ceb6384``.

The stubs reproduce the same name + protocol as the upstream fixtures
so test code can be ported with minimal renaming once FastVideo's
Cosmos3 pipeline class lands and a ``make_cosmos3_pipeline`` fixture
can be re-pointed at the real class.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn


def pytest_configure(config: pytest.Config) -> None:
    """Register ``local`` marker so ``pytestmark = [pytest.mark.local]``
    in sibling test files does not trigger ``PytestUnknownMarkWarning``.
    """
    config.addinivalue_line(
        "markers",
        "local: marker for local-only parity/scaffold tests (skipped in CI)",
    )


# ---------------------------------------------------------------------------
# Stub scheduler — mirrors vllm-omni ``StubScheduler``.
# ---------------------------------------------------------------------------
class StubScheduler:
    """Minimal UniPC-shaped scheduler stub used by pipeline call-graph tests.

    Mirrors ``tests/diffusion/models/cosmos3/conftest.py:16-30`` from
    the vllm-omni reference: records ``set_timesteps`` and ``step`` calls,
    advances ``latents`` by ``+noise_pred``, and exposes a ``config``
    namespace carrying ``num_train_timesteps`` and ``flow_shift``.
    """

    def __init__(
        self,
        timesteps: list[int] | None = None,
        *,
        flow_shift: float = 1.0,
    ) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000, flow_shift=flow_shift)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64, device=device)

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        **kwargs: Any,
    ):
        del kwargs
        self.step_calls.append((noise_pred.clone(), timestep.clone(), latents.clone()))
        return (latents + noise_pred,)


class _ModeLatentDist:
    """Stub for diffusers ``DiagonalGaussianDistribution.mode``."""

    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    """VAE stub returning deterministic latents shaped by VAE scale factors.

    Mirrors ``tests/diffusion/models/cosmos3/conftest.py:41-70``.
    """

    dtype = torch.float32

    def __init__(self, z_dim: int = 2, *, temporal: int = 4, spatial: int = 8) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=temporal,
            scale_factor_spatial=spatial,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] - 1) // self.config.scale_factor_temporal + 1
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latent_dist=_ModeLatentDist(latents))

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


class StubCosmos3Transformer(nn.Module):
    """Transformer stub that records per-call inputs and emits deterministic
    output tensors keyed by the first ``text_ids`` token.

    Mirrors ``tests/diffusion/models/cosmos3/conftest.py:73-114``.

    The stub also exposes ``cached_kv`` / ``cached_freqs_gen`` so pipeline
    tests can assert that:
      * ``reset_cache()`` is called before each diffusion loop;
      * UND/cond and uncond branches each populate their own cache exactly
        once and reuse it on subsequent timesteps.
    """

    def __init__(self, *, latent_channel_size: int = 2) -> None:
        super().__init__()
        self.latent_channel_size = latent_channel_size
        self.cached_kv: Any | None = None
        self.cached_freqs_gen: Any | None = None
        self.calls: list[dict[str, Any]] = []
        self.reset_calls = 0

    def reset_cache(self) -> None:
        self.reset_calls += 1
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        token = int(text_ids.reshape(-1)[0].item()) if text_ids.numel() else 0
        self.calls.append(
            {
                "token": token,
                "timestep": timestep.clone(),
                "text_mask": text_mask.clone(),
                "cache_before": self.cached_kv,
                "kwargs": dict(kwargs),
            }
        )
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        return torch.full_like(hidden_states, float(token))


def passthrough_progress_bar(iterable):
    return iterable


# ---------------------------------------------------------------------------
# Tiny config — mirrors test_cosmos3_transformer.py:15-29
# ---------------------------------------------------------------------------
def _tiny_cosmos3_config(**overrides: Any) -> dict:
    """Minimal Cosmos3 transformer config sufficient for shape-only construction.

    Mirrors ``tests/diffusion/models/cosmos3/test_cosmos3_transformer.py:15-29``.
    """
    config: dict = {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Guardrail no-op stub — mirrors conftest.py:121-129
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fake_cosmos3_guardrails(monkeypatch: pytest.MonkeyPatch):
    """Install a no-op replacement for the Cosmos3 guardrails module.

    The vllm-omni reference imports guardrails eagerly; FastVideo's port
    may either skip guardrails or use a different module path. Either
    way, an autouse stub avoids accidental network-dependent imports
    during scaffold-test collection.
    """
    module = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    module.is_guardrails_enabled = lambda od_config, sampling_params=None: False  # type: ignore[attr-defined]
    module.ensure_initialized = lambda od_config: None  # type: ignore[attr-defined]
    module.check_text_safety = lambda text: None  # type: ignore[attr-defined]
    module.check_video_safety = lambda video: video  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


# ---------------------------------------------------------------------------
# Pipeline factory — kept as a scaffold; activate in Phase 2b when the
# FastVideo Cosmos3 pipeline class exists.
# ---------------------------------------------------------------------------
@pytest.fixture
def make_cosmos3_pipeline():
    """Factory that returns a FastVideo Cosmos3 pipeline pre-wired with stubs.

    In Phase 2a (Tier A, no real weights), tests that consume this factory
    should ``pytest.skip`` if the FastVideo Cosmos3 pipeline class does not
    yet exist. In Phase 2b we replace the placeholder construction with
    ``object.__new__(<FastVideoCosmos3Pipeline>)`` + ``nn.Module.__init__``
    and re-point the stubs onto the real attribute names.

    Mirrors the upstream layout at
    ``tests/diffusion/models/cosmos3/conftest.py:132-157``.
    """

    def _make():
        try:
            from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
                Cosmos3OmniDiffusersPipeline,
            )
        except ImportError:
            pytest.skip(
                "FastVideo Cosmos3 pipeline class not yet implemented "
                "(Phase 2b will provide fastvideo.pipelines.basic.cosmos3)."
            )

        pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
        nn.Module.__init__(pipeline)
        pipeline.od_config = SimpleNamespace()
        pipeline.device = torch.device("cpu")
        pipeline.dtype = torch.float32
        pipeline.transformer = StubCosmos3Transformer(latent_channel_size=2)
        pipeline.vae = StubCosmos3VAE(z_dim=2)
        pipeline.vae_scale_factor_temporal = 4
        pipeline.vae_scale_factor_spatial = 8
        pipeline.scheduler = StubScheduler([9, 3], flow_shift=1.0)
        pipeline._base_scheduler_config = pipeline.scheduler.config
        pipeline._engine_init_flow_shift = 1.0
        pipeline._current_flow_shift = 1.0
        pipeline._guidance_scale = None
        pipeline._num_timesteps = None
        pipeline.progress_bar = passthrough_progress_bar
        return pipeline

    return _make


def make_sampling_params(**overrides: Any) -> SimpleNamespace:
    """Build a SamplingParams-like namespace with Cosmos3's expected fields.

    Mirrors ``tests/diffusion/models/cosmos3/conftest.py:160-176``.
    """
    values = {
        "height": None,
        "width": None,
        "num_frames": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "generator": None,
        "seed": 123,
        "num_outputs_per_prompt": 1,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "max_sequence_length": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)
