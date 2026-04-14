# SPDX-License-Identifier: Apache-2.0
"""Loader smoke tests for FLUX.1-dev component loading from a local checkpoint.

Verifies that tokenizers, CLIP/T5 text encoders, VAE, and FlowMatch scheduler
all load correctly from a Diffusers-layout ``FLUX.1-dev`` directory.

Requires ``official_weights/FLUX.1-dev`` (or set ``FLUX_DEV_ROOT``) and CUDA.

Run from repo root::

    pytest tests/local_tests/flux/test_flux_dev_component_loaders.py -vs
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
)
from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLUX_DEV_ROOT = os.environ.get(
    "FLUX_DEV_ROOT",
    str(_REPO_ROOT / "official_weights" / "FLUX.1-dev"),
)


def _flux_clip_post(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.pooler_output is None:
        raise RuntimeError("CLIP pooled output required")
    return outputs.pooler_output


def _flux_t5_post(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.last_hidden_state is None:
        raise RuntimeError("T5 last_hidden_state required")
    return outputs.last_hidden_state


@dataclass
class _FluxDevLoaderPipelineConfig(PipelineConfig):
    """Two text encoders (CLIP + T5) matching FLUX.1-dev model_index.json."""

    vae_config: AutoencoderKLVAEConfig = field(
        default_factory=AutoencoderKLVAEConfig
    )
    text_encoder_configs: tuple[CLIPTextConfig, T5Config] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config())
    )
    text_encoder_precisions: tuple[str, ...] = ("fp32", "bf16")
    preprocess_text_funcs: tuple = field(
        default_factory=lambda: (preprocess_text, preprocess_text)
    )
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (_flux_clip_post, _flux_t5_post)
    )


def test_flux_dev_model_index_components_load() -> None:
    """CLIP/T5 encoders, tokenizers, VAE, and FlowMatch scheduler load."""
    if not torch.cuda.is_available():
        pytest.skip("FLUX component loader test requires CUDA")
    if not Path(_FLUX_DEV_ROOT, "model_index.json").is_file():
        pytest.skip(
            "official_weights/FLUX.1-dev missing "
            "(set FLUX_DEV_ROOT or download black-forest-labs/FLUX.1-dev)"
        )

    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import (
        SchedulerLoader,
        TextEncoderLoader,
        TokenizerLoader,
        VAELoader,
    )

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        args = FastVideoArgs(
            model_path=_FLUX_DEV_ROOT,
            pipeline_config=_FluxDevLoaderPipelineConfig(),
            hsdp_shard_dim=1,
            pin_cpu_memory=False,
        )

        tok_clip = TokenizerLoader().load(
            os.path.join(_FLUX_DEV_ROOT, "tokenizer"), args
        )
        tok_t5 = TokenizerLoader().load(
            os.path.join(_FLUX_DEV_ROOT, "tokenizer_2"), args
        )
        assert "CLIP" in tok_clip.__class__.__name__
        assert "T5" in tok_t5.__class__.__name__

        te_clip = TextEncoderLoader().load(
            os.path.join(_FLUX_DEV_ROOT, "text_encoder"), args
        )
        te_t5 = TextEncoderLoader().load(
            os.path.join(_FLUX_DEV_ROOT, "text_encoder_2"), args
        )
        # Distributed init may wrap encoders in an FSDP shell.
        assert te_clip.__class__.__name__.endswith("CLIPTextModel")
        assert te_t5.__class__.__name__.endswith("T5EncoderModel")

        vae = VAELoader().load(os.path.join(_FLUX_DEV_ROOT, "vae"), args)
        assert vae.__class__.__name__ == "AutoencoderKL"

        scheduler = SchedulerLoader().load(
            os.path.join(_FLUX_DEV_ROOT, "scheduler"), args
        )
        assert scheduler.__class__.__name__ == "FlowMatchEulerDiscreteScheduler"
        scheduler.set_timesteps(4, mu=0.7)
        assert len(scheduler.timesteps) == 4
    finally:
        cleanup_dist_env_and_memory()
