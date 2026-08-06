# SPDX-License-Identifier: Apache-2.0
"""Public runtime-control coverage for MAGI-2 inference."""

from __future__ import annotations

import os
import random
from typing import cast

import numpy as np
import pytest
import torch

from fastvideo import envs
from fastvideo.api.compat import (
    generator_config_to_fastvideo_args,
    legacy_from_pretrained_to_config,
)
from fastvideo.api.schema import EngineConfig, GeneratorConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.magi2.magi2_pipeline import (
    _configure_deterministic_kernels,
)
from fastvideo.pipelines.basic.magi2.stages import output as magi2_output_stage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import FlexibleArgumentParser


def test_deterministic_cli_flag_is_enabled_without_a_value() -> None:
    """Expose ``--deterministic`` through the legacy FastVideo CLI."""
    parser = FlexibleArgumentParser()
    FastVideoArgs.add_cli_args(parser)
    arguments = parser.parse_args(
        ["--model-path", "/models/magi2", "--deterministic"]
    )
    assert arguments.deterministic is True


def test_deterministic_control_round_trips_through_typed_config(monkeypatch) -> None:
    """Preserve deterministic mode across legacy and typed engine adapters."""
    typed_config = legacy_from_pretrained_to_config(
        "/models/magi2",
        {"deterministic": True},
    )
    assert typed_config.engine.deterministic is True

    captured_kwargs: dict = {}

    def capture_fastvideo_args(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs

    monkeypatch.setattr(FastVideoArgs, "from_kwargs", capture_fastvideo_args)
    generator_config_to_fastvideo_args(
        GeneratorConfig(
            model_path="/models/magi2",
            engine=EngineConfig(deterministic=True),
        )
    )
    assert captured_kwargs["deterministic"] is True


def test_magi2_environment_controls_are_lazy(monkeypatch) -> None:
    """Read deterministic and latent-capture values from each worker's environment."""
    monkeypatch.setenv("MAGI2_DETERMINISTIC", "1")
    monkeypatch.setenv("MAGI_ATTENTION_DETERMINISTIC_MODE", "1")
    monkeypatch.setenv("MAGI2_SAVE_LATENT_PATH", "/tmp/magi2-latents")

    assert envs.MAGI2_DETERMINISTIC is True
    assert envs.MAGI_ATTENTION_DETERMINISTIC_MODE is True
    assert envs.MAGI2_SAVE_LATENT_PATH == "/tmp/magi2-latents"


def test_deterministic_kernel_configuration_repeats_cpu_rng_sequences(
    monkeypatch,
) -> None:
    """Seed Python, NumPy, and CPU PyTorch with repeatable sequences."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda_seed_calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed_calls.append)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    monkeypatch.delenv("MAGI_ATTENTION_DETERMINISTIC_MODE", raising=False)

    try:
        _configure_deterministic_kernels(seed=1234)
        first_sequence = (random.random(), np.random.random(4), torch.rand(4))
        _configure_deterministic_kernels(seed=1234)
        second_sequence = (random.random(), np.random.random(4), torch.rand(4))

        assert first_sequence[0] == second_sequence[0]
        assert np.array_equal(first_sequence[1], second_sequence[1])
        assert torch.equal(first_sequence[2], second_sequence[2])
        assert cuda_seed_calls and set(cuda_seed_calls) == {1234}
        assert torch.are_deterministic_algorithms_enabled()
        assert envs.MAGI2_DETERMINISTIC is True
        assert envs.MAGI_ATTENTION_DETERMINISTIC_MODE is True
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        torch.use_deterministic_algorithms(
            deterministic_algorithms,
            warn_only=deterministic_warn_only,
        )


def test_latent_saving_stage_writes_leader_latent(tmp_path, monkeypatch) -> None:
    """Save the post-refiner latent on the context-parallel leader rank."""
    latent_directory = tmp_path / "latents"
    monkeypatch.setenv("MAGI2_SAVE_LATENT_PATH", str(latent_directory))
    monkeypatch.setattr(
        magi2_output_stage.psm,
        "is_group_first_rank",
        lambda dimension: dimension == "cp",
    )
    stage = magi2_output_stage.Magi2LatentSavingStage()
    latent = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    batch = ForwardBatch(data_type="video", latents=latent)

    returned_batch = stage.forward(batch, cast(FastVideoArgs, object()))

    assert returned_batch is batch
    assert torch.equal(
        torch.load(latent_directory / "latent_0.pt", weights_only=True),
        latent,
    )
    assert stage.sample_index == 1


@pytest.mark.parametrize(
    ("configured_directory", "is_leader"),
    [("", True), ("nonleader-latents", False)],
    ids=["empty-path", "non-leader"],
)
def test_latent_saving_stage_skips_disabled_ranks(
    tmp_path,
    monkeypatch,
    configured_directory: str,
    is_leader: bool,
) -> None:
    """Skip latent writes for an empty path and for non-leader ranks."""
    monkeypatch.chdir(tmp_path)
    latent_directory = "" if configured_directory == "" else str(tmp_path / configured_directory)
    monkeypatch.setenv("MAGI2_SAVE_LATENT_PATH", latent_directory)
    monkeypatch.setattr(
        magi2_output_stage.psm,
        "is_group_first_rank",
        lambda dimension: is_leader,
    )
    stage = magi2_output_stage.Magi2LatentSavingStage()
    batch = ForwardBatch(data_type="video", latents=torch.ones(1))

    returned_batch = stage.forward(batch, cast(FastVideoArgs, object()))

    assert returned_batch is batch
    assert list(tmp_path.rglob("latent_*.pt")) == []
    assert stage.sample_index == 0
