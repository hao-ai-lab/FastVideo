# SPDX-License-Identifier: Apache-2.0
"""Strict MAGI-2 Flow UniPC scheduler parity against the official source."""

from __future__ import annotations

import ast
import math
import typing
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_close

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import (
        KarrasDiffusionSchedulers,
        SchedulerMixin,
        SchedulerOutput,
    )
    from diffusers.utils import deprecate
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler as FastVideoFlowUniPCMultistepScheduler,
    )
except ModuleNotFoundError as exc:
    if exc.name != "diffusers":
        raise
    pytest.skip(
        "MAGI-2 scheduler parity requires the Diffusers package.",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SCHEDULER_PATH = (
    REPO_ROOT.parent
    / "MAGI-2-preview"
    / "inference"
    / "pipeline"
    / "sampler.py"
)
OFFICIAL_SCHEDULER_CLASS_NAME = "FlowUniPCMultistepScheduler"


def _load_official_scheduler_class() -> type:
    """Load only the official scheduler class without model-runtime imports."""
    if not OFFICIAL_SCHEDULER_PATH.is_file():
        raise FileNotFoundError(
            f"Official MAGI-2 scheduler source is missing: {OFFICIAL_SCHEDULER_PATH}"
        )

    source = OFFICIAL_SCHEDULER_PATH.read_text(encoding="utf-8")
    source_module = ast.parse(source, filename=str(OFFICIAL_SCHEDULER_PATH))
    scheduler_nodes = [
        node
        for node in source_module.body
        if isinstance(node, ast.ClassDef)
        and node.name == OFFICIAL_SCHEDULER_CLASS_NAME
    ]
    if len(scheduler_nodes) != 1:
        raise AssertionError(
            f"Expected one {OFFICIAL_SCHEDULER_CLASS_NAME} definition in "
            f"{OFFICIAL_SCHEDULER_PATH}, found {len(scheduler_nodes)}"
        )

    scheduler_namespace = {
        "__name__": "magi2_official_scheduler",
        "Any": typing.Any,
        "ConfigMixin": ConfigMixin,
        "KarrasDiffusionSchedulers": KarrasDiffusionSchedulers,
        "List": list,
        "Optional": typing.Optional,
        "SchedulerMixin": SchedulerMixin,
        "SchedulerOutput": SchedulerOutput,
        "Tuple": tuple,
        "Union": typing.Union,
        "deprecate": deprecate,
        "math": math,
        "np": np,
        "register_to_config": register_to_config,
        "torch": torch,
    }
    scheduler_module = ast.Module(body=scheduler_nodes, type_ignores=[])
    exec(
        compile(scheduler_module, str(OFFICIAL_SCHEDULER_PATH), "exec"),
        scheduler_namespace,
    )
    return scheduler_namespace[OFFICIAL_SCHEDULER_CLASS_NAME]


OfficialFlowUniPCMultistepScheduler = _load_official_scheduler_class()


def _build_scheduler_pair(
    num_inference_steps: int,
    shift: float,
) -> tuple[SchedulerMixin, SchedulerMixin]:
    """Construct official and FastVideo schedulers with a shipping schedule."""
    official_scheduler = OfficialFlowUniPCMultistepScheduler()
    fastvideo_scheduler = FastVideoFlowUniPCMultistepScheduler()
    official_scheduler.set_timesteps(num_inference_steps, device="cpu", shift=shift)
    fastvideo_scheduler.set_timesteps(num_inference_steps, device="cpu", shift=shift)
    return official_scheduler, fastvideo_scheduler


def _assert_scheduler_tensor_exact(
    fastvideo_tensor: torch.Tensor,
    official_tensor: torch.Tensor,
) -> None:
    """Require identical scheduler tensor metadata and values."""
    assert fastvideo_tensor.shape == official_tensor.shape
    assert fastvideo_tensor.dtype == official_tensor.dtype
    assert fastvideo_tensor.stride() == official_tensor.stride()
    assert_close(fastvideo_tensor, official_tensor, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("num_inference_steps", "shift"),
    [(100, 7.0), (5, 5.0)],
    ids=("preview", "refiner"),
)
def test_set_timesteps_magi2_shipping_schedules_match_official(
    num_inference_steps: int,
    shift: float,
) -> None:
    """Match the preview and refiner timestep and sigma schedules exactly."""
    official_scheduler, fastvideo_scheduler = _build_scheduler_pair(
        num_inference_steps,
        shift,
    )

    _assert_scheduler_tensor_exact(
        fastvideo_scheduler.timesteps,
        official_scheduler.timesteps,
    )
    _assert_scheduler_tensor_exact(
        fastvideo_scheduler.sigmas,
        official_scheduler.sigmas,
    )


@pytest.mark.parametrize(
    ("num_inference_steps", "shift"),
    [(100, 7.0), (5, 5.0)],
    ids=("preview", "refiner"),
)
def test_step_magi2_shipping_denoise_trajectory_matches_official(
    num_inference_steps: int,
    shift: float,
) -> None:
    """Match every scheduler output across each shipping denoise schedule."""
    official_scheduler, fastvideo_scheduler = _build_scheduler_pair(
        num_inference_steps,
        shift,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260805)
    tensor_shape = (1, 2, 3, 4, 4)
    official_sample = torch.randn(
        tensor_shape,
        generator=generator,
        dtype=torch.float32,
    )
    fastvideo_sample = official_sample.clone()
    model_outputs = [
        torch.randn(tensor_shape, generator=generator, dtype=torch.float32)
        for _ in range(num_inference_steps)
    ]

    for step_index, model_output in enumerate(model_outputs):
        official_sample = official_scheduler.step(
            model_output,
            official_scheduler.timesteps[step_index],
            official_sample,
            return_dict=False,
        )[0]
        fastvideo_sample = fastvideo_scheduler.step(
            model_output,
            fastvideo_scheduler.timesteps[step_index],
            fastvideo_sample,
            return_dict=False,
        )[0]
        _assert_scheduler_tensor_exact(fastvideo_sample, official_sample)
