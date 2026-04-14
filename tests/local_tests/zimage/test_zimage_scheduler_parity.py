# SPDX-License-Identifier: Apache-2.0
"""
Parity tests for Z-Image FlowMatchEulerDiscreteScheduler.

These tests compare FastVideo's scheduler implementation against the
reference scheduler shipped in the local Z-Image repository.

Usage:
    pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as FastVideoScheduler,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_SRC = REPO_ROOT / "Z-Image" / "src"
ZIMAGE_SCHEDULER_CFG = (
    REPO_ROOT / "official_weights" / "Z-Image" / "scheduler" / "scheduler_config.json"
)


if str(ZIMAGE_SRC) not in sys.path:
    sys.path.insert(0, str(ZIMAGE_SRC))

try:
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler as ReferenceScheduler
except Exception as exc:  # pragma: no cover - handled by skip in fixture
    ReferenceScheduler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="module")
def scheduler_config() -> dict:
    if not ZIMAGE_SCHEDULER_CFG.exists():
        pytest.skip(f"Z-Image scheduler config not found: {ZIMAGE_SCHEDULER_CFG}")
    with ZIMAGE_SCHEDULER_CFG.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scheduler_pair(scheduler_config: dict):
    if ReferenceScheduler is None:
        pytest.skip(f"Cannot import Z-Image reference scheduler: {_IMPORT_ERROR}")

    kwargs = {
        "num_train_timesteps": scheduler_config["num_train_timesteps"],
        "shift": scheduler_config["shift"],
        "use_dynamic_shifting": scheduler_config["use_dynamic_shifting"],
    }
    ref = ReferenceScheduler(**kwargs)
    fv = FastVideoScheduler(**kwargs, use_reference_discrete_timesteps=True)
    return ref, fv


def _run_step_loop(ref_scheduler, fv_scheduler, sample_shape=(2, 4, 32, 32)):
    torch.manual_seed(123)

    ref_sample = torch.randn(sample_shape, dtype=torch.float32)
    fv_sample = ref_sample.clone()

    for t in ref_scheduler.timesteps:
        model_output = torch.randn_like(ref_sample)

        ref_next = ref_scheduler.step(
            model_output,
            t,
            ref_sample,
            return_dict=False,
        )[0]
        fv_next = fv_scheduler.step(
            model_output,
            t,
            fv_sample,
            return_dict=False,
        )[0]

        assert_close(ref_next, fv_next, atol=1e-6, rtol=1e-6)

        ref_sample = ref_next
        fv_sample = fv_next


def test_zimage_scheduler_parity_default_schedule_and_step(scheduler_pair):
    ref, fv = scheduler_pair

    num_inference_steps = 8
    ref.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
    fv.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")

    assert_close(ref.timesteps, fv.timesteps, atol=1e-4, rtol=1e-6)
    assert_close(ref.sigmas, fv.sigmas, atol=1e-7, rtol=1e-6)

    _run_step_loop(ref, fv)


def test_zimage_scheduler_parity_dynamic_shifting_with_mu(scheduler_config: dict):
    if ReferenceScheduler is None:
        pytest.skip(f"Cannot import Z-Image reference scheduler: {_IMPORT_ERROR}")

    kwargs = {
        "num_train_timesteps": scheduler_config["num_train_timesteps"],
        "shift": scheduler_config["shift"],
        "use_dynamic_shifting": True,
    }
    ref = ReferenceScheduler(**kwargs)
    fv = FastVideoScheduler(**kwargs, use_reference_discrete_timesteps=True)

    mu = 0.75
    num_inference_steps = 8
    ref.set_timesteps(num_inference_steps=num_inference_steps, device="cpu", mu=mu)
    fv.set_timesteps(num_inference_steps=num_inference_steps, device="cpu", mu=mu)

    assert_close(ref.timesteps, fv.timesteps, atol=1e-4, rtol=1e-6)
    assert_close(ref.sigmas, fv.sigmas, atol=1e-7, rtol=1e-6)

    _run_step_loop(ref, fv)
