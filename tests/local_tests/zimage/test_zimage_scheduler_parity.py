# SPDX-License-Identifier: Apache-2.0
"""
Parity tests for Z-Image FlowMatchEulerDiscreteScheduler.

These tests compare FastVideo's scheduler implementation against the
reference scheduler shipped in the local Z-Image repository.

Usage:
    pytest tests/local_tests/zimage/test_zimage_scheduler_parity.py -v
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as FastVideoScheduler,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ZIMAGE_REPO = REPO_ROOT / "Z-Image"
ZIMAGE_SRC = REPO_ROOT / "Z-Image" / "src"
ZIMAGE_SCHEDULER_CFG = (
    REPO_ROOT / "official_weights" / "Z-Image" / "scheduler" / "scheduler_config.json"
)
ZIMAGE_REFERENCE_REVISION = "26f23eda626ffadda020b04ff79488e1d72004cd"
PARITY_SCOPE = "implementation_subcomponent"


def _require_pinned_reference_module(module_name: str, source_file: Path):
    if not ZIMAGE_REPO.exists():
        pytest.skip(f"Pinned Z-Image reference clone not found: {ZIMAGE_REPO}")
    if not source_file.is_file():
        pytest.fail(f"Z-Image reference clone is incomplete; missing {source_file}")

    try:
        result = subprocess.run(
            ["git", "-C", str(ZIMAGE_REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"Cannot verify Z-Image reference revision: {exc}")

    actual_revision = result.stdout.strip()
    assert actual_revision == ZIMAGE_REFERENCE_REVISION, (
        "Z-Image reference clone is not at the pinned revision: "
        f"expected {ZIMAGE_REFERENCE_REVISION}, got {actual_revision}"
    )

    if str(ZIMAGE_SRC) not in sys.path:
        sys.path.insert(0, str(ZIMAGE_SRC))
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(f"Cannot import pinned Z-Image module {module_name}: {exc}")

    module_file = Path(module.__file__ or "").resolve()
    assert module_file.is_relative_to(ZIMAGE_SRC.resolve()), (
        f"{module_name} resolved outside the pinned clone: {module_file}"
    )
    return module


@pytest.fixture(scope="module")
def reference_scheduler_cls():
    module = _require_pinned_reference_module(
        "zimage.scheduler",
        ZIMAGE_SRC / "zimage" / "scheduler.py",
    )
    return module.FlowMatchEulerDiscreteScheduler


@pytest.fixture(scope="module")
def scheduler_config() -> dict:
    if not ZIMAGE_SCHEDULER_CFG.exists():
        pytest.skip(f"Z-Image scheduler config not found: {ZIMAGE_SCHEDULER_CFG}")
    with ZIMAGE_SCHEDULER_CFG.open("r", encoding="utf-8") as f:
        return json.load(f)


# Diffusers-style scheduler-config keys we do NOT want to forward — they are
# loader/state metadata, not constructor kwargs. Anything else in
# scheduler_config.json (including future keys like `time_shift_type`,
# `invert_sigmas`, etc.) must be forwarded so parity actually exercises the
# config-on-disk and not a hand-curated subset.
_SCHEDULER_CONFIG_LOADER_KEYS = frozenset({"_class_name", "_diffusers_version", "_name_or_path"})


def _scheduler_kwargs_from_config(scheduler_config: dict) -> dict:
    return {k: v for k, v in scheduler_config.items() if k not in _SCHEDULER_CONFIG_LOADER_KEYS}


def _make_scheduler_pair(reference_scheduler_cls, scheduler_config: dict, **overrides):
    kwargs = _scheduler_kwargs_from_config(scheduler_config)
    kwargs.update(overrides)

    ref = reference_scheduler_cls(**kwargs)
    # The pinned Z-Image pipeline mutates this immediately before scheduling.
    ref.sigma_min = 0.0

    # These values must be serialized by the future production pipeline config.
    # setdefault also allows this test to consume that config once it lands.
    fv_kwargs = dict(kwargs)
    fv_kwargs.setdefault("use_reference_discrete_timesteps", True)
    fv_kwargs.setdefault("sigma_min", 0.0)
    fv = FastVideoScheduler(**fv_kwargs)
    return ref, fv


@pytest.fixture(scope="module")
def scheduler_pair(reference_scheduler_cls, scheduler_config: dict):
    return _make_scheduler_pair(reference_scheduler_cls, scheduler_config)


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


def test_scheduler_positional_args_keep_existing_bindings_and_default_schedule():
    scheduler = FastVideoScheduler(1000, 1.0, False, 0.7)

    assert scheduler.config.base_shift == 0.7
    assert scheduler.config.use_reference_discrete_timesteps is False

    scheduler.set_timesteps(num_inference_steps=8, device="cpu")
    assert_close(
        scheduler.sigmas[-2],
        torch.tensor(scheduler.sigma_min, dtype=torch.float32),
    )


def test_scheduler_honors_explicit_zero_sigma_min():
    scheduler = FastVideoScheduler(
        sigma_min=0.0,
        use_reference_discrete_timesteps=True,
    )

    assert scheduler.sigma_min == 0.0
    scheduler.set_timesteps(num_inference_steps=8, device="cpu")
    assert_close(
        scheduler.timesteps[-1],
        torch.tensor(125.0, dtype=torch.float32),
    )


def test_reference_schedule_preserves_float64_linspace_rounding():
    scheduler = FastVideoScheduler(
        sigma_min=0.0,
        use_reference_discrete_timesteps=True,
    )
    scheduler.set_timesteps(num_inference_steps=9, device="cpu")

    expected = torch.from_numpy(
        np.linspace(1000.0, 0.0, 10)[:-1] / 1000.0,
    ).to(torch.float32) * 1000.0
    float32_regression = torch.from_numpy(
        np.linspace(1000.0, 0.0, 10, dtype=np.float32)[:-1] / np.float32(1000.0),
    ) * 1000.0

    # The fifth timestep differs by one float32 ULP depending on where the
    # linspace is rounded, so this assertion fails if dtype=np.float32 returns.
    assert expected[4].item() != float32_regression[4].item()
    assert scheduler.timesteps[4].item() == expected[4].item()


def test_zimage_scheduler_parity_dynamic_shifting_with_mu(
    reference_scheduler_cls,
    scheduler_config: dict,
):
    ref, fv = _make_scheduler_pair(
        reference_scheduler_cls,
        scheduler_config,
        use_dynamic_shifting=True,
    )

    mu = 0.75
    num_inference_steps = 8
    ref.set_timesteps(num_inference_steps=num_inference_steps, device="cpu", mu=mu)
    fv.set_timesteps(num_inference_steps=num_inference_steps, device="cpu", mu=mu)

    assert_close(ref.timesteps, fv.timesteps, atol=1e-4, rtol=1e-6)
    assert_close(ref.sigmas, fv.sigmas, atol=1e-7, rtol=1e-6)

    _run_step_loop(ref, fv)
