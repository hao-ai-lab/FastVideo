# SPDX-License-Identifier: Apache-2.0
"""Helios-Distilled DMD scheduler parity.

Coverage scope: implementation_subcomponent. The test compares the native
FastVideo scheduler against the exact Diffusers class declared by the pinned
Helios-Distilled scheduler config. It covers every pyramid stage, dynamic time
shift, first-chunk amplification, and both branches of the DMD step.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from diffusers import HeliosDMDScheduler as OfficialHeliosDMDScheduler
from torch.testing import assert_close

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULER_DIR = REPO_ROOT / "official_weights" / "helios" / "scheduler"
PARITY_SCOPE = "implementation_subcomponent"


def _scheduler_kwargs() -> dict:
    config_path = SCHEDULER_DIR / "scheduler_config.json"
    if not config_path.exists():
        pytest.skip(f"Helios scheduler config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("_class_name", "_diffusers_version", "scheduler_type"):
        config.pop(key, None)
    return config


def _fastvideo_scheduler_class():
    try:
        from fastvideo.models.schedulers.scheduling_helios_dmd import (
            HeliosDMDScheduler, )
    except ImportError as exc:
        raise AssertionError("Native FastVideo HeliosDMDScheduler has not been implemented yet") from exc
    return HeliosDMDScheduler


def _make_pair():
    kwargs = _scheduler_kwargs()
    # The official class declares fewer keys than the FastVideo port and has no
    # ``**kwargs``, so a checkpoint shipping the extra shift keys must be
    # filtered for it while the port still receives them.
    official_keys = inspect.signature(OfficialHeliosDMDScheduler.__init__).parameters
    official_kwargs = {key: value for key, value in kwargs.items() if key in official_keys}
    return (
        OfficialHeliosDMDScheduler(**official_kwargs),
        _fastvideo_scheduler_class()(**kwargs),
    )


def test_helios_dmd_scheduler_resolves_through_production_registry():
    from fastvideo.models.registry import ModelRegistry

    scheduler_cls, architecture = ModelRegistry.resolve_model_cls("HeliosDMDScheduler")

    assert architecture == "HeliosDMDScheduler"
    assert scheduler_cls is _fastvideo_scheduler_class()


def test_helios_dmd_scheduler_registers_shift_config_used_for_mu():
    """``HeliosPyramidDenoisingStage`` derives ``mu`` from these config keys.

    They must survive ``register_to_config``; otherwise a checkpoint-provided
    shift schedule is swallowed by ``**kwargs`` and silently replaced by the
    literals, so the dynamic-shift timesteps diverge from the official class.
    """
    from fastvideo.pipelines.basic.helios.pipeline_utils import calculate_shift

    kwargs = _scheduler_kwargs()
    official, fastvideo = _make_pair()
    for name, default in (
        ("base_image_seq_len", 256),
        ("max_image_seq_len", 4096),
        ("base_shift", 0.5),
        ("max_shift", 1.15),
    ):
        assert getattr(fastvideo.config, name) == getattr(official.config, name, kwargs.get(name, default))

    # The mu the pipeline feeds to set_timesteps must come from the registered
    # config, not from the literals.
    mu = calculate_shift(
        2176,
        fastvideo.config.base_image_seq_len,
        fastvideo.config.max_image_seq_len,
        fastvideo.config.base_shift,
        fastvideo.config.max_shift,
    )
    assert mu == calculate_shift(
        2176,
        kwargs.get("base_image_seq_len", 256),
        kwargs.get("max_image_seq_len", 4096),
        kwargs.get("base_shift", 0.5),
        kwargs.get("max_shift", 1.15),
    )


@pytest.mark.parametrize("stage_index", [0, 1, 2])
@pytest.mark.parametrize("amplify", [False, True])
def test_helios_dmd_scheduler_stage_schedule_parity(stage_index: int, amplify: bool):
    official, fastvideo = _make_pair()
    call_kwargs = {
        "num_inference_steps": 2,
        "stage_index": stage_index,
        "device": "cpu",
        "mu": 1.07,
        "is_amplify_first_chunk": amplify,
    }
    official.set_timesteps(**call_kwargs)
    fastvideo.set_timesteps(**call_kwargs)

    assert_close(fastvideo.timesteps, official.timesteps, atol=0, rtol=0)
    assert_close(fastvideo.sigmas, official.sigmas, atol=0, rtol=0)
    assert fastvideo.timestep_ratios == official.timestep_ratios
    assert fastvideo.start_sigmas == official.start_sigmas
    assert fastvideo.end_sigmas == official.end_sigmas
    assert fastvideo.ori_start_sigmas == official.ori_start_sigmas


@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_helios_dmd_scheduler_step_parity(stage_index: int):
    official, fastvideo = _make_pair()
    call_kwargs = {
        "num_inference_steps": 2,
        "stage_index": stage_index,
        "device": "cpu",
        "mu": 1.07,
    }
    official.set_timesteps(**call_kwargs)
    fastvideo.set_timesteps(**call_kwargs)

    generator = torch.Generator(device="cpu").manual_seed(20260711 + stage_index)
    sample = torch.randn(2, 4, 3, 4, 6, generator=generator)
    noisy_start = torch.randn(sample.shape, generator=generator)

    for step_index, (official_t, fastvideo_t) in enumerate(zip(official.timesteps, fastvideo.timesteps, strict=True)):
        model_output = torch.randn(sample.shape, generator=generator)
        official_sample = official.step(
            model_output=model_output,
            timestep=official_t,
            sample=sample,
            cur_sampling_step=step_index,
            dmd_noisy_tensor=noisy_start,
            dmd_sigmas=official.sigmas,
            dmd_timesteps=official.timesteps,
            all_timesteps=official.timesteps,
            return_dict=False,
        )[0]
        fastvideo_sample = fastvideo.step(
            model_output=model_output,
            timestep=fastvideo_t,
            sample=sample,
            cur_sampling_step=step_index,
            dmd_noisy_tensor=noisy_start,
            dmd_sigmas=fastvideo.sigmas,
            dmd_timesteps=fastvideo.timesteps,
            all_timesteps=fastvideo.timesteps,
            return_dict=False,
        )[0]
        assert_close(fastvideo_sample, official_sample, atol=0, rtol=0)
        sample = official_sample
