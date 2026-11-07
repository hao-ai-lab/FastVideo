# SPDX-License-Identifier: Apache-2.0
"""PAN2 flow-matching Euler schedule on FastVideo's FlowMatchEulerDiscreteScheduler.

Coverage scope: both. PAN2 loads Diffusers' ``FlowMatchEulerDiscreteScheduler``
from ``<checkpoint>/scheduler`` (shift 7.0, no dynamic shifting), optionally
overrides the shift with ``flow_shift``, calls
``set_timesteps(sigmas=np.linspace(1.0, 0.0, N + 1)[:-1])`` and steps float32
latents. The FastVideo side is loaded by the production ``SchedulerLoader`` and
its schedule is built by the stock ``TimestepPreparationStage`` from
``batch.sigmas``. Timesteps, sigmas, and every Euler update must be bit-exact.

The default asset is the tiny PAN2 checkpoint (config only), so this runs on a
CPU-only host::

    pytest tests/local_tests/schedulers/test_pan2_scheduler_parity.py -v -s

Set ``PAN2_MODEL_PATH`` to a local PAN2 root containing ``scheduler/`` to use
another checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.testing import assert_close

try:
    from diffusers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler
except ImportError:  # pragma: no cover - environment dependent
    pytest.skip("diffusers with FlowMatchEulerDiscreteScheduler is required", allow_module_level=True)

from fastvideo.models.loader.component_loader import SchedulerLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as FastVideoFlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage

PARITY_SCOPE = "both"
TINY_PAN2_REPO = "wuqing157/tiny-pan2-modular-pipe"
TINY_PAN2_REVISION = "3ff3130b4527f61aaab7bde01465c9884dc84cdd"
MODEL_PATH_ENV = "PAN2_MODEL_PATH"
LATENT_SHAPE = (1, 48, 3, 4, 6)


def _pan2_root(subfolder: str) -> Path:
    """Resolve a PAN2 root that contains ``subfolder``; skip when unavailable."""
    override = os.getenv(MODEL_PATH_ENV)
    if override:
        root = Path(override)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            pytest.skip("huggingface_hub is required to resolve the tiny PAN2 checkpoint")
        root = None
        last_error: Exception | None = None
        for local_files_only in (True, False):
            try:
                root = Path(
                    snapshot_download(
                        TINY_PAN2_REPO,
                        revision=TINY_PAN2_REVISION,
                        allow_patterns=[f"{subfolder}/*"],
                        local_files_only=local_files_only,
                    ))
                break
            except Exception as exc:  # noqa: BLE001 - offline/missing cache must skip.
                last_error = exc
        if root is None:
            pytest.skip(f"Tiny PAN2 checkpoint unavailable: {last_error}")
    if not (root / subfolder).is_dir():
        pytest.skip(f"PAN2 {subfolder}/ missing under {root}")
    return root


@pytest.fixture(scope="module")
def pan2_root() -> Path:
    root = _pan2_root("scheduler")
    if not (root / "scheduler" / "scheduler_config.json").is_file():
        pytest.skip(f"PAN2 scheduler config missing under {root}")
    return root


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin FastVideo's device helper to CPU; it reports ``mps`` on CPU-only Linux."""
    monkeypatch.setattr("fastvideo.pipelines.stages.timestep_preparation.get_local_torch_device",
                        lambda: torch.device("cpu"))


def _pan2_sigmas(num_steps: int) -> np.ndarray:
    """PAN2's pre-shift grid: uniform from 1 down to, but excluding, 0."""
    return np.linspace(1.0, 0.0, num_steps + 1)[:-1]


def _reference(root: Path, flow_shift: float | None = None) -> DiffusersFlowMatchEulerDiscreteScheduler:
    """Mirror the upstream PAN2 pipeline's scheduler construction."""
    scheduler = DiffusersFlowMatchEulerDiscreteScheduler.from_pretrained(str(root),
                                                                         subfolder="scheduler",
                                                                         local_files_only=True)
    if flow_shift is not None:
        scheduler._shift = flow_shift
    return scheduler


def _fastvideo(root: Path, flow_shift: float | None = None) -> FastVideoFlowMatchEulerDiscreteScheduler:
    """Load through the production SchedulerLoader, which applies ``flow_shift``."""
    fastvideo_args = SimpleNamespace(pipeline_config=SimpleNamespace(flow_shift=flow_shift))
    scheduler = SchedulerLoader().load(str(root / "scheduler"), fastvideo_args)
    assert type(scheduler) is FastVideoFlowMatchEulerDiscreteScheduler
    return scheduler


def _prepare_fastvideo_timesteps(scheduler: FastVideoFlowMatchEulerDiscreteScheduler,
                                 num_steps: int) -> ForwardBatch:
    """Drive the stock TimestepPreparationStage with PAN2's custom sigmas."""
    batch = ForwardBatch(data_type="video", num_inference_steps=num_steps, sigmas=_pan2_sigmas(num_steps).tolist())
    stage = TimestepPreparationStage(scheduler)
    assert stage.verify_input(batch, SimpleNamespace()).is_valid()
    batch = stage.forward(batch, SimpleNamespace())
    assert batch.timesteps is scheduler.timesteps
    return batch


def test_pan2_scheduler_loader_keeps_checkpoint_config(pan2_root: Path) -> None:
    """Without ``flow_shift`` the loader keeps every released config value, including shift 7.0."""
    raw = json.loads((pan2_root / "scheduler" / "scheduler_config.json").read_text())
    assert raw["_class_name"] == "FlowMatchEulerDiscreteScheduler"
    scheduler = _fastvideo(pan2_root)
    for key, value in raw.items():
        if not key.startswith("_"):
            assert scheduler.config[key] == value, key
    assert scheduler.shift == raw["shift"] == 7.0
    assert not scheduler.config.use_dynamic_shifting


@pytest.mark.parametrize("num_steps", [1, 4, 8, 30, 50])
def test_pan2_scheduler_schedule_matches_diffusers(num_steps: int, pan2_root: Path, cpu_runtime: None) -> None:
    """Timesteps and sigmas from ``batch.sigmas`` are bit-exact for short and long schedules."""
    reference = _reference(pan2_root)
    reference.set_timesteps(sigmas=_pan2_sigmas(num_steps), device="cpu")
    fastvideo = _fastvideo(pan2_root)
    batch = _prepare_fastvideo_timesteps(fastvideo, num_steps)

    print(f"[pan2-scheduler] N={num_steps} timesteps[:4]={reference.timesteps[:4].tolist()} "
          f"sigmas[-3:]={reference.sigmas[-3:].tolist()}")
    assert len(batch.timesteps) == num_steps
    assert_close(batch.timesteps, reference.timesteps, atol=0, rtol=0)
    assert_close(fastvideo.sigmas, reference.sigmas, atol=0, rtol=0)
    assert fastvideo.sigmas[0].item() == 1.0 and fastvideo.sigmas[-1].item() == 0.0


@pytest.mark.parametrize("flow_shift", [None, 5.0], ids=["checkpoint_shift", "flow_shift_override"])
def test_pan2_scheduler_steps_match_diffusers(flow_shift: float | None, pan2_root: Path, cpu_runtime: None) -> None:
    """Five float32 Euler updates are bit-exact, with and without a shift override."""
    num_steps = 5
    reference = _reference(pan2_root, flow_shift)
    reference.set_timesteps(sigmas=_pan2_sigmas(num_steps), device="cpu")
    fastvideo = _fastvideo(pan2_root, flow_shift)
    _prepare_fastvideo_timesteps(fastvideo, num_steps)
    assert fastvideo.shift == reference.shift == (flow_shift or 7.0)
    assert_close(fastvideo.timesteps, reference.timesteps, atol=0, rtol=0)

    generator = torch.Generator().manual_seed(2026)
    reference_latents = torch.randn(LATENT_SHAPE, generator=generator)
    fastvideo_latents = reference_latents.clone()
    max_diff = 0.0
    for index in range(num_steps):
        noise_pred = torch.randn(LATENT_SHAPE, generator=generator)
        reference_latents = reference.step(noise_pred, reference.timesteps[index], reference_latents).prev_sample
        fastvideo_latents = fastvideo.step(noise_pred, fastvideo.timesteps[index], fastvideo_latents).prev_sample
        assert fastvideo_latents.dtype == torch.float32
        max_diff = max(max_diff, (fastvideo_latents - reference_latents).abs().max().item())
        assert_close(fastvideo_latents, reference_latents, atol=0, rtol=0)
    print(f"[pan2-scheduler] flow_shift={flow_shift} steps={num_steps} max_abs={max_diff:.3e}")
    assert fastvideo.step_index == reference.step_index == num_steps


def test_pan2_scheduler_step_returns_model_output_dtype(pan2_root: Path, cpu_runtime: None) -> None:
    """Both implementations cast ``prev_sample`` to the model output dtype.

    PAN2 keeps float32 latents by stepping with ``noise_pred.float()``; a bf16
    noise prediction would silently drop the latents to bf16 on either side.
    """
    reference = _reference(pan2_root)
    reference.set_timesteps(sigmas=_pan2_sigmas(2), device="cpu")
    fastvideo = _fastvideo(pan2_root)
    _prepare_fastvideo_timesteps(fastvideo, 2)
    latents = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(5))
    noise_pred = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(6)).to(torch.bfloat16)
    expected = reference.step(noise_pred, reference.timesteps[0], latents).prev_sample
    actual = fastvideo.step(noise_pred, fastvideo.timesteps[0], latents).prev_sample
    assert expected.dtype == actual.dtype == torch.bfloat16
    assert_close(actual, expected, atol=0, rtol=0)
