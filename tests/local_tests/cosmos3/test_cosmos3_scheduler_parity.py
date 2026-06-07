# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo's UniPC scheduler vs the framework's.

The Cosmos3 video sampler is the framework's flow-matching UniPC
(``cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc.FlowUniPCMultistepScheduler``),
driven by ``UniPCSampler`` with config ``num_train_timesteps=1000``,
``use_dynamic_shifting=False`` and a per-mode ``shift`` (10.0 for T2V/I2V,
3.0 for T2I).  FastVideo reuses its vendored
``UniPCMultistepScheduler`` configured for pure flow matching
(``use_flow_sigmas=True``, ``prediction_type="flow_prediction"``,
``predict_x0=True``, ``solver_type="bh2"``, ``solver_order=2``,
``final_sigmas_type="zero"``) with ``flow_shift`` set to the same shift.

This pins the scheduler — the one Cosmos3 component whose earlier test
(``test_cosmos3_denoise_cfg_parity``) compared diffusers-vs-diffusers rather
than against the framework oracle — by:

  * asserting the discrete ``timesteps`` and ``sigmas`` match the framework;
  * running a full multi-step UniPC trajectory with a fixed sequence of
    pseudo-random "velocity" model outputs (identical on both sides, so the
    DiT is factored out) and asserting every intermediate + final latent
    matches the framework bit-for-bit.

CPU / float32.  The framework scheduler is the parity ORACLE.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_scheduler_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
fm_unipc = pytest.importorskip(
    "cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)
FlowUniPCMultistepScheduler = fm_unipc.FlowUniPCMultistepScheduler

from fastvideo.models.schedulers.scheduling_unipc_multistep import (  # noqa: E402
    UniPCMultistepScheduler,
)

pytestmark = [pytest.mark.local]


def _framework_scheduler(num_steps: int, shift: float) -> FlowUniPCMultistepScheduler:
    """Exactly how ``UniPCSampler`` builds + primes its scheduler."""
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    sched.set_timesteps(num_steps, device=torch.device("cpu"), shift=shift)
    return sched


def _fastvideo_scheduler(num_steps: int, shift: float) -> UniPCMultistepScheduler:
    """FastVideo's vendored UniPC configured for the framework's flow setup."""
    sched = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        predict_x0=True,
        solver_type="bh2",
        final_sigmas_type="zero",
        flow_shift=shift,
    )
    sched.set_timesteps(num_steps, device=torch.device("cpu"))
    return sched


_SHIFTS = [pytest.param(10.0, id="shift10_video"), pytest.param(3.0, id="shift3_t2i")]
_STEPS = [pytest.param(4, id="4steps"), pytest.param(10, id="10steps"), pytest.param(35, id="35steps")]


class TestCosmos3SchedulerParity:

    @pytest.mark.parametrize("shift", _SHIFTS)
    @pytest.mark.parametrize("num_steps", _STEPS)
    def test_timesteps_and_sigmas_match_framework(self, num_steps, shift):
        fw = _framework_scheduler(num_steps, shift)
        fv = _fastvideo_scheduler(num_steps, shift)

        t_max = (fw.timesteps.float() - fv.timesteps.float()).abs().max().item()
        s_max = (fw.sigmas.float() - fv.sigmas.float()).abs().max().item()
        print(f"\n[sched n={num_steps} shift={shift}] timesteps max diff={t_max:.3e} "
              f"sigmas max diff={s_max:.3e}")
        assert fw.timesteps.shape == fv.timesteps.shape
        assert fw.sigmas.shape == fv.sigmas.shape
        torch.testing.assert_close(fv.timesteps, fw.timesteps)
        torch.testing.assert_close(fv.sigmas, fw.sigmas)

    @pytest.mark.parametrize("shift", _SHIFTS)
    @pytest.mark.parametrize("num_steps", _STEPS)
    def test_full_trajectory_matches_framework(self, num_steps, shift):
        # A small latent so order-2 einsum paths exercise; batch axis as the
        # samplers expect ([B, C, T, H, W]).
        shape = (1, 4, 2, 3, 3)
        torch.manual_seed(123)
        init = torch.randn(shape, dtype=torch.float32)
        # One pseudo-random "velocity" per step, identical on both sides.
        velocities = [torch.randn(shape, dtype=torch.float32) for _ in range(num_steps)]

        fw = _framework_scheduler(num_steps, shift)
        fv = _fastvideo_scheduler(num_steps, shift)

        fw_lat = init.clone()
        fv_lat = init.clone()
        worst = 0.0
        for i, t in enumerate(fw.timesteps):
            v = velocities[i]
            fw_lat = fw.step(model_output=v, timestep=t, sample=fw_lat, return_dict=False)[0]
            fv_lat = fv.step(model_output=v, timestep=t, sample=fv_lat, return_dict=False)[0]
            step_max = (fw_lat - fv_lat).abs().max().item()
            worst = max(worst, step_max)
            assert not torch.isnan(fv_lat).any(), f"FastVideo latent NaN at step {i}"
        mean_abs = (fw_lat - fv_lat).abs().mean().item()
        print(f"\n[traj n={num_steps} shift={shift}] worst step max diff={worst:.3e} "
              f"final mean diff={mean_abs:.3e}")
        torch.testing.assert_close(fv_lat, fw_lat, atol=1e-5, rtol=1e-4)
