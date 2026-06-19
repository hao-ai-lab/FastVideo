"""FLUX.2 flow-match sigma schedule — the BFL empirical-mu grid (recipe-local; NOT in v2/loop/sampler.py).

FLUX.2 does NOT use ``FlowShiftPolicy.build_schedule`` (the Wan/LTX resolution-bucket shift). It builds a
resolution-dependent ``mu`` from the *packed* image token count and feeds it, with a custom
``sigmas = linspace(1, 1/N, N)`` grid, to diffusers' ``FlowMatchEulerDiscreteScheduler.set_timesteps``
under ``use_dynamic_shifting=True``. This module reproduces that scheduler's numpy result exactly.

Faithful to:
  * ``fastvideo/pipelines/basic/flux_2/flux_2_timestep_preparation.py:compute_empirical_mu`` (the BFL
    ``sampling.compute_empirical_mu``), and the ``sigmas = np.linspace(1.0, 1/N, N)`` set there;
  * ``fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py:set_timesteps`` with
    ``use_dynamic_shifting=True`` (-> ``time_shift(mu, 1.0, sigmas)``, exponential ``time_shift_type``)
    and the appended terminal ``0.0``.

The exponential time-shift is ``sigma' = exp(mu) / (exp(mu) + (1/sigma - 1)**1.0)`` (the ``sigma=1.0``
arg in the scheduler's ``time_shift(mu, 1.0, t)`` call collapses the ``(1/t - 1)**sigma`` exponent to 1).
"""
from __future__ import annotations

import numpy as np


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Resolution-dependent ``mu`` for the FLUX.2 flow-match scheduler (BFL official ``compute_empirical_mu``).

    ``image_seq_len`` is the PACKED image token count ``latent_h * latent_w`` (half-spatial, see
    ``flux2_latent_geometry``). Faithful copy of ``Flux2TimestepPreparationStage.compute_empirical_mu``.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _time_shift_exponential(mu: float, sigmas: np.ndarray) -> np.ndarray:
    """Diffusers exponential time-shift with ``sigma=1.0`` (the scheduler's ``time_shift(mu, 1.0, t)``)."""
    return np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))


def flux2_sigmas(num_steps: int, image_seq_len: int) -> np.ndarray:
    """The FLUX.2 σ schedule: ``num_steps+1`` points (the shifted grid + a terminal ``0.0``).

    1. base grid ``sigmas = linspace(1.0, 1/N, N)`` (the custom grid the pipeline passes);
    2. resolution-dependent ``mu`` shift ``time_shift(mu, 1.0, sigmas)`` (use_dynamic_shifting=True);
    3. append the terminal ``0.0`` (FlowMatchEuler's ``invert_sigmas=False`` tail).
    The loop integrates these pairwise via the flow-match Euler step (FLOW_MATCH_STEP).
    """
    n = max(1, int(num_steps))
    base = np.linspace(1.0, 1.0 / n, n, dtype=np.float64)
    mu = compute_empirical_mu(int(image_seq_len), n)
    shifted = _time_shift_exponential(mu, base)
    return np.concatenate([shifted, np.zeros(1, dtype=np.float64)]).astype(np.float64)
