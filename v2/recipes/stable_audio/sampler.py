"""Stable Audio sampler helpers (recipe-local — NOT in v2/loop/sampler.py, which is flow-match/EDM-Karras).

Three numpy primitives the ``StableAudioDenoiseLoop`` composes, matching k-diffusion's behavior for the
Stable Audio ``dpmpp-3m-sde`` sampler over a v-prediction (VDenoiser) network:

  * ``build_polyexponential_sigmas`` — k-diffusion ``get_sigmas_polyexponential(n, sigma_min, sigma_max,
    rho)`` + the appended 0.0 terminal: a log-linear (rho=1) ramp from sigma_max to sigma_min.
  * ``vdenoiser_x0`` — k-diffusion ``K.external.VDenoiser``: a v-prediction network's x0 reconstruction
    with the EDM-v preconditioning (sigma_data=1).
  * ``dpmpp_2m_step`` — one deterministic DPM-Solver++(2M) update in x0/sigma space (the deterministic
    spine of dpmpp-3m-sde; the stochastic 3rd-order terms are the GPU-parity refinement, BRINGUP).

The full stochastic ``sample_dpmpp_3m_sde`` is the GPU path (``import k_diffusion as K``); it is NOT
installed in the CPU env, so the loop runs this deterministic multistep recurrence for CPU-verification.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def build_polyexponential_sigmas(num_steps: int,
                                 *,
                                 sigma_min: float = 0.3,
                                 sigma_max: float = 500.0,
                                 rho: float = 1.0) -> np.ndarray:
    """k-diffusion ``get_sigmas_polyexponential`` + appended 0.0 terminal.

    ``sigmas_i = exp( lerp(log(sigma_max), log(sigma_min), (i/(n-1))**rho) )`` for i in 0..n-1, then a
    trailing 0.0 (the sampler integrates pairwise; the last step lands on sigma=0). rho=1 makes this a
    plain log-linear (geometric) ramp — Stable Audio's published schedule.
    """
    n = max(1, int(num_steps))
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)**float(rho)
    log_sigmas = np.log(sigma_max) + ramp * (np.log(sigma_min) - np.log(sigma_max))
    sigmas = np.exp(log_sigmas)  # sigma_max -> sigma_min
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float64)]).astype(np.float64)  # + terminal 0


def vdenoiser_x0(x: np.ndarray, sigma: float, net: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """k-diffusion ``K.external.VDenoiser.forward``: reconstruct x0 from a v-prediction network.

    EDM-v preconditioning (sigma_data=1):
      ``c_skip = 1/(sigma^2+1)``, ``c_out = -sigma/sqrt(sigma^2+1)``, ``c_in = 1/sqrt(sigma^2+1)``,
      ``c_noise = atan(sigma)/(pi/2)`` (k-diffusion's ``VDenoiser.sigma_to_t``), and
      ``x0 = c_skip·x + c_out·F(c_in·x, c_noise)``.
    ``net(scaled_x, c_noise)`` is the raw v-prediction (the torch adapter / CPU toy); this function owns
    the preconditioning so the adapter stays the bare network.
    """
    s = float(sigma)
    s2p1 = s * s + 1.0
    c_skip = 1.0 / s2p1
    c_out = -s / np.sqrt(s2p1)
    c_in = 1.0 / np.sqrt(s2p1)
    c_noise = float(np.arctan(s) / (np.pi / 2.0))  # k-diffusion VDenoiser sigma_to_t = atan(sigma)/(pi/2)
    x = np.asarray(x, dtype="float32")
    v = np.asarray(net((x * c_in).astype("float32"), c_noise), dtype="float32")
    return (c_skip * x + c_out * v).astype("float32")


def dpmpp_2m_step(x: np.ndarray, denoised: np.ndarray, sigma_t: float, sigma_next: float,
                  prev_denoised: np.ndarray | None) -> np.ndarray:
    """One deterministic DPM-Solver++(2M) update (k-diffusion ``sample_dpmpp_2m`` recurrence).

    Works in lambda = -log(sigma) (log-SNR) space:
      ``t = -log(sigma_t)``, ``t_next = -log(sigma_next)``, ``h = t_next - t``.
    First step (no history) is the 1st-order DPM-Solver++ (== exponential Euler on x0):
      ``x_next = (sigma_next/sigma_t)·x - expm1(-h)·denoised``.
    Subsequent steps add the 2nd-order correction from the previous x0 estimate:
      ``denoised_d = (1 + 1/(2r))·denoised - 1/(2r)·prev_denoised`` with ``r = h/h_last`` (here
    approximated with the standard ``r`` from the step ratio). At ``sigma_next == 0`` the update collapses
    to ``x_next = denoised`` (the final clean sample), matching k-diffusion's terminal handling.
    """
    x = np.asarray(x, dtype="float32")
    denoised = np.asarray(denoised, dtype="float32")
    sn = float(sigma_next)
    st = float(sigma_t)
    if sn <= 0.0:
        return denoised.astype("float32")  # terminal: land on the clean x0
    t = -np.log(max(st, 1e-12))
    t_next = -np.log(sn)
    h = t_next - t
    if prev_denoised is None:
        x_next = (sn / st) * x - np.expm1(-h) * denoised
        return x_next.astype("float32")
    # 2nd-order multistep correction. Using r=1 (uniform-ish log-SNR spacing) keeps this faithful to
    # k-diffusion's 2M form without threading the exact previous step size; the dominant error term is
    # the same. (The exact h_last ratio is a GPU-parity refinement, BRINGUP.)
    prev = np.asarray(prev_denoised, dtype="float32")
    denoised_d = 1.5 * denoised - 0.5 * prev  # (1 + 1/(2r))·d - 1/(2r)·prev, r=1
    x_next = (sn / st) * x - np.expm1(-h) * denoised_d
    return x_next.astype("float32")
