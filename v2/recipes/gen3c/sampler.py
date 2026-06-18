"""GEN3C EDM sampler helpers (kept local to the recipe — NOT in ``v2/loop/sampler.py``).

GEN3C's pipeline FORCES a diffusers ``EDMEulerScheduler(sigma_max=80, sigma_min=0.0002, sigma_data=0.5)``
at runtime (``Gen3CPipeline.initialize_pipeline``), overriding whatever the converted model_index points
at. That scheduler's defaults are ``sigma_schedule='karras'`` (ρ=7) and ``final_sigmas_type='zero'``, so:

  * the ramp is the Karras ρ-interpolation ``σ_max → σ_min`` (verified bit-identical to
    ``v2.loop.sampler.build_karras_sigmas`` on the ramp), and
  * the appended terminal σ is **0.0** (``final_sigmas_type='zero'``), NOT ``σ_min``. This is the one
    difference from the Cosmos recipe (Cosmos clamps the terminal to σ_min); GEN3C's last Euler step
    integrates from ``σ[-2]=σ_min`` to ``0.0`` (the velocity still divides by ``σ[-2]>0``, so no NaN).

``edm_init_sigma`` is the EDMEulerScheduler ``init_noise_sigma = √(σ_max² + σ_data²)`` used to scale the
initial Gaussian latent (``Gen3CLatentPreparationStage`` multiplies ``randn`` by it).
"""
from __future__ import annotations

import numpy as np


def build_edm_euler_sigmas(num_steps: int,
                           sigma_max: float = 80.0,
                           sigma_min: float = 0.0002,
                           rho: float = 7.0) -> np.ndarray:
    """The diffusers ``EDMEulerScheduler`` σ schedule for GEN3C: a length-``num_steps`` Karras ρ ramp
    ``σ_max → σ_min`` then a single appended terminal ``0.0`` (``final_sigmas_type='zero'``) — giving
    ``num_steps+1`` points the EDM loop integrates pairwise."""
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    min_inv_rho = sigma_min**(1.0 / rho)
    max_inv_rho = sigma_max**(1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho  # σ_max -> σ_min (Karras ρ)
    return np.concatenate([sigmas, np.zeros(1)]).astype(np.float64)  # + the 0.0 terminal (final='zero')


def edm_init_sigma(sigma_max: float = 80.0, sigma_data: float = 0.5) -> float:
    """``EDMEulerScheduler.init_noise_sigma = √(σ_max² + σ_data²)`` — the initial-latent noise scale."""
    return float((sigma_max**2 + sigma_data**2)**0.5)
