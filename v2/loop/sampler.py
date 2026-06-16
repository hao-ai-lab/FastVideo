"""Samplers as a small library (design_v3 §5.1: families use samplers as a library).

Flow-match Euler is shared by the Wan and LTX-2 denoise step bodies. On a GPU box the real
UniPC multistep / distilled schedulers are wrapped by the torch component adapters; this
numpy form is the CPU-testable, bit-reproducible reference used by the loop tests.

Flow-match interpolation: ``x_t = (1-σ_t)·x0 + σ_t·ε``; the model predicts velocity
``v = ε - x0``. Then a deterministic Euler step to σ_next is simply ``x_next = x_t + (σ_next-σ_t)·v``
(this is exactly LTX-2's ``velocity=(latents-denoised)/σ; latents += velocity·dt``).
"""
from __future__ import annotations

import numpy as np


def build_flow_sigmas(num_steps: int, shift: float = 1.0, terminal: float = 0.0) -> np.ndarray:
    """σ schedule from 1→terminal over ``num_steps+1`` points, with flow-shift applied.

    Flow-shift: ``σ' = shift·σ / (1 + (shift-1)·σ)`` (Wan/LTX). shift=1 is the identity.
    """
    base = np.linspace(1.0, terminal, num_steps + 1, dtype=np.float64)
    if shift != 1.0:
        base = shift * base / (1.0 + (shift - 1.0) * base)
    return base


def x0_from_velocity(x_t: np.ndarray, velocity: np.ndarray, sigma_t: float) -> np.ndarray:
    """x0 = x_t - σ_t·v (flow prediction → clean sample; Wan ``x0 = x_t - σ·model_output``)."""
    return x_t - sigma_t * velocity


def flow_match_euler_step(x_t: np.ndarray, velocity: np.ndarray,
                          sigma_t: float, sigma_next: float) -> np.ndarray:
    """One deterministic Euler step along the flow ODE."""
    return x_t + (sigma_next - sigma_t) * velocity


def add_noise(x0: np.ndarray, noise: np.ndarray, sigma: float) -> np.ndarray:
    """Forward flow-match interpolation: x_σ = (1-σ)·x0 + σ·noise."""
    return (1.0 - sigma) * x0 + sigma * noise
