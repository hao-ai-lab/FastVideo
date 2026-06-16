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


def flow_sde_step_with_logprob(x_t, velocity, sigma_t: float, sigma_next: float, *,
                               noise=None, prev_sample=None, noise_scale: float = 0.7):
    """FlowGRPO-style stochastic step + Gaussian log-prob (PromptRL/UniRL §6; FlowGRPO lineage).

    The RL *rollout* sampler (vs the deterministic ODE ``flow_match_euler_step`` used at serve time).
    Injects noise for exploration and returns the log-prob of the realized sample under the step's
    Gaussian — the quantity the FlowGRPO PPO ratio uses. ``prev_sample=None`` samples a new step;
    passing ``prev_sample`` (the rollout sample) recomputes its log-prob under the *current* velocity
    (the ratio's numerator at update time). Returns ``(prev_sample, log_prob, mean, eff_std)``.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    s = min(float(sigma_t), 0.9999)
    dt = float(sigma_next) - float(sigma_t)                       # negative (σ decreases)
    std = float(np.sqrt(s / (1.0 - s)) * noise_scale)
    denom = 2.0 * max(s, 1e-6)
    mean = x_t * (1.0 + std ** 2 / denom * dt) + velocity * (1.0 + std ** 2 * (1.0 - s) / denom) * dt
    eff_std = max(std * np.sqrt(max(-dt, 1e-12)), 1e-6)
    if prev_sample is None:
        n = noise if noise is not None else np.zeros_like(x_t)
        prev_sample = mean + eff_std * np.asarray(n, dtype=np.float64)
    prev_sample = np.asarray(prev_sample, dtype=np.float64)
    var = eff_std ** 2
    log_prob = float(np.mean(-((prev_sample - mean) ** 2) / (2.0 * var)
                             - np.log(eff_std) - 0.5 * np.log(2.0 * np.pi)))
    return prev_sample.astype("float32"), log_prob, mean.astype("float32"), float(eff_std)
