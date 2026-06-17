"""Torch solver-step kernels for the GPU backend (design_v3 §17).

GROUND TRUTH (from the fastvideo-kernel audit): there is NO fused flow-match / SDE *solver* kernel.
fastvideo-kernel ships only primitives — sparse/tiled attention, RMS/LayerNorm, int8 GEMM/quant.
So the GPU solver step is a plain torch elementwise op with the SAME math as ``loop/sampler.py``'s
numpy reference (the consistency oracle that the accel stand-in already proves matches bit-for-bit).

WRITTEN-NOT-RUN: imports torch; loaded only lazily from ``torch_cuda.py``. Under the current numpy
loop surface these marshal numpy↔torch at the boundary (a torch-native surface would keep the latent
on-device through forward→combine→solver — the perf follow-up, Risk G in GPU_BRINGUP.md).
"""
from __future__ import annotations

import numpy as np
import torch


def _as_torch(a):
    """numpy → cuda torch (passthrough if already torch). Returns (tensor, was_numpy)."""
    if torch.is_tensor(a):
        return a, False
    return torch.as_tensor(np.asarray(a), device="cuda"), True


def _back(t, like_numpy: bool):
    return t.detach().to("cpu", torch.float32).numpy() if like_numpy else t


def flow_match_step(x_t, velocity, sigma_t, sigma_next):
    """One deterministic flow-match Euler step — identical to sampler.flow_match_euler_step."""
    xt, was_np = _as_torch(x_t)
    v, _ = _as_torch(velocity)
    return _back(xt + (float(sigma_next) - float(sigma_t)) * v, was_np)


def flow_sde_step(x_t, velocity, sigma_t, sigma_next, *, noise=None, prev_sample=None, noise_scale=0.7):
    """FlowGRPO stochastic step + Gaussian log-prob — the torch port of
    sampler.flow_sde_step_with_logprob. Returns (prev_sample, log_prob, mean, eff_std)."""
    xt, was_np = _as_torch(x_t)
    v, _ = _as_torch(velocity)
    xt = xt.double()
    v = v.double()
    s = min(float(sigma_t), 0.9999)
    dt = float(sigma_next) - float(sigma_t)
    std = float((s / (1.0 - s)) ** 0.5 * noise_scale)
    denom = 2.0 * max(s, 1e-6)
    mean = xt * (1.0 + std ** 2 / denom * dt) + v * (1.0 + std ** 2 * (1.0 - s) / denom) * dt
    eff_std = max(std * max(-dt, 1e-12) ** 0.5, 1e-6)
    if prev_sample is None:
        n = _as_torch(noise)[0].double() if noise is not None else torch.zeros_like(xt)
        prev = mean + eff_std * n
    else:
        prev = _as_torch(prev_sample)[0].double()
    var = eff_std ** 2
    log_prob = float((-((prev - mean) ** 2) / (2.0 * var)
                      - float(np.log(eff_std)) - 0.5 * float(np.log(2.0 * np.pi))).mean().item())
    return (_back(prev.float(), was_np), log_prob, _back(mean.float(), was_np), float(eff_std))
