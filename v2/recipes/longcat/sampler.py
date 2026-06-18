"""LongCat sigma schedule — the recipe-local sampler helper (kept HERE, not in the shared
``v2/loop/sampler.py``, so the port stays a self-contained package).

LongCat's ``get_timesteps_sigmas`` (non-distill, the base T2V path) builds the inference sigmas with
``torch.linspace(1.0, 0.001, num_inference_steps)`` and hands them to ``FlowMatchEulerDiscreteScheduler``;
the scheduler appends a trailing ``0.0`` (its final-step boundary). This is NOT the flow-shift linspace Wan
uses (``build_flow_sigmas`` gives ``num_steps+1`` points terminating at 0.0). Faithful to
``fastvideo/pipelines/stages/longcat_refine_timestep.py`` lines 63-69 (which documents the base schedule) and
the scheduler's terminal-sigma append.
"""
from __future__ import annotations

import numpy as np


def longcat_linspace_sigmas(num_steps: int, sigma_min: float = 0.001) -> np.ndarray:
    """The LongCat inference sigma schedule: ``linspace(1.0, sigma_min, num_steps)`` (NOT terminal 0) with a
    trailing ``0.0`` appended for the final Euler boundary — giving ``num_steps+1`` entries the loop integrates
    pairwise (``num_steps`` actual denoise steps). ``sigma_min`` is LongCat's 0.001 terminal."""
    n = max(int(num_steps), 1)
    ramp = np.linspace(1.0, float(sigma_min), n, dtype=np.float64)  # num_steps points, 1.0 -> 0.001
    return np.concatenate([ramp, np.zeros(1, dtype=np.float64)])  # + the scheduler's trailing-0 boundary
