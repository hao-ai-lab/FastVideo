"""Matrix-Game 2.0 DMD sampler helpers — the few-step distilled epsilon->x0 math, ported from
``fastvideo/models/utils.py`` (``pred_noise_to_pred_video`` / ``pred_noise_to_x_bound``) and the
``FlowUniPCMultistepScheduler`` sigma/timestep table it indexes. Kept recipe-local (not in
``v2/loop/sampler.py``) because the DMD epsilon->x0 + re-add_noise schedule is specific to this arch.

The distilled checkpoints run a FIXED 3-step DMD schedule ``[1000,666,333]``. The scheduler
(``FlowUniPCMultistepScheduler(shift=5.0)``) is used ONLY as a sigma/timestep lookup table + ``add_noise``,
never as a UniPC multistep corrector. With ``warp_denoising_step=True`` the nominal integer steps are
remapped through its timestep grid: ``timesteps = cat(scheduler.timesteps, [0])[1000 - dmd_steps]``.

This numpy form is the CPU-testable, bit-reproducible reference. ``build_flow_unipc_table`` reproduces the
scheduler's ``sigmas``/``timesteps`` (flow-shift over a 1000-step alpha ramp; same formula as
``FlowShiftPolicy``) so the loop runs on the CPU toy without the real scheduler. On GPU the torch scheduler
is built from the card's ``scheduler`` load_id and the SAME math applies.
"""
from __future__ import annotations

import numpy as np

# Matrix-Game 2.0 distilled hyper-params (fastvideo MatrixGame2I2V480PConfig + MatrixGame2WanVideoArchConfig).
MATRIXGAME2_FLOW_SHIFT = 5.0
MATRIXGAME2_DMD_STEPS: tuple[int, ...] = (1000, 666, 333)
MATRIXGAME2_NUM_TRAIN_TIMESTEPS = 1000
MATRIXGAME2_CONTEXT_NOISE = 0
MATRIXGAME2_NUM_FRAMES_PER_BLOCK = 3


def build_flow_unipc_table(shift: float = MATRIXGAME2_FLOW_SHIFT,
                           num_train_timesteps: int = MATRIXGAME2_NUM_TRAIN_TIMESTEPS) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``FlowUniPCMultistepScheduler.__init__`` (scheduling_flow_unipc_multistep.py): a descending
    1000-step alpha ramp ``alphas = linspace(1, 1/N, N)[::-1]``, ``sigmas = 1 - alphas``, flow-shifted
    ``sigma' = shift*sigma / (1 + (shift-1)*sigma)``, then ``timesteps = sigma' * N``. Returns
    ``(sigmas, timesteps)`` as float64, the table the epsilon->x0 conversion indexes by nearest timestep."""
    alphas = np.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps, dtype=np.float64)[::-1].copy()
    sigmas = 1.0 - alphas
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps


def warp_dmd_timesteps(dmd_steps: tuple[int, ...] = MATRIXGAME2_DMD_STEPS,
                       *,
                       scheduler_timesteps: np.ndarray | None = None,
                       shift: float = MATRIXGAME2_FLOW_SHIFT,
                       num_train_timesteps: int = MATRIXGAME2_NUM_TRAIN_TIMESTEPS) -> np.ndarray:
    """``warp_denoising_step``: remap the nominal DMD steps through the scheduler timestep grid, faithful to
    ``MatrixGame2CausalDenoisingStage.forward``: ``timesteps = cat(scheduler.timesteps, [0])[1000 - steps]``.
    The scheduler grid here is the FULL 1000-step table (set_timesteps with num_train_timesteps steps gives
    the same grid the pipeline's LatentPreparationStage builds before warping)."""
    if scheduler_timesteps is None:
        _sig, scheduler_timesteps = build_flow_unipc_table(shift=shift, num_train_timesteps=num_train_timesteps)
    grid = np.concatenate([np.asarray(scheduler_timesteps, dtype=np.float64), np.array([0.0])])
    idx = num_train_timesteps - np.asarray(dmd_steps, dtype=np.int64)
    return grid[idx]


def _sigma_at(timestep: float, sigmas: np.ndarray, timesteps: np.ndarray) -> float:
    """Nearest-neighbor sigma lookup: ``sigma_t = sigmas[argmin|timesteps - timestep|]`` — exactly the
    ``torch.argmin((timesteps - timestep).abs())`` in ``pred_noise_to_pred_video``."""
    idx = int(np.argmin(np.abs(np.asarray(timesteps, dtype=np.float64) - float(timestep))))
    return float(sigmas[idx])


def pred_noise_to_pred_video(pred_noise: np.ndarray, noise_input_latent: np.ndarray, timestep: float,
                             sigmas: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    """epsilon -> x0: ``pred_video = noise_input_latent - sigma_t * pred_noise`` (single-expert path).
    Faithful to ``fastvideo/models/utils.py:pred_noise_to_pred_video``."""
    sigma_t = _sigma_at(timestep, sigmas, timesteps)
    return np.asarray(noise_input_latent, dtype=np.float64) - sigma_t * np.asarray(pred_noise, dtype=np.float64)


def pred_noise_to_x_bound(pred_noise: np.ndarray, noise_input_latent: np.ndarray, timestep: float,
                          boundary_timestep: float, sigmas: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    """High-noise-expert epsilon -> x0: ``pred_video = x - (sigma_t - sigma_boundary) * pred_noise``.
    Faithful to ``fastvideo/models/utils.py:pred_noise_to_x_bound`` (MoE distilled variants only; the Base
    distilled checkpoint has ``boundary_ratio=None`` -> this path is unused)."""
    sigma_t = _sigma_at(timestep, sigmas, timesteps)
    sigma_b = _sigma_at(boundary_timestep, sigmas, timesteps)
    return (np.asarray(noise_input_latent, dtype=np.float64) -
            (sigma_t - sigma_b) * np.asarray(pred_noise, dtype=np.float64))


def add_noise(x0: np.ndarray, noise: np.ndarray, timestep: float, sigmas: np.ndarray,
              timesteps: np.ndarray) -> np.ndarray:
    """Flow-match forward interpolation at the NEXT DMD timestep: ``x_t = (1 - sigma_t)*x0 + sigma_t*noise``
    — the scheduler's ``add_noise`` (re-noise between the 3 DMD steps). ``sigma_t`` is looked up at the next
    timestep, matching ``scheduler.add_noise(pred_video, noise, next_timestep)``."""
    sigma_t = _sigma_at(timestep, sigmas, timesteps)
    return (1.0 - sigma_t) * np.asarray(x0, dtype=np.float64) + sigma_t * np.asarray(noise, dtype=np.float64)
