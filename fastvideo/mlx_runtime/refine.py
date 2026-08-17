# SPDX-License-Identifier: Apache-2.0
"""Two-pass spatial refine for the MLX Wan runtime (H3 / LTX-2 pattern).

Biggest quality lever on Apple Silicon without a new model or training:
generate at base resolution, then run a second denoising pass with the
*same* DiT at a higher resolution.

This is the MLX-side port of the CUDA refine template in
``fastvideo/pipelines/basic/ltx2/stages/ltx2_refine.py`` and the H3
"base + regenerate" pattern documented in
``docs/design/mac_qad_two_product_strategy.md``:

1. :func:`plan_refine_resolutions` — split the request into stage-1
   (base) and stage-2 (target) pixel sizes, validating VAE / patch
   alignment the way :class:`LTX2RefineInitStage` does.
2. :func:`upsample_latents_spatial` — 2× (or N×) spatial upsample of
   clean latents. Wan has no learned latent upsampler on Mac, so this
   is bilinear over the H×W plane (temporal axis untouched) — same
   role as LTX-2's ``upsample_video`` hand-off, without the learned
   residual.
3. :func:`prepare_refine_latents` — upsample + re-noise the clean
   stage-1 latents to the stage-2 sigma so the second denoise has
   something to refine (mirrors :class:`LTX2UpsampleStage` +
   ``apply_ltx2_gaussian_noiser``).
4. :func:`run_two_pass_dmd` — orchestrate stage-1 denoise → refine
   hand-off → stage-2 denoise with the same model / prompt embeds.

No LoRA swap, no dedicated SR weights, no new training — pure pipeline
work reusable by Wan2.1-14B and Wan2.2-5B on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Sequence

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, add_noise, dmd_step

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mlx.core as mx

logger = init_logger(__name__)

# Default stage-2 noise level when the caller does not supply a schedule.
# Matches the first entry of LTX-2's STAGE_2_DISTILLED_SIGMA_VALUES in spirit
# (start the refine denoise from a high-noise level) without hard-wiring the
# LTX-2 distilled grid onto Wan's flow-match schedule.
DEFAULT_REFINE_SIGMA = 0.909375


@dataclass(frozen=True)
class RefinePlan:
    """Resolved stage-1 / stage-2 geometry for a two-pass refine run."""

    target_height: int
    target_width: int
    stage1_height: int
    stage1_width: int
    spatial_scale: int
    vae_spatial_compression: int
    vae_temporal_compression: int
    num_frames: int

    @property
    def stage1_latent_height(self) -> int:
        """Return the stage-1 latent height after VAE spatial compression."""
        return self.stage1_height // self.vae_spatial_compression

    @property
    def stage1_latent_width(self) -> int:
        """Return the stage-one latent width after VAE spatial compression."""
        return self.stage1_width // self.vae_spatial_compression

    @property
    def stage2_latent_height(self) -> int:
        """Calculate the target-resolution latent height.

        Returns:
            int: The target height divided by the VAE spatial compression factor.
        """
        return self.target_height // self.vae_spatial_compression

    @property
    def stage2_latent_width(self) -> int:
        """Return the target image width in latent-space units."""
        return self.target_width // self.vae_spatial_compression

    @property
    def latent_frames(self) -> int:
        """Calculate the number of latent frames after VAE temporal compression.

        Returns:
            int: The compressed latent frame count.
        """
        return (self.num_frames - 1) // self.vae_temporal_compression + 1


def plan_refine_resolutions(
        *,
        height: int,
        width: int,
        num_frames: int,
        spatial_scale: int = 2,
        vae_spatial_compression: int = 8,
        vae_temporal_compression: int = 4,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        enabled: bool = True,
        mode_label: str = "Refine",
) -> RefinePlan:
    """
    Validate the requested dimensions and create the stage-1 and target-resolution refinement plan.

    Parameters:
        height (int): Target image height in pixels.
        width (int): Target image width in pixels.
        num_frames (int): Number of frames in the input sequence.
        spatial_scale (int): Factor used to reduce spatial dimensions for stage 1.
        vae_spatial_compression (int): Spatial compression factor of the VAE.
        vae_temporal_compression (int): Temporal compression factor of the VAE.
        patch_size (tuple[int, int, int]): Temporal and spatial patch dimensions used to validate latent-grid alignment.
        enabled (bool): Whether to use two-pass refinement.
        mode_label (str): Name of the calling mode, used to prefix validation
            errors so ``--fast-spatial`` failures do not read as refine failures.

    Returns:
        RefinePlan: The validated stage-1 and target-resolution plan.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be positive, got {height}x{width}")
    if spatial_scale < 1:
        raise ValueError(f"spatial_scale must be >= 1, got {spatial_scale}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if vae_spatial_compression < 1 or vae_temporal_compression < 1:
        raise ValueError("VAE compression factors must be positive")
    if height % vae_spatial_compression != 0 or width % vae_spatial_compression != 0:
        raise ValueError(f"height/width must be divisible by vae_spatial_compression={vae_spatial_compression} "
                         f"(got {height}x{width}).")
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(f"num_frames must be 1 modulo vae_temporal_compression={vae_temporal_compression} "
                         f"(got {num_frames}).")

    if not enabled or spatial_scale == 1:
        plan = RefinePlan(
            target_height=height,
            target_width=width,
            stage1_height=height,
            stage1_width=width,
            spatial_scale=1,
            vae_spatial_compression=vae_spatial_compression,
            vae_temporal_compression=vae_temporal_compression,
            num_frames=num_frames,
        )
        _validate_plan(plan, patch_size=patch_size, mode_label=mode_label)
        return plan

    if height % spatial_scale != 0 or width % spatial_scale != 0:
        raise ValueError(f"{mode_label} requires height/width divisible by spatial_scale={spatial_scale} "
                         f"(got {height}x{width}).")

    stage1_height = height // spatial_scale
    stage1_width = width // spatial_scale
    # Stage-1 must land on a VAE-aligned grid so the first denoise produces
    # valid latents; the LTX-2 init stage enforces the same constraint.
    if (stage1_height % vae_spatial_compression != 0 or stage1_width % vae_spatial_compression != 0):
        raise ValueError(f"{mode_label} requires height/width divisible by "
                         f"{spatial_scale * vae_spatial_compression} "
                         f"(got {height}x{width}, vae_spatial={vae_spatial_compression}).")

    plan = RefinePlan(
        target_height=height,
        target_width=width,
        stage1_height=stage1_height,
        stage1_width=stage1_width,
        spatial_scale=spatial_scale,
        vae_spatial_compression=vae_spatial_compression,
        vae_temporal_compression=vae_temporal_compression,
        num_frames=num_frames,
    )
    _validate_plan(plan, patch_size=patch_size, mode_label=mode_label)
    logger.info(
        "[MLX refine] enabled: stage1=%dx%d stage2=%dx%d scale=%dx",
        stage1_width,
        stage1_height,
        width,
        height,
        spatial_scale,
    )
    return plan


def _validate_plan(plan: RefinePlan, *, patch_size: tuple[int, int, int], mode_label: str = "Refine") -> None:
    """
    Validate that both refinement stages have latent dimensions aligned to the patch grid.

    Parameters:
        patch_size (tuple[int, int, int]): Temporal, height, and width patch dimensions.

    Raises:
        ValueError: If a stage's spatial latent dimensions or the temporal latent
            dimension is not divisible by the corresponding patch dimension.
    """
    pt, ph, pw = patch_size
    for label, lh, lw in (
        ("stage1", plan.stage1_latent_height, plan.stage1_latent_width),
        ("stage2", plan.stage2_latent_height, plan.stage2_latent_width),
    ):
        if lh % ph != 0 or lw % pw != 0:
            raise ValueError(f"{mode_label} {label} latent grid {lh}x{lw} is not divisible by "
                             f"patch spatial size {ph}x{pw}.")
        if plan.latent_frames % pt != 0:
            raise ValueError(f"{mode_label} latent_frames={plan.latent_frames} is not divisible by "
                             f"patch temporal size {pt}.")


def upsample_latents_spatial(
    latents: Any,
    *,
    scale: int = 2,
    mode: str = "bilinear",
) -> Any:
    """
    Upsample the spatial dimensions of 5-D latent arrays while preserving the batch, channel, and temporal dimensions.

    Parameters:
        latents (Any): Latents with shape ``(B, C, T, H, W)``.
        scale (int): Integer factor for enlarging the spatial dimensions.
        mode (str): Interpolation mode, either ``"nearest"`` or ``"bilinear"``.

    Returns:
        Any: Latents with shape ``(B, C, T, H * scale, W * scale)``.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return latents

    # Accept both mx.array and np.ndarray so unit tests can run without MLX.
    is_mlx = hasattr(latents, "dtype") and type(latents).__module__.startswith("mlx")
    if is_mlx:
        return _upsample_latents_mlx(latents, scale=scale, mode=mode)
    return _upsample_latents_numpy(np.asarray(latents), scale=scale, mode=mode)


def _upsample_latents_numpy(
    latents: np.ndarray,
    *,
    scale: int,
    mode: str,
) -> np.ndarray:
    """Upsample 5-D latent arrays spatially using nearest-neighbor or bilinear interpolation.

    Parameters:
        latents (np.ndarray): Latents with shape ``(B, C, T, H, W)``.
        scale (int): Spatial upsampling factor.
        mode (str): Interpolation mode, either ``"nearest"`` or ``"bilinear"``.

    Returns:
        np.ndarray: Spatially upsampled latents with preserved batch, channel, and temporal dimensions.

    Raises:
        ValueError: If the latents are not five-dimensional or the interpolation mode is unsupported.
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected 5-D latents (B,C,T,H,W), got shape {latents.shape}")
    b, c, t, h, w = latents.shape
    if mode == "nearest":
        # (B,C,T,H,1,W,1) -> broadcast to (B,C,T,H,scale,W,scale) -> merge.
        out = np.repeat(np.repeat(latents, scale, axis=3), scale, axis=4)
        return out

    if mode != "bilinear":
        raise ValueError(f"Unsupported upsample mode: {mode}")

    # Bilinear over the spatial plane. Flatten (B,C,T) into a batch of 2-D
    # maps so a single vectorized gather covers every frame/channel.
    src = latents.reshape(b * c * t, h, w).astype(np.float32, copy=False)
    out_h, out_w = h * scale, w * scale
    # Map output pixel centers onto the input grid (align_corners=False).
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0.0, h - 1.0)
    xs = np.clip(xs, 0.0, w - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(np.float32))[:, None]
    wx = (xs - x0.astype(np.float32))[None, :]
    # Gather the four corners: shape (N, out_h, out_w).
    Ia = src[:, y0[:, None], x0[None, :]]
    Ib = src[:, y0[:, None], x1[None, :]]
    Ic = src[:, y1[:, None], x0[None, :]]
    Id = src[:, y1[:, None], x1[None, :]]
    wa = (1.0 - wy) * (1.0 - wx)
    wb = (1.0 - wy) * wx
    wc = wy * (1.0 - wx)
    wd = wy * wx
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.reshape(b, c, t, out_h, out_w).astype(latents.dtype, copy=False)


def _upsample_latents_mlx(latents: mx.array, *, scale: int, mode: str) -> mx.array:
    """
    Upsample MLX latent tensors along their spatial dimensions.

    Parameters:
        latents (mx.array): A latent tensor with shape `(B, C, T, H, W)`.
        scale (int): The integer spatial upsampling factor.
        mode (str): The interpolation mode, such as `"nearest"` or `"bilinear"`.

    Returns:
        mx.array: The spatially upsampled latent tensor with its original data type.
    """
    import mlx.core as mx

    # Route through NumPy for the interpolation math. Latent tensors at Mac
    # resolutions are small (e.g. 1×16×21×30×52 ≈ 1 MB) so the host hop is
    # cheaper than carrying a bespoke Metal bilinear kernel, and it keeps
    # the CPU-only unit tests and the MLX path on one implementation.
    np_latents = np.array(latents.astype(mx.float32))
    up = _upsample_latents_numpy(np_latents, scale=scale, mode=mode)
    return mx.array(up).astype(latents.dtype)


def prepare_refine_latents(
    clean_latents: Any,
    *,
    scale: int = 2,
    sigma: float = DEFAULT_REFINE_SIGMA,
    noise: Any | None = None,
    add_noise_flag: bool = True,
    upsample_mode: str = "bilinear",
    seed: int | None = None,
) -> Any:
    """
    Upsample clean latents spatially and optionally mix them with Gaussian noise.

    Parameters:
        clean_latents: The stage-1 latent tensor.
        sigma: Noise mixing factor between 0 and 1.
        noise: Optional noise tensor to mix with the upsampled latents.
        add_noise_flag: Whether to apply noise mixing.
        upsample_mode: Spatial interpolation mode.
        seed: Optional seed for generated noise.

    Returns:
        The upsampled latents, optionally mixed with noise.

    Raises:
        ValueError: If sigma is outside the range from 0 to 1.
    """
    if sigma < 0.0 or sigma > 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")

    upsampled = upsample_latents_spatial(clean_latents, scale=scale, mode=upsample_mode)
    if not add_noise_flag or sigma == 0.0:
        return upsampled

    is_mlx = hasattr(upsampled, "dtype") and type(upsampled).__module__.startswith("mlx")
    if noise is None:
        noise = _draw_noise_like(upsampled, seed=seed, is_mlx=is_mlx)
    return add_noise(upsampled, noise, float(sigma))


def refine_sigma_from_schedule(
    schedule: MLXDMDSchedule,
    timesteps: Sequence[float | int],
) -> float:
    """Derive the refinement noise level from the first refinement timestep.

    Parameters:
        schedule (MLXDMDSchedule): Schedule used to map timesteps to noise levels.
        timesteps (Sequence[float | int]): Refinement timesteps, whose first value determines the sigma.

    Returns:
        float: Sigma corresponding to the first refinement timestep.

    Raises:
        ValueError: If `timesteps` is empty.
    """
    if not timesteps:
        raise ValueError("timesteps must be non-empty to derive a refine sigma")
    return float(schedule.sigma_for(float(timesteps[0])))


def default_refine_timesteps(
    schedule: MLXDMDSchedule,
    timesteps: Sequence[float | int],
) -> list[float]:
    """Derive stage-2 timesteps from the stage-1 DMD grid.

    The stage-2 pass must start *below* full noise, otherwise the hand-off
    ``(1 - sigma) * upsampled + sigma * noise`` weights stage 1 at zero and
    the refine pass silently becomes a plain full-resolution generation at
    twice the cost. FastWan's stage-1 grid opens at ``t=1000`` (``sigma``
    exactly 1.0), so reusing it verbatim — which is what happens when
    ``--refine-dmd-denoising-steps`` is left unset — discards stage 1.

    Dropping the leading full-noise entries keeps the pass on timesteps the
    distilled student was actually trained on (no off-grid ``t`` the DiT has
    never seen) while letting the stage-1 structure through.

    Parameters:
        schedule (MLXDMDSchedule): Schedule used to map timesteps to noise levels.
        timesteps (Sequence[float | int]): The stage-1 DMD timestep grid.

    Returns:
        list[float]: The stage-1 grid with leading full-noise timesteps removed.

    Raises:
        ValueError: If every timestep in the grid is at full noise, leaving no
            usable refine step.
    """
    steps = [float(step) for step in timesteps]
    first = 0
    while first < len(steps) and schedule.sigma_for(steps[first]) >= 1.0:
        first += 1
    if first == len(steps):
        raise ValueError(f"No usable refine timesteps in {steps}: every entry is at sigma >= 1 "
                         "(full noise), which would discard the stage-1 result. Pass "
                         "explicit stage-2 timesteps below the full-noise step.")
    return steps[first:]


def run_dmd_loop(
    *,
    dit: Any,
    latents: Any,
    encoder_hidden_states: Any,
    freqs_cis: tuple[Any, Any],
    timesteps: Sequence[float | int],
    schedule: MLXDMDSchedule,
    mx_dtype: Any,
    seed: int | None = None,
    step_callback: Callable[[int, int], None] | None = None,
    label: str = "denoise",
) -> Any:
    """
    Denoise latents over the supplied timesteps using the DMD schedule.

    Parameters:
        timesteps (Sequence[float | int]): Denoising timesteps in execution order.
        seed (int | None): Seed for reproducible intermediate noise generation.
        step_callback (Callable[[int, int], None] | None): Callback receiving the
            completed step number and total step count.
        label (str): Label used for progress output when no callback is provided.

    Returns:
        Any: The denoised latents.
    """
    import mlx.core as mx

    renoise_rng = np.random.default_rng(seed) if seed is not None else None
    latents_out = latents
    n_steps = len(timesteps)
    for step_index, timestep in enumerate(timesteps):
        noise_input = latents_out
        ts_val = float(timestep)
        timestep_mx = mx.array([ts_val]).astype(mx.float32)
        noise_pred = dit(
            latents_out.astype(mx_dtype),
            encoder_hidden_states,
            timestep_mx,
            freqs_cis,
        )
        noise_input_f32 = noise_input.astype(mx.float32)
        pred_noise_f32 = noise_pred.astype(mx.float32)
        if step_index < n_steps - 1:
            next_ts: float | None = float(timesteps[step_index + 1])
            if renoise_rng is not None:
                renoise = mx.array(renoise_rng.standard_normal(tuple(noise_input_f32.shape)).astype(np.float32))
            else:
                renoise = mx.random.normal(noise_input_f32.shape).astype(mx.float32)
        else:
            next_ts, renoise = None, None
        latents_out = dmd_step(
            latents=noise_input_f32,
            noise_input_latent=noise_input_f32,
            pred_noise=pred_noise_f32,
            schedule=schedule,
            timestep=ts_val,
            next_timestep=next_ts,
            noise=renoise,
        ).astype(mx_dtype)
        mx.eval(latents_out)
        if step_callback is not None:
            step_callback(step_index + 1, n_steps)
        else:
            print(f"{label} step {step_index + 1}/{n_steps} complete")
    return latents_out


@dataclass(frozen=True)
class TwoPassResult:
    """Outputs of :func:`run_two_pass_dmd`."""

    latents: Any
    stage1_latents: Any
    plan: RefinePlan
    refine_sigma: float


def run_two_pass_dmd(
    *,
    dit: Any,
    encoder_hidden_states: Any,
    noise_latents_stage1: Any,
    freqs_cis_stage1: tuple[Any, Any],
    freqs_cis_stage2: tuple[Any, Any] | None,
    plan: RefinePlan,
    schedule: MLXDMDSchedule,
    timesteps: Sequence[float | int],
    refine_timesteps: Sequence[float | int] | None = None,
    mx_dtype: Any,
    seed: int = 0,
    add_noise_flag: bool = True,
    upsample_mode: str = "bilinear",
    refine_sigma: float | None = None,
    step_callback: Callable[[str, int, int], None] | None = None,
) -> TwoPassResult:
    """
    Run base denoising and, when enabled, spatial refinement denoising.

    Parameters:
        dit: DiT callable used for both denoising passes.
        encoder_hidden_states: Prompt embeddings shared across both passes.
        noise_latents_stage1: Initial stage-1 noise latents.
        freqs_cis_stage1: RoPE tables for the stage-1 resolution.
        freqs_cis_stage2: RoPE tables for the stage-2 resolution, required when refinement is enabled.
        plan: Refinement geometry and configuration.
        schedule: Flow-matching schedule used by both passes.
        timesteps: Stage-1 denoising timesteps.
        refine_timesteps: Stage-2 denoising timesteps. Uses `timesteps` when omitted.
        mx_dtype: MLX dtype used for DiT inputs and outputs.
        seed: Base seed for reproducible noise generation.
        add_noise_flag: Whether to add noise to the upsampled stage-1 latents.
        upsample_mode: Spatial upsampling mode, either `"bilinear"` or `"nearest"`.
        refine_sigma: Stage-2 starting noise level. Derived from the first refinement timestep when omitted.
        step_callback: Optional callback receiving the phase name, step index, and total step count.

    Returns:
        TwoPassResult containing the final latents, stage-1 latents, refinement plan, and applied refinement sigma.

    Raises:
        ValueError: If refinement is enabled without stage-2 RoPE tables, without refinement timesteps, or if upsampled latents do not match the planned stage-2 dimensions.
    """
    stage1_cb = None
    stage2_cb = None
    if step_callback is not None:
        stage1_cb = lambda i, n: step_callback("stage1", i, n)  # noqa: E731
        stage2_cb = lambda i, n: step_callback("stage2", i, n)  # noqa: E731

    stage1_latents = run_dmd_loop(
        dit=dit,
        latents=noise_latents_stage1,
        encoder_hidden_states=encoder_hidden_states,
        freqs_cis=freqs_cis_stage1,
        timesteps=timesteps,
        schedule=schedule,
        mx_dtype=mx_dtype,
        seed=seed,
        step_callback=stage1_cb,
        label="stage1 denoise",
    )

    if plan.spatial_scale == 1:
        return TwoPassResult(
            latents=stage1_latents,
            stage1_latents=stage1_latents,
            plan=plan,
            refine_sigma=0.0,
        )

    if freqs_cis_stage2 is None:
        raise ValueError("freqs_cis_stage2 is required when refine spatial_scale > 1")

    if refine_timesteps is not None:
        stage2_timesteps = [float(step) for step in refine_timesteps]
        if not stage2_timesteps:
            raise ValueError("refine_timesteps must be non-empty when refine is enabled")
    else:
        # Not `list(timesteps)`: the stage-1 grid opens at full noise, which
        # would weight the stage-1 result at zero. See default_refine_timesteps.
        stage2_timesteps = default_refine_timesteps(schedule, timesteps)
    grid_sigma = refine_sigma_from_schedule(schedule, stage2_timesteps)
    sigma = float(refine_sigma) if refine_sigma is not None else grid_sigma
    if refine_sigma is not None and abs(sigma - grid_sigma) > 1e-6:
        # The loop tells the DiT `stage2_timesteps[0]`, which implies grid_sigma.
        # Overriding the hand-off noise level breaks that correspondence, so the
        # model is denoising from a level it was not told about. Useful for
        # exploring schedules that bottom out too high, but say so out loud.
        logger.warning(
            "[MLX refine] refine_sigma=%.4f overrides the schedule's %.4f for timestep %g; "
            "the DiT is told a timestep that no longer matches the noise it receives.",
            sigma,
            grid_sigma,
            stage2_timesteps[0],
        )

    # A hand-off at sigma >= 1 is `0 * upsampled + 1 * noise`: stage 1 is
    # thrown away and refine degrades to a plain full-res run at 2x the cost.
    # Fail loudly rather than silently burning the first pass.
    if add_noise_flag and sigma >= 1.0:
        raise ValueError(f"Refine hand-off sigma={sigma:.4f} (from stage-2 timestep "
                         f"{stage2_timesteps[0]:g}) discards the stage-1 result entirely: "
                         "the upsampled latents are weighted (1 - sigma) = 0. Start the "
                         "stage-2 grid below the full-noise timestep, or pass "
                         "add_noise_flag=False to hand off the clean upsample.")

    stage2_input = prepare_refine_latents(
        stage1_latents,
        scale=plan.spatial_scale,
        sigma=sigma,
        add_noise_flag=add_noise_flag,
        upsample_mode=upsample_mode,
        seed=seed + 1,
    )

    # Shape guard: upsampled latents must match the stage-2 RoPE grid.
    expected_h = plan.stage2_latent_height
    expected_w = plan.stage2_latent_width
    got_h, got_w = int(stage2_input.shape[-2]), int(stage2_input.shape[-1])
    if got_h != expected_h or got_w != expected_w:
        raise ValueError(f"Refine upsample produced {got_h}x{got_w} latents, expected "
                         f"{expected_h}x{expected_w} for target "
                         f"{plan.target_height}x{plan.target_width}.")

    logger.info(
        "[MLX refine] stage2 start: latent=%dx%d sigma=%.4f steps=%d",
        expected_w,
        expected_h,
        sigma,
        len(stage2_timesteps),
    )

    stage2_latents = run_dmd_loop(
        dit=dit,
        latents=stage2_input,
        encoder_hidden_states=encoder_hidden_states,
        freqs_cis=freqs_cis_stage2,
        timesteps=stage2_timesteps,
        schedule=schedule,
        mx_dtype=mx_dtype,
        seed=seed + 2,
        step_callback=stage2_cb,
        label="stage2 refine",
    )
    return TwoPassResult(
        latents=stage2_latents,
        stage1_latents=stage1_latents,
        plan=plan,
        refine_sigma=sigma,
    )


__all__ = [
    "DEFAULT_REFINE_SIGMA",
    "RefinePlan",
    "TwoPassResult",
    "default_refine_timesteps",
    "plan_refine_resolutions",
    "prepare_refine_latents",
    "refine_sigma_from_schedule",
    "run_dmd_loop",
    "run_two_pass_dmd",
    "upsample_latents_spatial",
]


def _draw_noise_like(like: Any, *, seed: int | None, is_mlx: bool) -> Any:
    """Generate Gaussian noise with the shape and array type of the input."""
    shape = tuple(int(s) for s in like.shape)
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise_np = rng.standard_normal(shape).astype(np.float32)
        if is_mlx:
            import mlx.core as mx

            return mx.array(noise_np).astype(mx.float32)
        return noise_np.astype(np.asarray(like).dtype, copy=False)
    if is_mlx:
        import mlx.core as mx

        return mx.random.normal(shape).astype(mx.float32)
    return np.random.standard_normal(shape).astype(np.asarray(like).dtype, copy=False)
