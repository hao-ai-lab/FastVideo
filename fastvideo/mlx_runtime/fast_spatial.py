# SPDX-License-Identifier: Apache-2.0
"""Spatial fast mode for the MLX Wan runtime (RIFE's spatial twin).

RIFE ``--fast`` cuts *frames* (temporal). This module cuts *pixels*
(spatial): denoise at ``target // scale``, decode at that size, then
resample the decoded frames up to the target. No second denoise pass —
that is ``--refine`` (quality). The two compose:

* ``--fast-spatial`` alone → speed (≈ scale² fewer tokens)
* ``--refine`` alone → quality two-pass (H3 / LTX-2)
* ``--fast`` + ``--refine`` → fewer frames at base res, full-res refine
* ``--fast`` + ``--fast-spatial`` → fewer frames *and* fewer pixels

The upsample runs in **pixel** space, after the VAE decode. It used to run
in latent space (bilinear over the latent H/W plane, sharing the refine
hand-off primitive) and that is what made spatial fast mode incoherent: an
interpolated Wan latent is off the decoder's manifold, so decode returned
the right silhouette under a smeared veil. ``--refine`` can get away with
the latent-space upsample because a second DMD pass re-denoises the result;
spatial fast mode hands the latent straight to the decoder, so it cannot.
See :mod:`fastvideo.mlx_runtime.frame_upsample` for the full rationale.

MetalFX is intentionally not used: it needs game-engine motion vectors
and depth that diffusion output lacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.frame_upsample import (
    DEFAULT_PIXEL_UPSAMPLE_MODE,
    PIXEL_UPSAMPLE_MODES,
    upsample_frames,
)
from fastvideo.mlx_runtime.refine import RefinePlan, plan_refine_resolutions

logger = init_logger(__name__)

# Resampling from a smaller decode loses high-frequency detail the same way
# RIFE's flow warp does, so spatial fast mode borrows ``--fast``'s remedy: a
# light unsharp pass. 0.4 recovers perceived crispness on Wan2.1 output at 2x
# without the halos that show up by ~0.8.
DEFAULT_FAST_SPATIAL_SHARPEN = 0.4


@dataclass(frozen=True)
class FastSpatialPlan:
    """Resolved geometry for a spatial-fast (upsample-only) run."""

    plan: RefinePlan
    upsample_mode: str
    sharpen: float = DEFAULT_FAST_SPATIAL_SHARPEN

    @property
    def enabled(self) -> bool:
        """
        Determine whether spatial scaling is enabled.

        Returns:
            `true` if the spatial scale is greater than one, `false` otherwise.
        """
        return self.plan.spatial_scale > 1

    @property
    def scale(self) -> int:
        """Provides the configured spatial scaling factor.

        Returns:
            int: The spatial scaling factor.
        """
        return self.plan.spatial_scale

    @property
    def target_height(self) -> int:
        """
        Return the target output height for the spatial plan.

        Returns:
            int: Target output height in pixels.
        """
        return self.plan.target_height

    @property
    def target_width(self) -> int:
        """Return the target image width in pixels.

        Returns:
            int: The target image width.
        """
        return self.plan.target_width

    @property
    def stage1_height(self) -> int:
        """
        Provide the stage-one latent height used for reduced-resolution processing.

        Returns:
            int: The stage-one latent height.
        """
        return self.plan.stage1_height

    @property
    def stage1_width(self) -> int:
        """Get the stage-one latent width.

        Returns:
            int: The stage-one latent width.
        """
        return self.plan.stage1_width


def plan_fast_spatial(
    *,
    height: int,
    width: int,
    num_frames: int,
    spatial_scale: int = 2,
    vae_spatial_compression: int = 8,
    vae_temporal_compression: int = 4,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    upsample_mode: str = DEFAULT_PIXEL_UPSAMPLE_MODE,
    sharpen: float = DEFAULT_FAST_SPATIAL_SHARPEN,
    enabled: bool = True,
) -> FastSpatialPlan:
    """
    Build a plan for reduced-resolution denoising followed by pixel-space upsampling.

    Parameters:
        upsample_mode (str): Pixel interpolation kernel, one of
            :data:`~fastvideo.mlx_runtime.frame_upsample.PIXEL_UPSAMPLE_MODES`.
        sharpen (float): Unsharp strength applied after the resize.

    Returns:
        FastSpatialPlan: The validated spatial-fast processing plan.

    Raises:
        ValueError: If the upsample mode is unsupported or ``sharpen`` is negative.
    """
    if upsample_mode not in PIXEL_UPSAMPLE_MODES:
        raise ValueError(f"Unsupported upsample mode: {upsample_mode!r} "
                         f"(expected one of {', '.join(PIXEL_UPSAMPLE_MODES)})")
    if sharpen < 0.0:
        raise ValueError(f"sharpen must be >= 0, got {sharpen}")
    plan = plan_refine_resolutions(
        height=height,
        width=width,
        num_frames=num_frames,
        spatial_scale=spatial_scale,
        vae_spatial_compression=vae_spatial_compression,
        vae_temporal_compression=vae_temporal_compression,
        patch_size=patch_size,
        enabled=enabled,
        mode_label="fast-spatial",
    )
    if plan.spatial_scale > 1:
        logger.info(
            "[MLX fast-spatial] denoise+decode %dx%d → upsample %dx to %dx%d (%s, sharpen=%.2f)",
            plan.stage1_width,
            plan.stage1_height,
            plan.spatial_scale,
            plan.target_width,
            plan.target_height,
            upsample_mode,
            sharpen,
        )
    return FastSpatialPlan(plan=plan, upsample_mode=upsample_mode, sharpen=sharpen)


def apply_fast_spatial_upsample(
    frames: Iterable[np.ndarray],
    spatial: FastSpatialPlan,
) -> list[np.ndarray]:
    """Resample decoded stage-1 frames up to the target resolution.

    This runs on decoded RGB frames, *not* on latents: see the module
    docstring for why the latent-space version produced a blurred veil.

    Parameters:
        frames (Iterable[np.ndarray]): Decoded HxWx3 uint8 RGB frames, produced
            by decoding at the stage-one resolution.
        spatial (FastSpatialPlan): Plan defining the target size, interpolation
            kernel, and unsharp strength.

    Returns:
        list[np.ndarray]: Frames at the target resolution. When spatial scaling
            is disabled the frames are returned unchanged, as a list.
    """
    if not spatial.enabled:
        return list(frames)
    return upsample_frames(
        frames,
        width=spatial.target_width,
        height=spatial.target_height,
        mode=spatial.upsample_mode,
        sharpen=spatial.sharpen,
    )


def resolve_spatial_mode(
    *,
    refine: bool,
    fast_spatial: bool,
) -> str:
    """Select the active spatial processing mode, with refinement taking precedence.

    Returns:
        str: ``"refine"`` when refinement is enabled, ``"fast_spatial"`` when
            spatial-fast processing is enabled, or ``"off"`` otherwise.
    """
    if refine:
        return "refine"
    if fast_spatial:
        return "fast_spatial"
    return "off"


__all__ = [
    "DEFAULT_FAST_SPATIAL_SHARPEN",
    "FastSpatialPlan",
    "apply_fast_spatial_upsample",
    "plan_fast_spatial",
    "resolve_spatial_mode",
]
