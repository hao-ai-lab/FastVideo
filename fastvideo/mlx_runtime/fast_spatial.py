# SPDX-License-Identifier: Apache-2.0
"""Spatial fast mode for the MLX Wan runtime (RIFE's spatial twin).

RIFE ``--fast`` cuts *frames* (temporal). This module cuts *pixels*
(spatial): denoise at ``target // scale``, then upsample clean latents
back to the target grid before decode. No second denoise pass — that is
``--refine`` (quality). The two compose:

* ``--fast-spatial`` alone → speed (≈ scale² fewer tokens)
* ``--refine`` alone → quality two-pass (H3 / LTX-2)
* ``--fast`` + ``--refine`` → fewer frames at base res, full-res refine
* ``--fast`` + ``--fast-spatial`` → fewer frames *and* fewer pixels

MetalFX is intentionally not used: it needs game-engine motion vectors
and depth that diffusion output lacks. Latent bilinear upsample is the
same primitive as the refine hand-off, without the re-noise / re-denoise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.refine import RefinePlan, plan_refine_resolutions, upsample_latents_spatial

logger = init_logger(__name__)


@dataclass(frozen=True)
class FastSpatialPlan:
    """Resolved geometry for a spatial-fast (upsample-only) run."""

    plan: RefinePlan
    upsample_mode: str

    @property
    def enabled(self) -> bool:
        return self.plan.spatial_scale > 1

    @property
    def scale(self) -> int:
        return self.plan.spatial_scale

    @property
    def target_height(self) -> int:
        return self.plan.target_height

    @property
    def target_width(self) -> int:
        return self.plan.target_width

    @property
    def stage1_height(self) -> int:
        return self.plan.stage1_height

    @property
    def stage1_width(self) -> int:
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
    upsample_mode: str = "bilinear",
    enabled: bool = True,
) -> FastSpatialPlan:
    """Build a spatial-fast plan (shared validation with refine)."""
    if upsample_mode not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported upsample mode: {upsample_mode}")
    plan = plan_refine_resolutions(
        height=height,
        width=width,
        num_frames=num_frames,
        spatial_scale=spatial_scale,
        vae_spatial_compression=vae_spatial_compression,
        vae_temporal_compression=vae_temporal_compression,
        patch_size=patch_size,
        enabled=enabled,
    )
    if plan.spatial_scale > 1:
        logger.info(
            "[MLX fast-spatial] denoise %dx%d → upsample %dx to %dx%d (%s)",
            plan.stage1_width,
            plan.stage1_height,
            plan.spatial_scale,
            plan.target_width,
            plan.target_height,
            upsample_mode,
        )
    return FastSpatialPlan(plan=plan, upsample_mode=upsample_mode)


def apply_fast_spatial_upsample(
    clean_latents: Any,
    spatial: FastSpatialPlan,
) -> Any:
    """Upsample stage-1 clean latents to the target grid (no re-noise)."""
    if not spatial.enabled:
        return clean_latents
    up = upsample_latents_spatial(
        clean_latents,
        scale=spatial.scale,
        mode=spatial.upsample_mode,
    )
    expected_h = spatial.plan.stage2_latent_height
    expected_w = spatial.plan.stage2_latent_width
    got_h, got_w = int(up.shape[-2]), int(up.shape[-1])
    if got_h != expected_h or got_w != expected_w:
        raise ValueError(
            f"fast-spatial upsample produced {got_h}x{got_w} latents, expected "
            f"{expected_h}x{expected_w} for target "
            f"{spatial.target_height}x{spatial.target_width}."
        )
    return up


def resolve_spatial_mode(
    *,
    refine: bool,
    fast_spatial: bool,
) -> str:
    """Return the active spatial mode: ``off`` | ``fast_spatial`` | ``refine``.

    Refine is the quality path and wins when both flags are set — fast-spatial
    is a pure speed upsample with no second denoise, so stacking it under
    refine would be a no-op (refine already densifies at full res).
    """
    if refine:
        return "refine"
    if fast_spatial:
        return "fast_spatial"
    return "off"


__all__ = [
    "FastSpatialPlan",
    "apply_fast_spatial_upsample",
    "plan_fast_spatial",
    "resolve_spatial_mode",
]
