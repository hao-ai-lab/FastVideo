# SPDX-License-Identifier: Apache-2.0
"""Experimental Apple MLX runtime helpers.

This package is intentionally small for now. It exists to grow the Apple-native
FastWan path in measurable steps: shape planning, primitive benchmarks, then
Wan block parity, then full DiT/runtime support.
"""

from fastvideo.mlx_runtime.fastwan import (
    FastWanShape,
    MLXQuantizationSpec,
    MLXWanDiT,
    MLXWanTransformerBlock,
    UnsupportedMLXQuantizationError,
    ensure_quantization_supported,
    fastwan_shape,
    fastwan_shape_from_config,
    mlx_dit_from_diffusers_safetensors,
    mlx_block_weights_from_torch,
    mlx_block_weights_from_diffusers_safetensors,
    quantization_support_error,
)
from fastvideo.mlx_runtime.checkpoint import (
    load_mlx_dit_checkpoint,
    save_mlx_dit_checkpoint,
)
from fastvideo.mlx_runtime.memory import (
    AppliedMemoryLimits,
    add_memory_limit_args,
    apply_memory_limits,
    gib_to_bytes,
)
from fastvideo.mlx_runtime.refine import (
    DEFAULT_REFINE_SIGMA,
    RefinePlan,
    TwoPassResult,
    default_refine_timesteps,
    plan_refine_resolutions,
    prepare_refine_latents,
    refine_sigma_from_schedule,
    run_dmd_loop,
    run_two_pass_dmd,
    upsample_latents_spatial,
)
from fastvideo.mlx_runtime.frame_upsample import (
    DEFAULT_PIXEL_UPSAMPLE_MODE,
    PIXEL_UPSAMPLE_MODES,
    unsharp,
    upsample_frame,
    upsample_frames,
)
from fastvideo.mlx_runtime.fast_spatial import (
    DEFAULT_FAST_SPATIAL_SHARPEN,
    FastSpatialPlan,
    apply_fast_spatial_upsample,
    plan_fast_spatial,
    resolve_spatial_mode,
)
from fastvideo.mlx_runtime.prompt_enhance import (
    DEFAULT_ENHANCE_SYSTEM_PROMPT,
    DEFAULT_MLX_LM_MODEL,
    EnhanceResult,
    enhance_prompt,
    enhance_prompt_template,
    enhance_result_as_metrics,
    load_or_enhance_prompt,
)

__all__ = [
    "AppliedMemoryLimits",
    "DEFAULT_ENHANCE_SYSTEM_PROMPT",
    "DEFAULT_MLX_LM_MODEL",
    "DEFAULT_FAST_SPATIAL_SHARPEN",
    "DEFAULT_PIXEL_UPSAMPLE_MODE",
    "DEFAULT_REFINE_SIGMA",
    "EnhanceResult",
    "FastSpatialPlan",
    "FastWanShape",
    "MLXQuantizationSpec",
    "MLXWanDiT",
    "MLXWanTransformerBlock",
    "RefinePlan",
    "TwoPassResult",
    "UnsupportedMLXQuantizationError",
    "add_memory_limit_args",
    "apply_fast_spatial_upsample",
    "apply_memory_limits",
    "enhance_prompt",
    "enhance_prompt_template",
    "enhance_result_as_metrics",
    "ensure_quantization_supported",
    "fastwan_shape",
    "fastwan_shape_from_config",
    "gib_to_bytes",
    "load_mlx_dit_checkpoint",
    "load_or_enhance_prompt",
    "mlx_dit_from_diffusers_safetensors",
    "mlx_block_weights_from_diffusers_safetensors",
    "mlx_block_weights_from_torch",
    "PIXEL_UPSAMPLE_MODES",
    "default_refine_timesteps",
    "plan_fast_spatial",
    "plan_refine_resolutions",
    "prepare_refine_latents",
    "quantization_support_error",
    "refine_sigma_from_schedule",
    "resolve_spatial_mode",
    "run_dmd_loop",
    "run_two_pass_dmd",
    "save_mlx_dit_checkpoint",
    "unsharp",
    "upsample_frame",
    "upsample_frames",
    "upsample_latents_spatial",
]
