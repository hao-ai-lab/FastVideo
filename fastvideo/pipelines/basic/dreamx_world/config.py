# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for DreamX-World pipeline configs."""

from fastvideo.configs.pipelines.dreamx_world import (
    DreamXWorld5BARPipelineConfig,
    DreamXWorld5BCamPipelineConfig,
    make_dreamx_world_5b_ar_dit_config,
    make_dreamx_world_5b_cam_dit_config,
    make_dreamx_world_5b_cam_text_encoder_config,
    make_dreamx_world_5b_cam_vae_config,
)

__all__ = [
    "DreamXWorld5BARPipelineConfig",
    "DreamXWorld5BCamPipelineConfig",
    "make_dreamx_world_5b_ar_dit_config",
    "make_dreamx_world_5b_cam_dit_config",
    "make_dreamx_world_5b_cam_text_encoder_config",
    "make_dreamx_world_5b_cam_vae_config",
]
