# SPDX-License-Identifier: Apache-2.0
"""Presets for Cosmos Predict video generation pipeline."""

from dataclasses import dataclass, field
from fastvideo.configs.pipelines.base import PipelinePreset


@dataclass
class CosmosPredictPreset(PipelinePreset):
    """Presets for Cosmos Predict (Text-to-Video/Video-to-Video)."""
    
    num_inference_steps: int = 36
    guidance_scale: float = 7.0
    
    # Cosmos Predict expects 1000 for num_train_timesteps and generates num_frames directly
    num_frames: int = 93
    height: int = 704
    width: int = 1280
    fps: int = 24

    # Scheduler config for UniPCMultistepScheduler
    scheduler_type: str = "unipc"
    scheduler_config: dict = field(
        default_factory=lambda: {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "predict_x0": True,
        }
    )


@dataclass
class CosmosPredict14BPreset(CosmosPredictPreset):
    """Presets for Cosmos Predict 14B."""
    pass
