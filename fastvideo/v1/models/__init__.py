# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Dict
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


def get_scheduler(module_path: str, architecture: str,
                  inference_args: InferenceArgs) -> Dict:
    """Create a scheduler based on the inference args. Can be overridden by subclasses."""
    if hasattr(inference_args,
               'denoise_type') and inference_args.denoise_type == "flow":
        # TODO(will): add schedulers to register or create a new scheduler registry
        # TODO(will): default to config file but allow override through
        # inference args. Currently only uses inference args.
        from fastvideo.v1.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchDiscreteScheduler
        return FlowMatchDiscreteScheduler(
            shift=inference_args.flow_shift,
            solver=inference_args.flow_solver,
        )
    else:
        raise ValueError(f"Invalid denoise type: {inference_args.denoise_type}")


__all__ = [
    "set_random_seed",
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "get_model",
    "get_scheduler",
]
