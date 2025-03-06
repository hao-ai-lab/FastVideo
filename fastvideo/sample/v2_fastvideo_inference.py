#!/usr/bin/env python3
# Copyright 2024 FastVideo Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FastVideo inference script that replaces shell script launcher."""

import os
import sys
from typing import List

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from fastvideo.inference_args import prepare_inference_args, InferenceArgs
from fastvideo.logger import init_logger

from fastvideo.inference_engine import InferenceEngine
from fastvideo.logger import init_logger
# from fastvideo.distributed.parallel_state import initialize_model_parallel
# from fastvideo.distributed.distributed import init_distributed_environment
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info

logger = init_logger(__name__)


def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

def load_prompts(inference_args: InferenceArgs) -> List[str]:
    if inference_args.prompt.endswith('.txt'):
        with open(inference_args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [inference_args.prompt]
    return prompts


def run_inference(inference_args: InferenceArgs):
    """
    Main function for running inference.
    
    Args:
        inference_args: The inference arguments.
    """
    # Initialize distributed environment
    initialize_distributed()

    # Create output directory
    os.makedirs(inference_args.output_path, exist_ok=True)
    logger.info(f"Output directory created: {inference_args.output_path}")
    
    prompts = load_prompts(inference_args.prompt)
    
    logger.info("Creating inference engine...")
    engine = InferenceEngine.create_engine(inference_args)
    logger.info("Inference engine created successfully")
    
    # Run inference for each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
        
        # Run inference
        # TODO: need to think what's the right API for this
        outputs = engine.run(
            prompt=prompt,
            inference_args=inference_args,
        )

        # Process outputs
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))
            
        # Save video
        os.makedirs(os.path.dirname(inference_args.output_path), exist_ok=True)
        imageio.mimsave(
            os.path.join(inference_args.output_path, f"{prompt[:100]}.mp4"), 
            frames, 
            fps=inference_args.fps
        )
        





@record
def inference_subprocess(inference_args: InferenceArgs) -> None:
    """The subprocess function for inference."""
    run_inference(inference_args)

def main():
    """Main entry point for inference."""
    # Parse arguments using the existing argument parser
    inference_args = prepare_inference_args(sys.argv[1:])
    
    # Configure distributed launch
    num_gpus = inference_args.sp_size * inference_args.tp_size
    
    # Set up launch configuration
    launch_config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_gpus,
        run_id="fastvideo_inference",
        role="default",
        rdzv_endpoint="127.0.0.1:29503",  # Using the same port as in the shell script
        rdzv_backend="static",
        rdzv_configs={"rank": 0, "timeout": 900},
        rdzv_timeout=-1,
        max_restarts=0,
        monitor_interval=5,
        start_method="spawn",
        metrics_cfg={},
        local_addr=None,
    )
    
    # Pass all arguments to the subprocess
    launcher = elastic_launch(launch_config, inference_subprocess)
    launcher(sys.argv)
    
    logger.info(f"Inference completed. Output saved to {inference_args.output_path}")

if __name__ == "__main__":
    main()
