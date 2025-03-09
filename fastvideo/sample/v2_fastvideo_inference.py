import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

# Fix the import path
from fastvideo.inference_engine import InferenceEngine
from fastvideo.inference_args import InferenceArgs
from fastvideo.utils.utils import FlexibleArgumentParser
from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
from fastvideo.logger import init_logger
import torch.distributed as dist
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
logger = init_logger(__name__)

def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

def main(inference_args: InferenceArgs):
    # initialize_distributed()
    # print(nccl_info.sp_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    # logger.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    print(inference_args.sp_size)

    # Initialize tensor model parallel groups
    initialize_model_parallel(
        sequence_model_parallel_size=inference_args.sp_size
    )
    initialize_sequence_parallel_state(world_size)
    # initialize_distributed()


    print('Creating engine')
    # Create inference object using the updated API
    engine = InferenceEngine.create_engine(
        inference_args,
    )
    print('Engine created')
    # return

    # Load prompts
    if inference_args.prompt.endswith('.txt'):
        with open(inference_args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [inference_args.prompt]

    # Process each prompt
    for prompt in prompts:
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


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    InferenceArgs.add_cli_args(parser)
    args = parser.parse_args()
    inference_args = InferenceArgs.from_cli_args(args)
    inference_args.check_inference_args()

    # Validate arguments
    if inference_args.vae_sp and not inference_args.vae_tiling:
        raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")
        
    main(inference_args)