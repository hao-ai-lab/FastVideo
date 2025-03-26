# SPDX-License-Identifier: Apache-2.0

import os

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
import sys
# Fix the import path
from fastvideo.v1.inference_engine import InferenceEngine
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.inference_args import prepare_inference_args
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel
from fastvideo.v1.logger import logger


def initialize_distributed_and_parallelism(inference_args: InferenceArgs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    inference_args.device_str = device_str
    inference_args.device = torch.device(device_str)
    initialize_model_parallel(
        sequence_model_parallel_size=inference_args.sp_size,
        tensor_model_parallel_size=inference_args.tp_size,
    )


def main(inference_args: InferenceArgs):
    initialize_distributed_and_parallelism(inference_args)
    engine = InferenceEngine.create_engine(inference_args, )

    if inference_args.prompt_path is not None:
        with open(inference_args.prompt_path) as f:
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
        imageio.mimsave(os.path.join(inference_args.output_path,
                                     f"{prompt[:100]}.mp4"),
                        frames,
                        fps=inference_args.fps)


if __name__ == "__main__":
    inference_args = prepare_inference_args(sys.argv[1:])
    main(inference_args)
