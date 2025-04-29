# type: ignore

# SPDX-License-Identifier: Apache-2.0

import os
import sys

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.v1.distributed import (init_distributed_environment,
                                      initialize_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs, prepare_fastvideo_args
# Fix the import path
from fastvideo.v1.inference_engine import InferenceEngine


def initialize_distributed_and_parallelism(fastvideo_args: FastVideoArgs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    fastvideo_args.device_str = device_str
    fastvideo_args.device = torch.device(device_str)
    assert fastvideo_args.sp_size is not None
    assert fastvideo_args.tp_size is not None
    initialize_model_parallel(
        sequence_model_parallel_size=fastvideo_args.sp_size,
        tensor_model_parallel_size=fastvideo_args.tp_size,
    )


def main(fastvideo_args: FastVideoArgs):
    initialize_distributed_and_parallelism(fastvideo_args)
    engine = InferenceEngine.create_engine(fastvideo_args, )

    if fastvideo_args.prompt_path is not None:
        with open(fastvideo_args.prompt_path) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        if fastvideo_args.prompt is None:
            raise ValueError("prompt or prompt_path is required")
        prompts = [fastvideo_args.prompt]

    # Process each prompt
    for prompt in prompts:
        outputs = engine.run(
            prompt=prompt,
            fastvideo_args=fastvideo_args,
        )

        # Process outputs
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save video
        os.makedirs(os.path.dirname(fastvideo_args.output_path), exist_ok=True)
        imageio.mimsave(os.path.join(fastvideo_args.output_path,
                                     f"{prompt[:100]}.mp4"),
                        frames,
                        fps=fastvideo_args.fps)


if __name__ == "__main__":
    fastvideo_args = prepare_fastvideo_args(sys.argv[1:])
    main(fastvideo_args)
