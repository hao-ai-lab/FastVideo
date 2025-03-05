import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

# Fix the import path
from fastvideo.inference import DiffusionInference
from fastvideo.inference_args import InferenceArgs
from fastvideo.utils.utils import FlexibleArgumentParser



def main(inference_args: InferenceArgs):
    # initialize_distributed()
    # print(nccl_info.sp_size)


    print('Loading pipeline')
    # Create inference object using the updated API
    diffusion = DiffusionInference.load_pipeline(
        # model_path=models_root_path,
        inference_args,
        # model_loader_cls=ModelLoader,
        # pipeline_cls=HunyuanVideoPipeline
    )
    print('Pipeline loaded')
    # return

    # Load prompts
    if inference_args.prompt.endswith('.txt'):
        with open(inference_args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [inference_args.prompt]

    # Process each prompt
    for prompt in prompts:
        outputs = diffusion.predict(
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
    
    # Validate arguments
    if inference_args.vae_sp and not inference_args.vae_tiling:
        raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")
        
    main(inference_args)
