import argparse

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline


def main(args):
    # Set the random seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(args.seed)
    # do not invert
    scheduler = FlowMatchEulerDiscreteScheduler()
    if args.transformer_path is not None:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path)
    else:
        transformer = MochiTransformer3DModel.from_pretrained(args.model_path, subfolder="transformer/")
    pipe = MochiPipeline.from_pretrained(args.model_path, transformer=transformer, scheduler=scheduler)
    pipe.enable_vae_tiling()
    # pipe.to("cuda:1")
    pipe.enable_model_cpu_offload()

    # Generate videos from the input prompt
    with torch.autocast("cuda", dtype=torch.bfloat16):
        videos = pipe(
            prompt=args.prompts,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames

    for prompt, video in zip(args.prompts, videos):
        export_to_video(video, args.output_path + f"_{prompt}.mp4", fps=30)


if __name__ == "__main__":
    # arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=[])
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs.mp4")
    args = parser.parse_args()
    main(args)
