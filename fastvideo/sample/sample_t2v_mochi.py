import torch
from diffusers import MochiPipeline, MochiTransformer3DModel
from diffusers.utils import export_to_video, load_image, load_video
import argparse

def main(args):
    # Set the random seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(args.seed)

    if args.transformer_path is not None:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16)
        pipe = MochiPipeline.from_pretrained(args.model_path, transformer = transformer, torch_dtype=torch.bfloat16)
    else:
        pipe = MochiPipeline.from_pretrained(args.model_path,  torch_dtype=torch.bfloat16)
    pipe.enable_vae_tiling()
    # pipe.to("cuda:1")
    pipe.enable_model_cpu_offload()

    # Generate videos from the input prompt
    video = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    ).frames[0]


    export_to_video(video, args.output_path, fps=30)

if __name__ == "__main__":
    # arg parse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.")
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
