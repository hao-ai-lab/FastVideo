import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import random
import numpy as np
import argparse
import os
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Generate video using Hunyuan model')
    
    parser.add_argument('--prompt', type=str, default="", help='Text prompt for video generation')
    parser.add_argument('--model_path', type=str, default="/mbz/users/hao.zhang/data/hunyuan_diffusers", help='Path to the Hunyuan model directory')
    parser.add_argument('--output_dir', type=str, default='outputs_video/hunyuan_hf', help='Directory to save the output video')
    parser.add_argument('--height', type=int, default=480, help='Height of the output video')
    parser.add_argument('--width', type=int, default=848, help='Width of the output video')
    parser.add_argument('--num_frames', type=int, default=93, help='Number of frames to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for generation')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second for the output video')


    return parser.parse_args()

def main():
    args = parse_args()
    prompt_candidates = ["Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
        "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature.",
       "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. Mid-shot, warm and cheerful tones.",
        "A superintelligent humanoid robot waking up. The robot has a sleek metallic body with futuristic design features. Its glowing red eyes are the focal point, emanating a sharp, intense light as it powers on. The scene is set in a dimly lit, high-tech laboratory filled with glowing control panels, robotic arms, and holographic screens. The setting emphasizes advanced technology and an atmosphere of mystery. The ambiance is eerie and dramatic, highlighting the moment of awakening and the robots immense intelligence. Photorealistic style with a cinematic, dark sci-fi aesthetic. Aspect ratio: 16:9 --v 6.1",
        "fox in the forest close-up quickly turned its head to the left.",
        "Man walking his dog in the woods on a hot sunny day",
        "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic."]
    # Set random seed
    #args.prompt = "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
    generator = torch.Generator("cpu").manual_seed(args.seed)
    # Load transformer model
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    # Initialize pipeline
    pipe = HunyuanVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        transformer=transformer,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    #pipe.vae = pipe.vae.to(torch.bfloat16)
    pipe.vae.enable_tiling()
    
    # Move to GPU
    device = torch.cuda.current_device()
    pipe.to(device)
    #pipe.enable_model_cpu_offload(device)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    file_name = args.prompt[:20]
    output_path = os.path.join(args.output_dir, file_name + 'output.mp4')

    # Generate video
    output = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=generator
    ).frames[0]

    # Save video
    export_to_video(output, output_path, fps=args.fps)
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    main()
