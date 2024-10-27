import torch
from diffusers import MochiPipeline

from diffusers.utils import export_to_video, load_image, load_video

def main():
    # Set the random seed for reproducibility
    generator = torch.Generator("cpu").manual_seed(12345)

    # Load the Mochi pipeline
    model_path = "data/mochi"
    prompt = "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."
    num_frames = 163
    height = 480
    width = 848
    num_inference_steps = 64

    pipe = MochiPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    # Generate videos from the input prompt
    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.5
    ).frames[0]


    export_to_video(video, './outputs.mp4', fps=30)

if __name__ == "__main__":
    main()
