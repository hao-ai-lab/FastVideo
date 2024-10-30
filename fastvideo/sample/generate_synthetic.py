import json


import torch
from fastvideo.sample.mochi_pipe import MochiPipeline
import os
from diffusers.utils import export_to_video
import argparse

def generate_video_and_latent(pipe, prompt, height, width, num_frames, num_inference_steps, guidance_scale):
    # Set the random seed for reproducibility
    generator = torch.Generator("cpu").manual_seed(12345)
    # Generate videos from the input prompt
    video, latent, prompt_embed, prompt_attention_mask = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="latent_and_video"
    )

    return video[0], latent[0], prompt_embed[0], prompt_attention_mask[0]
    
    # return dummy tensor to debug first
    # return torch.zeros(1, 3, 480, 848), torch.zeros(1, 256, 16, 16)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--prompt_path", type=str, default="data/dummyVid/videos2caption.json")
    parser.add_argument("--dataset_output_dir", type=str, default="data/dummySynthetic")
    args = parser.parse_args()

    with open(args.prompt_path, 'r') as f:
        data = json.load(f)
    prompt_list = []
    for item in data:
        prompt_list.append(item["cap"])
        
        
    pipe = MochiPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload(gpu_id=0)
    # make dir if not exist 
    
    os.makedirs(args.dataset_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "prompt_attention_mask"), exist_ok=True)
    for i, prompt in enumerate(prompt_list):
        video, latent, prompt_embed, prompt_attention_mask = generate_video_and_latent(pipe, prompt, args.height, args.width, args.num_frames, args.num_inference_steps, args.guidance_scale)
        # save latent
        video_name = data[i]["path"].split("/")[-1].split(".")[0]
        latent_path = os.path.join(args.dataset_output_dir, "latent", video_name + ".pt")
        prompt_embed_path = os.path.join(args.dataset_output_dir, "prompt_embed", video_name + ".pt")
        video_path = os.path.join(args.dataset_output_dir, "video", video_name + ".mp4")
        prompt_attention_mask_path = os.path.join(args.dataset_output_dir, "prompt_attention_mask", video_name + ".pt")
        # save latent
        torch.save(latent, latent_path)
        torch.save(prompt_embed, prompt_embed_path)
        torch.save(prompt_attention_mask, prompt_attention_mask_path)
        export_to_video(video, video_path, fps=30)
        data[i]["latent_path"] = video_name + ".pt"
        data[i]["prompt_embed_path"] = video_name + ".pt"
        data[i]["prompt_attention_mask"] = video_name + ".pt"
    # save json
    with open(os.path.join(args.dataset_output_dir, "videos2caption.json"), 'w') as f:
        json.dump(data, f, indent=4)

    