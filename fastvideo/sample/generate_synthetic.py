import argparse
import json
import os

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video

from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline


def generate_video_and_latent(pipe, prompt, height, width, num_frames, num_inference_steps, guidance_scale):
    # Set the random seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(12345)
    # Generate videos from the input prompt
    noise, video, latent, prompt_embed, prompt_attention_mask = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="latent_and_video",
    )
    # prompt_embed has negative prompt at index 0
    return noise[0], video[0], latent[0], prompt_embed[1], prompt_attention_mask[1]

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

    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    if not isinstance(args.prompt_path, list):
        args.prompt_path = [args.prompt_path]
    if len(args.prompt_path) == 1 and args.prompt_path[0].endswith("txt"):
        text_prompt = open(args.prompt_path[0], "r").readlines()
        text_prompt = [i.strip() for i in text_prompt]

    pipe = MochiPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload(gpu_id=local_rank)
    # make dir if not exist

    os.makedirs(args.dataset_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "noise"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_output_dir, "prompt_attention_mask"), exist_ok=True)
    data = []
    for i, prompt in enumerate(text_prompt):
        if i % world_size != local_rank:
            continue
        (
            noise,
            video,
            latent,
            prompt_embed,
            prompt_attention_mask,
        ) = generate_video_and_latent(
            pipe,
            prompt,
            args.height,
            args.width,
            args.num_frames,
            args.num_inference_steps,
            args.guidance_scale,
        )
        # save latent
        video_name = str(i)
        noise_path = os.path.join(args.dataset_output_dir, "noise", video_name + ".pt")
        latent_path = os.path.join(args.dataset_output_dir, "latent", video_name + ".pt")
        prompt_embed_path = os.path.join(args.dataset_output_dir, "prompt_embed", video_name + ".pt")
        video_path = os.path.join(args.dataset_output_dir, "video", video_name + ".mp4")
        prompt_attention_mask_path = os.path.join(args.dataset_output_dir, "prompt_attention_mask", video_name + ".pt")
        # save latent
        torch.save(noise, noise_path)
        torch.save(latent, latent_path)
        torch.save(prompt_embed, prompt_embed_path)
        torch.save(prompt_attention_mask, prompt_attention_mask_path)
        export_to_video(video, video_path, fps=30)
        item = {}

        item["cap"] = prompt
        item["video"] = video_name + ".mp4"
        item["noise"] = video_name + ".pt"
        item["latent_path"] = video_name + ".pt"
        item["prompt_embed_path"] = video_name + ".pt"
        item["prompt_attention_mask"] = video_name + ".pt"
        data.append(item)
    dist.barrier()
    local_data = data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)

    # save json
    if local_rank == 0:
        all_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.dataset_output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_data, f, indent=4)
