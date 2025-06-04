import argparse
import json
import os

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

from fastvideo.distill.solver import PCMFMScheduler
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info


def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def main(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    # Peiyuan: GPU seed will cause A100 and H100 to produce different results .....

    if args.scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        linear_quadratic = True if "linear_quadratic" in args.scheduler_type else False
        scheduler = PCMFMScheduler(
            1000,
            args.shift,
            args.num_euler_timesteps,
            linear_quadratic,
            args.linear_threshold,
            args.linear_range,
        )
    if args.transformer_path is not None:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path)
    else:
        transformer = MochiTransformer3DModel.from_pretrained(args.model_path, subfolder="transformer/")

    pipe = MochiPipeline.from_pretrained(args.model_path, transformer=transformer, scheduler=scheduler)

    pipe.enable_vae_tiling()

    if args.lora_checkpoint_dir is not None:
        print(f"Loading LoRA weights from {args.lora_checkpoint_dir}")
        config_path = os.path.join(args.lora_checkpoint_dir, "lora_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        rank = lora_config_dict["lora_params"]["lora_rank"]
        lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        lora_scaling = lora_alpha / rank
        pipe.load_lora_weights(args.lora_checkpoint_dir, adapter_name="default")
        pipe.set_adapters(["default"], [lora_scaling])
        print(f"Successfully Loaded LoRA weights from {args.lora_checkpoint_dir}")
    # pipe.to(device)

    pipe.enable_model_cpu_offload(device)

    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = (torch.load(args.prompt_embed_path, map_location="cpu",
                                    weights_only=True).to(device).unsqueeze(0))
        encoder_attention_mask = (torch.load(args.encoder_attention_mask_path, map_location="cpu",
                                             weights_only=True).to(device).unsqueeze(0))
        prompts = None
    elif args.prompt_path is not None:
        prompts = [line.strip() for line in open(args.prompt_path, "r")]
        prompt_embeds = None
        encoder_attention_mask = None
    else:
        prompts = args.prompts
        prompt_embeds = None
        encoder_attention_mask = None

    if prompts is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                generator = torch.Generator("cpu").manual_seed(args.seed)
                video = pipe(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).frames
                if nccl_info.global_rank == 0:
                    os.makedirs(args.output_path, exist_ok=True)
                    suffix = prompt.split(".")[0]
                    export_to_video(
                        video[0],
                        os.path.join(args.output_path, f"{suffix}.mp4"),
                        fps=30,
                    )
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generator = torch.Generator("cpu").manual_seed(args.seed)
            videos = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=encoder_attention_mask,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).frames

        if nccl_info.global_rank == 0:
            export_to_video(videos[0], args.output_path + ".mp4", fps=30)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./outputs.mp4")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--scheduler_type", type=str, default="euler")
    parser.add_argument("--encoder_attention_mask_path", type=str, default=None)
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument("--linear_threshold", type=float, default=0.025)
    parser.add_argument("--linear_range", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
