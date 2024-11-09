import torch
from fastvideo.model.pipeline_mochi import MochiPipeline
import torch.distributed as dist

from diffusers.utils import export_to_video
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
import argparse
import os
from fastvideo.model.modeling_mochi import MochiTransformer3DModel

def initialize_distributed():
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    


def main(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    generator = torch.Generator(device).manual_seed(args.seed)
    weight_dtype = torch.bfloat16
    
    if args.transformer_path is not None:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path, torch_dtype=torch.bfloat16)
        pipe = MochiPipeline.from_pretrained(args.model_path, transformer = transformer, torch_dtype=torch.bfloat16)
    else:
        pipe = MochiPipeline.from_pretrained(args.model_path,  torch_dtype=torch.bfloat16)
    pipe.enable_vae_tiling()
    pipe.to(device)
    #pipe.enable_model_cpu_offload()
    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = torch.load(args.prompt_embed_path, map_location="cpu", weights_only=True).to(device).unsqueeze(0)
        encoder_attention_mask = torch.load(args.encoder_attention_mask_path, map_location="cpu", weights_only=True).to(device).unsqueeze(0)
        prompts = None
    else:
        prompts = args.prompts
        prompt_embeds = None
        encoder_attention_mask = None
    videos = pipe(
        prompt=prompts,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=encoder_attention_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames

    if nccl_info.global_rank <= 0:
        if prompts is not None:
            for video, prompt in zip(videos, prompts):
                suffix = prompt.split(".")[0]
                export_to_video(video, args.output_path + f"_{suffix}.mp4", fps=30)
        else:
            export_to_video(videos[0], args.output_path + ".mp4", fps=30)

if __name__ == "__main__":
    # arg parse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs='+', default=[])
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
    parser.add_argument("--encoder_attention_mask_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
