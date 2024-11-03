import torch
from fastvideo.sample.pipeline_mochi import MochiPipeline
import torch.distributed as dist

from diffusers.utils import export_to_video, load_image, load_video
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
import argparse
import os
import diffusers
from fastvideo.sample.monkey_patch.sp_attention import NewMochiAttnProcessor2_0
from fastvideo.sample.monkey_patch.sp_rope import NewMochiRoPE

def initialize_distributed():
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    

def hf_mochi_add_sp_monkey_patch():
    initialize_distributed()
    diffusers.models.attention_processor.MochiAttnProcessor2_0.__call__ = NewMochiAttnProcessor2_0.__call__
    diffusers.models.transformers.transformer_mochi.MochiRoPE._get_positions = NewMochiRoPE._get_positions

def main(args):
    hf_mochi_add_sp_monkey_patch()
    print(nccl_info.world_size)
    device = torch.cuda.current_device()
    generator = torch.Generator(device).manual_seed(args.seed)
    weight_dtype = torch.bfloat16
    
    pipe = MochiPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.enable_vae_tiling()
    pipe.to(device)
    #pipe.enable_model_cpu_offload()

    # Generate videos from the input prompt
    video = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]

    dist.barrier()
    if nccl_info.rank <= 0:
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./outputs.mp4")
    args = parser.parse_args()
    main(args)
