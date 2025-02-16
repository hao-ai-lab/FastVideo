from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from fastvideo.models.stepvideo.config import parse_args
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.models.stepvideo.modules.model import StepVideoModel
import os
from fastvideo.utils.logging_ import main_print
from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=local_rank)
    initialize_sequence_parallel_state(world_size)
if __name__ == "__main__":
    args = parse_args()
    initialize_distributed()
    device = torch.cuda.current_device()
    
    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")
    transformer = StepVideoModel.from_pretrained(os.path.join(args.model_dir, "transformer"), torch_dtype=torch.bfloat16, device_map=device)
    scheduler = FlowMatchDiscreteScheduler()
    pipeline = StepVideoPipeline(transformer, scheduler)
    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )
    
    
    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
    
    dist.destroy_process_group()