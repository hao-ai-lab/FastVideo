from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from fastvideo.models.stepvideo.config import parse_args
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.models.stepvideo.parallel import initialize_parall_group, get_parallel_group
from fastvideo.models.stepvideo.modules.model import StepVideoModel
import os
from fastvideo.utils.logging_ import main_print
from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler

if __name__ == "__main__":
    args = parse_args()
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    local_rank = get_parallel_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    
    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")
    transformer = StepVideoModel.from_pretrained(os.path.join(args.model_dir, "transformer"))
    scheduler = FlowMatchDiscreteScheduler()
    pipeline = StepVideoPipeline(transformer, scheduler).to(device, torch.bfloat16)
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