from fastvideo.dataset import getdataset
from torch.utils.data import DataLoader
from fastvideo.utils.dataset_utils import Collate
import argparse
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import json
import os
from diffusers import AutoencoderKLMochi
import torch.distributed as dist

logger = get_logger(__name__)

def main(args):
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size, 'local rank', local_rank)
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = 4, 8, 8
    args.ae_stride = args.ae_stride_h
    patch_size_t, patch_size_h, patch_size_w = 1, 2, 2
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    accelerator = Accelerator(
        project_config=accelerator_project_config,
    )
    train_dataset = getdataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    encoder_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    vae = AutoencoderKLMochi.from_pretrained("/ephemeral/hao.zhang/outputfolder/ckptfolder/mochi_diffuser", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latents"), exist_ok=True)
    
    json_data = []
    for idx, data in enumerate(train_dataloader):
        if idx % world_size != local_rank:
            continue
        print(f"currently generating sample {idx} on gpu {encoder_device}")
        # try:
        video_name = str(idx)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                print(f"shape for pixel value {data['pixel_values'][0].shape}")
                latents = vae.encode(data['pixel_values'][0].to(encoder_device))['latent_dist'].sample()

                print(type(latents))
                print(latents.shape)
                latents_path = os.path.join(args.output_dir, "latents", video_name + ".pt")
                print(f"video {idx} processed in vae encoder")
                # save latent
                torch.save(latents, latents_path)
                item = {}
                item["latents"] = video_name + ".pt"
                item["caption"] = data['text']
                json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption_temp.json"), 'w') as f:
            json.dump(all_json_data, f, indent=4)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--mochi_dir", type=str, required=True)
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--target_length", type=int, default=65)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--group_frame", action="store_true") # TODO
    parser.add_argument("--group_resolution", action="store_true") # TODO
    parser.add_argument("--dataset", default='t2v')
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default='google/t5-v1_1-xxl')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )

    args = parser.parse_args()
    main(args)
