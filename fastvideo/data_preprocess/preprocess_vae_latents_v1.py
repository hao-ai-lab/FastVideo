import argparse
import json
import os

import torch
# import torch.distributed as dist
# from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.dataset import getdataset
# from fastvideo.utils.load import load_vae
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import VAELoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel, get_world_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.configs.models.vaes import WanVAEConfig

logger = init_logger(__name__)

model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
path = maybe_download_model(model_path)
# PIPELINE_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VAE_PATH = os.path.join(path, "vae")
print(VAE_PATH)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size, sequence_model_parallel_size=world_size)
    print("world_size", world_size, "local rank", local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    world_group = get_world_group()

    vae_precision = "fp16" 
    fastvideo_args = FastVideoArgs(model_path=VAE_PATH,
                         use_cpu_offload=False,
                         vae_precision=vae_precision)
    fastvideo_args.device = device
    # fastvideo_args.dit_config = HunyuanVideoConfig()
    fastvideo_args.vae_config = WanVAEConfig()

    train_dataset = getdataset(args)
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    # encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(local_rank)
    # if not dist.is_initialized():
    #     dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    vae_loader = VAELoader()
    vae = vae_loader.load(VAE_PATH, "vae", fastvideo_args)
    # vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    # vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                latents = vae.encode(data["pixel_values"].to(device)).sample()
            for idx, video_path in enumerate(data["path"]):
                video_name = os.path.basename(video_path).split(".")[0]
                latent_path = os.path.join(args.output_dir, "latent", video_name + ".pt")
                torch.save(latents[idx].to(torch.bfloat16), latent_path)
                item = {}
                item["length"] = latents[idx].shape[1]
                item["latent_path"] = video_name + ".pt"
                item["caption"] = data["text"][idx]
                json_data.append(item)
                print(f"{video_name} processed")
    world_group.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    for i in range(world_size):
        if local_rank == i:
            world_group.broadcast_object(local_data, src=i)
        else:
            gathered_data[i] = world_group.broadcast_object(None, src=i)
    gathered_data[local_rank] = json_data
    print(gathered_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption_temp.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    # parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range", type=int, default=2.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--dataset", default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    args = parser.parse_args()
    main(args)
