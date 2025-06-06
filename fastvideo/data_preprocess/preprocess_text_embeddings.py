import argparse
import json
import os

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.utils.load import load_text_encoder, load_vae

logger = get_logger(__name__)


class T5dataset(Dataset):

    def __init__(
        self,
        json_path,
        vae_debug,
    ):
        self.json_path = json_path
        self.vae_debug = vae_debug
        with open(self.json_path, "r") as f:
            train_dataset = json.load(f)
            self.train_dataset = sorted(train_dataset, key=lambda x: x["latent_path"])

    def __getitem__(self, idx):
        caption = self.train_dataset[idx]["caption"]
        filename = self.train_dataset[idx]["latent_path"].split(".")[0]
        length = self.train_dataset[idx]["length"]
        if self.vae_debug:
            latents = torch.load(
                os.path.join(args.output_dir, "latent", self.train_dataset[idx]["latent_path"]),
                map_location="cpu",
            )
        else:
            latents = []

        return dict(caption=caption, latents=latents, filename=filename, length=length)

    def __len__(self):
        return len(self.train_dataset)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    videoprocessor = VideoProcessor(vae_scale_factor=8)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)

    latents_json_path = os.path.join(args.output_dir, "videos2caption_temp.json")
    train_dataset = T5dataset(latents_json_path, args.vae_debug)
    text_encoder = load_text_encoder(args.model_type, args.model_path, device=device)
    vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    if args.model_type != "wan":
        vae.enable_tiling()
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_type):
                prompt_embeds, prompt_attention_mask = text_encoder.encode_prompt(prompt=data["caption"], )
                if args.vae_debug:
                    latents = data["latents"]
                    video = vae.decode(latents.to(device), return_dict=False)[0]
                    video = videoprocessor.postprocess_video(video)
                for idx, video_name in enumerate(data["filename"]):
                    prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                    video_path = os.path.join(args.output_dir, "video", video_name + ".mp4")
                    prompt_attention_mask_path = os.path.join(args.output_dir, "prompt_attention_mask",
                                                              video_name + ".pt")
                    # save latent
                    torch.save(prompt_embeds[idx], prompt_embed_path)
                    torch.save(prompt_attention_mask[idx], prompt_attention_mask_path)
                    print(f"sample {video_name} saved")
                    if args.vae_debug:
                        export_to_video(video[idx], video_path, fps=fps)
                    item = {}
                    item["length"] = int(data["length"][idx])
                    item["latent_path"] = video_name + ".pt"
                    item["prompt_embed_path"] = video_name + ".pt"
                    item["prompt_attention_mask"] = video_name + ".pt"
                    item["caption"] = data["caption"][idx]
                    json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    args = parser.parse_args()
    main(args)
