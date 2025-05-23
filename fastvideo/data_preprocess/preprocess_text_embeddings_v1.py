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

# from fastvideo.utils.load import load_text_encoder, load_vae

from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import VAELoader, TextEncoderLoader, TokenizerLoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel, get_world_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.configs.models.encoders.t5 import T5Config

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
    rank = int(os.getenv("RANK", 0))
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size, sequence_model_parallel_size=world_size)
    print("world_size", world_size, "local rank", local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    world_group = get_world_group()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(local_rank)
    # if not dist.is_initialized():
    #     dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    videoprocessor = VideoProcessor(vae_scale_factor=8)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)

    vae_precision = "fp16" 
    text_encoder_precision = "fp32"
    fastvideo_args = FastVideoArgs(model_path=args.model_path,
                         use_cpu_offload=False,
                         vae_precision=vae_precision,
                         text_encoder_precisions=(text_encoder_precision,))
    fastvideo_args.device = device
    fastvideo_args.device_str = f"cuda:{local_rank}"

    # fastvideo_args.dit_config = HunyuanVideoConfig()
    fastvideo_args.vae_config = WanVAEConfig()
    fastvideo_args.text_encoder_configs = (T5Config(),)

    # vae_loader = VAELoader()
    # vae = vae_loader.load_vae()
    text_encoder_loader = TextEncoderLoader()
    tokenizer_loader = TokenizerLoader()

    model_path = args.model_path
    path = maybe_download_model(model_path)
    # PIPELINE_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    ENCODER_PATH = os.path.join(path, "text_encoder")
    TOKENIZER_PATH = os.path.join(path, "tokenizer")
    print(ENCODER_PATH)
    text_encoder = text_encoder_loader.load(ENCODER_PATH, "text_encoder", fastvideo_args)
    tokenizer = tokenizer_loader.load(TOKENIZER_PATH, "tokenizer", fastvideo_args)

    latents_json_path = os.path.join(args.output_dir, "videos2caption_temp.json")
    train_dataset = T5dataset(latents_json_path, args.vae_debug)
    # text_encoder = load_text_encoder(args.model_type, args.model_path, device=device)
    # vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    # vae.enable_tiling()
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
            # with torch.autocast("cuda", dtype=torch.float32):
            print(data["caption"])
            text_inputs = tokenizer(data["caption"], **fastvideo_args.text_encoder_configs[0].tokenizer_kwargs).to(
                fastvideo_args.device)
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs["attention_mask"]
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            from fastvideo.v1.configs.pipelines.wan import t5_postprocess_text
            post_process_func = t5_postprocess_text
            prompt_embeds = post_process_func(outputs)
            prompt_attention_mask = attention_mask
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
                    export_to_video(video[idx], video_path, fps=16)
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
    # parser.add_argument("--model_type", type=str, default="mochi")
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
