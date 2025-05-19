import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets import Dataset
import pyarrow.dataset as ds

from fastvideo.dataset import getdataset

from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import maybe_download_model, shallow_asdict
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.models.loader.component_loader import VAELoader, TextEncoderLoader, TokenizerLoader
from fastvideo.v1.forward_context import set_forward_context
from fastvideo import PipelineConfig

logger = init_logger(__name__)

BASE_MODEL_PATH = "/workspace/data/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
VAE_PATH = os.path.join(MODEL_PATH, "vae")
TEXT_ENCODER_PATH = os.path.join(MODEL_PATH, "text_encoder")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer")

def main(args):
    local_rank = int(os.getenv("RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size, sequence_model_parallel_size=world_size)
    torch.cuda.set_device(local_rank)
    train_dataset = getdataset(args)
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    
    pipeline_config = PipelineConfig.from_pretrained(MODEL_PATH)
    kwargs = {
        "use_cpu_offload": False,
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=True, load_decoder=False),
    }
    pipeline_config_args = shallow_asdict(pipeline_config)
    pipeline_config_args.update(kwargs)
    fastvideo_args = FastVideoArgs(model_path=MODEL_PATH,
                                   num_gpus=world_size,
                                   device_str="cuda",
                                   **pipeline_config_args,
                                   )
    fastvideo_args.check_fastvideo_args()
    fastvideo_args.device = torch.device(f"cuda:{local_rank}")

    vae_loader = VAELoader()
    text_encoder_loader = TextEncoderLoader()
    tokenizer_loader = TokenizerLoader()
    vae = vae_loader.load(VAE_PATH, "", fastvideo_args)
    text_encoder = text_encoder_loader.load(TEXT_ENCODER_PATH, "", fastvideo_args)
    tokenizer = tokenizer_loader.load(TOKENIZER_PATH, "", None)
    
    os.makedirs(args.output_dir, exist_ok=True)

    validation_dataset = {"encoder_hidden_states": [], "encoder_attention_mask": []}
    latent_dataset = {"latent": [], "encoder_hidden_states": [], "encoder_attention_mask": []}
    with open(args.validation_prompt_txt, "r", encoding="utf-8") as file:
        lines = file.readlines()
    prompts = [line.strip() for line in lines]
    for prompt in prompts:
        with torch.inference_mode():
            # Text Encoder
            prompt = fastvideo_args.preprocess_text_funcs[0](data["text"])
            text_inputs = tokenizer(prompt, **fastvideo_args.text_encoder_configs[0].tokenizer_kwargs).to(
            fastvideo_args.device)
            input_ids = text_inputs["input_ids"]
            prompt_attention_mask = text_inputs["attention_mask"]
            with set_forward_context(current_timestep=0, attn_metadata=None):
                prompt_embeds = text_encoder(
                    input_ids=input_ids,
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True,
                )
            prompt_embeds = fastvideo_args.postprocess_text_funcs[0](prompt_embeds)
            validation_dataset["encoder_hidden_states"].append(prompt_embeds)
            validation_dataset["encoder_attention_mask"].append(prompt_attention_mask)
            
    for i, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            # VAE
            with torch.autocast("cuda", dtype=torch.float32):
                latents = vae.encode(data["pixel_values"].to(fastvideo_args.device)).sample()
            # Text Encoder
            prompt = fastvideo_args.preprocess_text_funcs[0](data["text"])
            text_inputs = tokenizer(prompt, **fastvideo_args.text_encoder_configs[0].tokenizer_kwargs).to(
            fastvideo_args.device)
            input_ids = text_inputs["input_ids"]
            prompt_attention_mask = text_inputs["attention_mask"]
            with set_forward_context(current_timestep=0, attn_metadata=None):
                prompt_embeds = text_encoder(
                    input_ids=input_ids,
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True,
                )
            prompt_embeds = fastvideo_args.postprocess_text_funcs[0](prompt_embeds)
            assert latents.shape[0] == prompt_embeds.shape[0]
            assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]
            for idx in range(latents.shape[0]):
                latent_dataset["latent"].append(latents[idx].unsqueeze(0))
                latent_dataset["encoder_hidden_states"].append(prompt_embeds[idx].unsqueeze(0))
                latent_dataset["encoder_attention_mask"].append(prompt_attention_mask[idx].unsqueeze(0))
    validation_dataset = Dataset.from_dict(validation_dataset).with_format("torch")
    latent_dataset = Dataset.from_dict(latent_dataset).with_format("torch")
    ds.write_dataset(
        data=latent_dataset.data.table,
        base_dir=os.path.join(args.output_dir, "train"),
        format="parquet",
    )
    ds.write_dataset(
        data=validation_dataset.data.table,
        base_dir=os.path.join(args.output_dir, "valid"),
        format="parquet",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--validation_prompt_txt", type=str)
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
