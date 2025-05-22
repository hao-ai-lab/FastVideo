import argparse
import os

import torch
# import torch.distributed as dist
from accelerate.logging import get_logger

# from fastvideo.utils.load import load_text_encoder

from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import VAELoader, TextEncoderLoader, TokenizerLoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel, get_world_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.configs.models.encoders.t5 import T5Config

logger = get_logger(__name__)


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

    # text_encoder = load_text_encoder(args.model_type, args.model_path, device=device)
    # autocast_type = torch.float16 if args.model_type == "hunyuan" else torch.bfloat16
    # output_dir/validation/prompt_attention_mask
    # output_dir/validation/prompt_embed
    os.makedirs(os.path.join(args.output_dir, "validation"), exist_ok=True)
    os.makedirs(
        os.path.join(args.output_dir, "validation", "prompt_attention_mask"),
        exist_ok=True,
    )
    os.makedirs(os.path.join(args.output_dir, "validation", "prompt_embed"), exist_ok=True)

    with open(args.validation_prompt_txt, "r", encoding="utf-8") as file:
        lines = file.readlines()
    prompts = [line.strip() for line in lines]
    for prompt in prompts:
        with torch.inference_mode():
            # with torch.autocast("cuda", dtype=autocast_type):
            text_inputs = tokenizer(prompt, **fastvideo_args.text_encoder_configs[0].tokenizer_kwargs).to(
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

            file_name = prompt.split(".")[0]
            prompt_embed_path = os.path.join(args.output_dir, "validation", "prompt_embed", f"{file_name}.pt")
            prompt_attention_mask_path = os.path.join(
                args.output_dir,
                "validation",
                "prompt_attention_mask",
                f"{file_name}.pt",
            )
            torch.save(prompt_embeds[0], prompt_embed_path)
            torch.save(prompt_attention_mask[0], prompt_attention_mask_path)
            print(f"sample {file_name} saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--validation_prompt_txt", type=str)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    main(args)
