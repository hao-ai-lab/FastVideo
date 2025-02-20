import argparse
import os

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from fastvideo.utils.load import load_text_encoder

logger = get_logger(__name__)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    text_encoder = load_text_encoder(args.model_type, args.model_path, device=device)
    autocast_type = torch.float16 if args.model_type == "hunyuan" else torch.bfloat16
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
            with torch.autocast("cuda", dtype=autocast_type):
                prompt_embeds, prompt_attention_mask = text_encoder.encode_prompt(prompt)
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
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--validation_prompt_txt", type=str)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    main(args)
