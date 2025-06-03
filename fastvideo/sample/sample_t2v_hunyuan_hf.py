import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from diffusers import BitsAndBytesConfig
from diffusers.utils import export_to_video

from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info


def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def inference(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    # Peiyuan: GPU seed will cause A100 and H100 to produce different results .....
    weight_dtype = torch.bfloat16

    if args.transformer_path is not None:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.transformer_path)
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model_path,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=weight_dtype)

    pipe = HunyuanVideoPipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=weight_dtype)

    pipe.enable_vae_tiling()

    if args.lora_checkpoint_dir is not None:
        print(f"Loading LoRA weights from {args.lora_checkpoint_dir}")
        config_path = os.path.join(args.lora_checkpoint_dir, "lora_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        rank = lora_config_dict["lora_params"]["lora_rank"]
        lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        lora_scaling = lora_alpha / rank
        pipe.load_lora_weights(args.lora_checkpoint_dir, adapter_name="default")
        pipe.set_adapters(["default"], [lora_scaling])
        print(f"Successfully Loaded LoRA weights from {args.lora_checkpoint_dir}")
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = (torch.load(args.prompt_embed_path, map_location="cpu",
                                    weights_only=True).to(device).unsqueeze(0))
        encoder_attention_mask = (torch.load(args.encoder_attention_mask_path, map_location="cpu",
                                             weights_only=True).to(device).unsqueeze(0))
        prompts = None
    elif args.prompt_path is not None:
        prompts = [line.strip() for line in open(args.prompt_path, "r")]
        prompt_embeds = None
        encoder_attention_mask = None
    else:
        prompts = args.prompts
        prompt_embeds = None
        encoder_attention_mask = None

    if prompts is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                generator = torch.Generator("cpu").manual_seed(args.seed)
                video = pipe(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).frames
                if nccl_info.global_rank == 0:
                    os.makedirs(args.output_path, exist_ok=True)
                    suffix = prompt.split(".")[0]
                    export_to_video(
                        video[0],
                        os.path.join(args.output_path, f"{suffix}.mp4"),
                        fps=24,
                    )
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generator = torch.Generator("cpu").manual_seed(args.seed)
            videos = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=encoder_attention_mask,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames

        if nccl_info.global_rank == 0:
            export_to_video(videos[0], args.output_path + ".mp4", fps=24)


def inference_quantization(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_path

    if args.quantization == "nf4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_quant_type="nf4",
                                                 llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16,
                                                                     quantization_config=quantization_config)
    if args.quantization == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16,
                                                                     quantization_config=quantization_config)
    elif not args.quantization:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16).to(device)

    print("Max vram for read transformer:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
    torch.cuda.reset_max_memory_allocated(device)

    if not args.cpu_offload:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        pipe.transformer = transformer
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
    torch.cuda.reset_max_memory_allocated(device)
    pipe.scheduler._shift = args.flow_shift
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    print("Max vram for init pipeline:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
    if args.prompt.endswith('.txt'):
        with open(args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [args.prompt]

    generator = torch.Generator("cpu").manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.cuda.reset_max_memory_allocated(device)
    for prompt in prompts:
        start_time = time.perf_counter()
        output = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).frames[0]
        export_to_video(output, os.path.join(args.output_path, f"{prompt[:100]}.mp4"), fps=args.fps)
        print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
        print("Max vram for denoise:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift", type=int, default=7, help="Flow shift parameter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "fp8"])
    parser.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument("--prompt-template-video", type=str, default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    args = parser.parse_args()
    if args.quantization:
        inference_quantization(args)
    else:
        inference(args)
