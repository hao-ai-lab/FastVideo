from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.models.stepvideo.modules.model import StepVideoModel
import os
import argparse
from fastvideo.utils.logging_ import main_print
from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)
import json


def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    main_print(f"world_size: {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=local_rank)
    initialize_sequence_parallel_state(world_size)


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="StepVideo inference script")

    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)

    args = parser.parse_args(namespace=namespace)

    return args


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    group.add_argument(
        "--vae_url",
        type=str,
        default='127.0.0.1',
        help="vae url.",
    )
    group.add_argument(
        "--caption_url",
        type=str,
        default='127.0.0.1',
        help="caption url.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    # Flow Matching
    group.add_argument(
        "--time_shift",
        type=float,
        default=13,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow_reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow_solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model_dir",
        type=str,
        default="./ckpts",
        help=
        "Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--model_resolution",
        type=str,
        default="540p",
        choices=["540p"],
        help=
        "Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--num_frames",
        type=int,
        default=204,
        help="How many frames to sample from a video. ",
    )
    group.add_argument(
        "--height",
        type=int,
        default=768,
        help="The height of video sample",
    )
    group.add_argument(
        "--width",
        type=int,
        default=768,
        help="The width of video sample",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument("--seed",
                       type=int,
                       default=1234,
                       help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument(
        "--pos_magic",
        type=str,
        default=
        "超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。",
        help="Positive magic prompt for sampling.")
    group.add_argument(
        "--neg_magic",
        type=str,
        default="画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。",
        help="Negative magic prompt for sampling.")
    group.add_argument("--cfg_scale",
                       type=float,
                       default=9.0,
                       help="Classifier free guidance scale.")
    group.add_argument("--mask_search_files_path",
                       type=str,
                       default="assets/mask_strategy.json")
    group.add_argument("--mask_strategy_file_path",
                       type=str,
                       default="assets/mask_strategy_stepvideo.json")
    group.add_argument("--skip_time_steps", type=int, default=10)
    group.add_argument(
        "--mask_strategy_selected",
        type=lambda x: [int(i) for i in x.strip('[]').split(',')
                        ],  # Convert string to list of integers
        default=[1, 2, 6],  # Now can be directly set as a list
        help="order of candidates")

    return parser


if __name__ == "__main__":
    args = parse_args()
    initialize_distributed()
    main_print(f"sequence parallel size: {nccl_info.sp_size}")
    device = torch.cuda.current_device()

    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")
    transformer = StepVideoModel.from_pretrained(os.path.join(
        args.model_dir, "transformer"),
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device)
    scheduler = FlowMatchDiscreteScheduler()
    pipeline = StepVideoPipeline(transformer,
                                 scheduler,
                                 save_path=args.save_path)
    pipeline.setup_api(
        vae_url=args.vae_url,
        caption_url=args.caption_url,
    )

    with open(args.mask_strategy_file_path, 'r') as f:
        mask_strategy = json.load(f)

    with open(args.prompt) as f:
        prompts = [line.strip() for line in f.readlines()]
    existing_cnt = 0
    for prompt in prompts:
        main_print(f"Generating video for prompt: {prompt}")
        videos = pipeline(prompt=prompt,
                          num_frames=args.num_frames,
                          height=args.height,
                          width=args.width,
                          num_inference_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          time_shift=args.time_shift,
                          pos_magic=args.pos_magic,
                          neg_magic=args.neg_magic,
                          output_file_name=prompt[:150],
                          mask_strategy=mask_strategy)

    dist.destroy_process_group()
