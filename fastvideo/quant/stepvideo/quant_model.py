import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
from fastvideo.models.stepvideo.modules.model import StepVideoModel
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.models.stepvideo.config import parse_args
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_fp8_supported

from quant_utils import check_module_device_consistency, quant_layer_refactor_, apply_func_to_submodules
        
def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
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
    parser = add_quant_args(parser)

    args = parser.parse_args(namespace=namespace)

    return args

def add_quant_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Choose which nn.Linear layers do you want to quantize"
    )
    
    group.add_argument('--quant_layers', 
        nargs='+', 
        choices=['ff', 'attn1', 'attn2'], 
        required=True, 
        help="Select layers from ['ff', 'attn1', 'attn2']"
    )

    return parser

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

    return parser

if __name__ == "__main__":
    args = parse_args()
    initialize_distributed()
    device = torch.cuda.current_device()

    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")

    scheduler = FlowMatchDiscreteScheduler()
    transformer = StepVideoModel.from_pretrained(os.path.join(args.model_dir, "transformer"), torch_dtype=torch.bfloat16, device_map=device)

    ## FP8 W8A8 Quantization
    ## stationary per-tensor quant for weight + dynamic per-token quant for activation
    main_print("Quantizing model")
    assert set(args.quant_layers) <= set(['ff', 'attn1', 'attn2'])
    quant_config = Fp8Config(activation_scheme="dynamic")
    apply_func_to_submodules(transformer, class_type=nn.Linear, function=quant_layer_refactor_, quant_layers=args.quant_layers, name=None, parent_module=None, quant_config=quant_config, full_name=None)

    assert check_module_device_consistency(transformer)
    assert cutlass_fp8_supported()
    
    pipeline = StepVideoPipeline(transformer, scheduler)
    pipeline.setup_api(
        vae_url=args.vae_url,
        caption_url=args.caption_url,
    )

    if args.prompt.endswith('.txt'):
        with open(args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [args.prompt]
    for prompt in prompts:
        videos = pipeline(prompt=prompt,
                          num_frames=args.num_frames,
                          height=args.height,
                          width=args.width,
                          num_inference_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          time_shift=args.time_shift,
                          pos_magic=args.pos_magic,
                          neg_magic=args.neg_magic,
                          output_file_name=prompt[:50])

    dist.destroy_process_group()
