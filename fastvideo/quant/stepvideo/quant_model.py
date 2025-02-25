from fastvideo.models.stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
import torch.nn as nn
from fastvideo.models.stepvideo.config import parse_args
from fastvideo.models.stepvideo.utils import setup_seed
from fastvideo.models.stepvideo.modules.model import StepVideoModel
import os
import gc
import argparse
from fastvideo.utils.logging_ import main_print
from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from quant_layer import FP8Linear
from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_fp8_supported

def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name):
    input_size = submodule.in_features
    output_size = submodule.out_features
    bias = True if submodule.bias is not None else False
    skip_bias_add = False if submodule.bias is not None else True
    params_dtype = submodule.weight.dtype
    device = submodule.device
    prefix = full_name

    quant_linear = FP8Linear(input_size, output_size, bias=bias, skip_bias_add=skip_bias_add, params_dtype=params_dtype, quant_config=quant_config, prefix=prefix)
    quant_linear.weight.copy_(submodule.weight)

    del submodule
    torch.cuda.empty_cache()

    quant_linear.quant_method.process_weights_after_loading(quant_linear)
    
    setattr(parent_module, name, quant_linear.to(device))

        
def apply_func_to_submodules(module, class_type, function, parent_name="", quant_layers=[], **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """

    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        parent_module = module

        # INFO: pass from the parent call into func
        if 'name' in kwargs:
            kwargs['name']=name
        if 'full_name' in kwargs:
            kwargs['full_name'] = full_name
        if 'parent_module' in kwargs:
            kwargs['parent_module'] = module
        if isinstance(submodule, class_type):
            for quant_layer in quant_layers:
                if quant_layer in full_name:
                    function(submodule, **kwargs)
                    break

        # Recursively apply the function to submodules
        apply_func_to_submodules(submodule, class_type, function, parent_name=full_name, **kwargs)
        
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

    return parser

def get_model_size(model: torch.nn.Module):
    """
    Computes the memory footprint of a PyTorch model in megabytes (MB).
    """
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (total_params + total_buffers) / (1024 ** 2)  # Convert bytes to MB
    return total_size

def check_module_device_consistency(module: torch.nn.Module):
    devices = {param.device for param in module.parameters()}
    return len(devices) == 1

if __name__ == "__main__":
    args = parse_args()
    initialize_distributed()
    device = torch.cuda.current_device()

    setup_seed(args.seed)
    main_print("Loading model, this might take a while...")

    scheduler = FlowMatchDiscreteScheduler()
    transformer = StepVideoModel.from_pretrained(os.path.join(args.model_dir, "transformer"), torch_dtype=torch.bfloat16, device_map=device)

    ## Replace nn.Linear layers with Fp8Linear layers
    quant_layers = ['ff']
    assert set(quant_layers) <= set(['ff', 'attn1', 'attn2'])
    quant_config = Fp8Config(activation_scheme="dynamic")
    apply_func_to_submodules(transformer, class_type=nn.Linear, function=quant_layer_refactor_, quant_layers=quant_layers, name=None, parent_module=None, quant_config=quant_config, full_name=None)
    print(transformer)
    print(torch.cuda.max_memory_allocated() / 1024 / 1024)
    print(transformer.transformer_blocks[0].ff.net[2].weight.data[0, :20])
    # print(transformer.dtype)
    print("All params on same device: ", check_module_device_consistency(transformer))
    print("Model Size After Quant: ", get_model_size(transformer))
    # print(transformer.transformer_blocks[0].ff.net[2].weight.dtype)
    # print(transformer.transformer_blocks[0].ff.net[2].weight_scale.dtype)

    print("FP8 supported: ", cutlass_fp8_supported())
    
    pipeline = StepVideoPipeline(transformer, scheduler)
    pipeline.setup_api(
        vae_url=args.vae_url,
        caption_url=args.caption_url,
    )

    prompt = args.prompt
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
