# Copyright 2023-2024 SGLang Team
# Adapted from SGLang server_args.py
# Copyright 2024-2025 FastVideo Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The arguments of FastVideo Inference."""

import argparse
import dataclasses
from fastvideo.logger import init_logger
from fastvideo.utils.utils import FlexibleArgumentParser
from typing import List, Optional




@dataclasses.dataclass
class InferenceArgs:
    # Model and path configuration
    model_path: str
    model: str = "HYVideo-T/2-cfgdistill"
    dit_weight: Optional[str] = None
    model_dir: Optional[str] = None
    
    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: Optional[str] = None
    
    # Parallelism
    tp_size: int = 1
    sp_size: int = 1
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    
    # Video generation parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 117
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    embedded_cfg_scale: float = 6.0
    flow_shift: int = 7
    flow_reverse: bool = False

    output_type: str = "pil"
    
    # Model configuration
    latent_channels: int = 16
    precision: str = "bf16"
    rope_theta: int = 256
    
    # VAE configuration
    vae: str = "884-16c-hy"
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    vae_url: Optional[str] = None
    vae_sp: bool = False
    
    # Text encoder configuration
    text_encoder: str = "llm"
    text_encoder_precision: str = "fp16"
    text_states_dim: int = 4096
    text_len: int = 256
    tokenizer: str = "llm"
    prompt_template: str = "dit-llm-encode"
    prompt_template_video: str = "dit-llm-encode-video"
    hidden_state_skip_layer: int = 2
    apply_final_norm: bool = False
    
    # Secondary text encoder
    text_encoder_2: str = "clipL"
    text_encoder_precision_2: str = "fp16"
    text_states_dim_2: int = 768
    tokenizer_2: str = "clipL"
    text_len_2: int = 77
    caption_url: Optional[str] = None
    
    # Flow Matching parameters
    flow_solver: str = "euler"
    use_linear_quadratic_schedule: bool = False
    linear_schedule_end: int = 25
    denoise_type: str = "flow"
    
    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    rel_l1_thresh: float = 0.15
    enable_torch_compile: bool = False
    
    # Scheduler options
    scheduler_type: str = "euler"
    linear_threshold: float = 0.1
    linear_range: float = 0.75
    
    # HunYuan specific parameters
    neg_prompt: Optional[str] = None
    batch_size: int = 1
    num_videos: int = 1
    fps: int = 24
    load_key: str = "module"
    use_cpu_offload: bool = False
    reproduce: bool = False
    disable_autocast: bool = False
    
    # StepVideo specific parameters
    time_shift: float = 13.0
    cfg_scale: float = 9.0
    
    # Optimization flags
    enable_teacache: bool = False
    
    # Logging
    log_level: str = "info"
    
    # Kernel backend
    attention_backend: Optional[str] = None
    
    # Inference parameters
    prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"
    seed: int = 1024
    seeds: Optional[List[int]] = None
    device_str: Optional[str] = None
    device = None

    def __post_init__(self):
        pass

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--model",
            type=str,
            default=InferenceArgs.model,
            help="Model type to use",
        )
        parser.add_argument(
            "--dit-weight",
            type=str,
            help="Path to the DiT model weights",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )
        
        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=InferenceArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=InferenceArgs.revision,
            help="The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=InferenceArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sequence-parallel-size",
            "--sp-size",
            type=int,
            default=InferenceArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=InferenceArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Video generation parameters
        parser.add_argument(
            "--height",
            type=int,
            default=InferenceArgs.height,
            help="Height of generated video",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=InferenceArgs.width,
            help="Width of generated video",
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            default=InferenceArgs.num_frames,
            help="Number of frames to generate",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=InferenceArgs.num_inference_steps,
            help="Number of inference steps",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=InferenceArgs.guidance_scale,
            help="Guidance scale for classifier-free guidance",
        )
        parser.add_argument(
            "--guidance-rescale",
            type=float,
            default=InferenceArgs.guidance_rescale,
            help="Guidance rescale for classifier-free guidance",
        )
        parser.add_argument(
            "--embedded-cfg-scale",
            type=float,
            default=InferenceArgs.embedded_cfg_scale,
            help="Embedded CFG scale",
        )
        parser.add_argument(
            "--flow-shift",
            "--shift",
            type=int,
            default=InferenceArgs.flow_shift,
            help="Flow shift parameter",
        )
        parser.add_argument(
            "--flow-reverse",
            action="store_true",
            help="Reverse flow direction",
        )
        parser.add_argument(
            "--output-type",
            type=str,
            default=InferenceArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )
        
        # Model configuration
        parser.add_argument(
            "--latent-channels",
            type=int,
            default=InferenceArgs.latent_channels,
            help="Number of latent channels",
        )
        parser.add_argument(
            "--precision",
            type=str,
            default=InferenceArgs.precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for the model",
        )
        parser.add_argument(
            "--rope-theta",
            type=int,
            default=InferenceArgs.rope_theta,
            help="Theta used in RoPE",
        )
        
        # VAE configuration
        parser.add_argument(
            "--vae",
            type=str,
            default=InferenceArgs.vae,
            help="VAE model to use",
        )
        parser.add_argument(
            "--vae-precision",
            type=str,
            default=InferenceArgs.vae_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for VAE",
        )
        parser.add_argument(
            "--vae-tiling",
            action="store_true",
            default=InferenceArgs.vae_tiling,
            help="Enable VAE tiling",
        )
        parser.add_argument(
            "--vae-url",
            type=str,
            help="URL for VAE server (StepVideo)",
        )
        parser.add_argument(
            "--vae-sp",
            action="store_true",
            help="Enable VAE spatial parallelism",
        )
        
        # Text encoder configuration
        parser.add_argument(
            "--text-encoder",
            type=str,
            default=InferenceArgs.text_encoder,
            help="Text encoder to use",
        )
        parser.add_argument(
            "--text-encoder-precision",
            type=str,
            default=InferenceArgs.text_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for text encoder",
        )
        parser.add_argument(
            "--text-states-dim",
            type=int,
            default=InferenceArgs.text_states_dim,
            help="Dimension of text states",
        )
        parser.add_argument(
            "--text-len",
            type=int,
            default=InferenceArgs.text_len,
            help="Maximum text length",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=InferenceArgs.tokenizer,
            help="Tokenizer to use",
        )
        parser.add_argument(
            "--prompt-template",
            type=str,
            default=InferenceArgs.prompt_template,
            help="Template for prompt processing",
        )
        parser.add_argument(
            "--prompt-template-video",
            type=str,
            default=InferenceArgs.prompt_template_video,
            help="Template for video prompt processing",
        )
        parser.add_argument(
            "--hidden-state-skip-layer",
            type=int,
            default=InferenceArgs.hidden_state_skip_layer,
            help="Number of layers to skip for hidden states",
        )
        parser.add_argument(
            "--apply-final-norm",
            action="store_true",
            help="Apply final normalization",
        )
        
        # Secondary text encoder
        parser.add_argument(
            "--text-encoder-2",
            type=str,
            default=InferenceArgs.text_encoder_2,
            help="Secondary text encoder to use",
        )
        parser.add_argument(
            "--text-encoder-precision-2",
            type=str,
            default=InferenceArgs.text_encoder_precision_2,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for secondary text encoder",
        )
        parser.add_argument(
            "--text-states-dim-2",
            type=int,
            default=InferenceArgs.text_states_dim_2,
            help="Dimension of secondary text states",
        )
        parser.add_argument(
            "--tokenizer-2",
            type=str,
            default=InferenceArgs.tokenizer_2,
            help="Secondary tokenizer to use",
        )
        parser.add_argument(
            "--text-len-2",
            type=int,
            default=InferenceArgs.text_len_2,
            help="Maximum secondary text length",
        )
        parser.add_argument(
            "--caption-url",
            type=str,
            help="URL for caption server (StepVideo)",
        )
        
        # Flow Matching parameters
        parser.add_argument(
            "--flow-solver",
            type=str,
            default=InferenceArgs.flow_solver,
            help="Solver for flow matching",
        )
        parser.add_argument(
            "--use-linear-quadratic-schedule",
            action="store_true",
            help="Use linear quadratic schedule for flow matching",
        )
        parser.add_argument(
            "--linear-schedule-end",
            type=int,
            default=InferenceArgs.linear_schedule_end,
            help="End step for linear quadratic schedule for flow matching",
        )
        parser.add_argument(
            "--denoise-type",
            type=str,
            default=InferenceArgs.denoise_type,
            help="Denoise type for noised inputs",
        )
        
        # STA (Spatial-Temporal Attention) parameters
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--rel-l1-thresh",
            type=float,
            default=InferenceArgs.rel_l1_thresh,
            help="Relative L1 threshold for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Use torch.compile for speeding up STA inference without teacache",
        )
        
        # Scheduler options
        parser.add_argument(
            "--scheduler-type",
            type=str,
            default=InferenceArgs.scheduler_type,
            help="Type of scheduler to use",
        )
        parser.add_argument(
            "--linear-threshold",
            type=float,
            default=InferenceArgs.linear_threshold,
            help="Linear threshold for PCM scheduler",
        )
        parser.add_argument(
            "--linear-range",
            type=float,
            default=InferenceArgs.linear_range,
            help="Linear range for PCM scheduler",
        )
        
        # HunYuan specific parameters
        parser.add_argument(
            "--neg-prompt",
            type=str,
            default=InferenceArgs.neg_prompt,
            help="Negative prompt for sampling",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=InferenceArgs.batch_size,
            help="Batch size for inference",
        )
        parser.add_argument(
            "--num-videos",
            type=int,
            default=InferenceArgs.num_videos,
            help="Number of videos to generate per prompt",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=InferenceArgs.fps,
            help="Frames per second for output video",
        )
        parser.add_argument(
            "--load-key",
            type=str,
            default=InferenceArgs.load_key,
            help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model",
        )
        parser.add_argument(
            "--use-cpu-offload",
            action="store_true",
            help="Use CPU offload for the model load",
        )
        parser.add_argument(
            "--reproduce",
            action="store_true",
            help="Enable reproducibility by setting random seeds and deterministic algorithms",
        )
        parser.add_argument(
            "--disable-autocast",
            action="store_true",
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )
        
        # StepVideo specific parameters
        parser.add_argument(
            "--time-shift",
            type=float,
            default=InferenceArgs.time_shift,
            help="Time shift parameter for StepVideo",
        )
        parser.add_argument(
            "--cfg-scale",
            type=float,
            default=InferenceArgs.cfg_scale,
            help="CFG scale for StepVideo",
        )
        
        # Optimization flags
        parser.add_argument(
            "--enable-teacache",
            action="store_true",
            help="Enable TeaCache optimization",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=InferenceArgs.log_level,
            help="The logging level of all loggers.",
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=["flashinfer", "triton", "torch_native"],
            default=InferenceArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        
        # Inference parameters
        parser.add_argument(
            "--prompt",
            type=str,
            help="Text prompt for video generation",
        )
        parser.add_argument(
            "--prompt-path",
            "--prompt_path",
            type=str,
            help="Path to a text file containing the prompt",
        )
        parser.add_argument(
            "--output-path",
            "--output_path",
            type=str,
            default=InferenceArgs.output_path,
            help="Directory to save generated videos",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=InferenceArgs.seed,
            help="Random seed for reproducibility",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.sp_size = args.sequence_parallel_size
        args.flow_shift = getattr(args, "shift", args.flow_shift)
        
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        
        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs = {}
        for attr in attrs:
            # Convert snake_case attribute name to kebab-case CLI argument name
            cli_attr = attr.replace('_', '-')
            
            # Handle renamed attributes or those with multiple CLI names
            if attr == 'tp_size' and hasattr(args, 'tensor_parallel_size'):
                kwargs[attr] = args.tensor_parallel_size
            elif attr == 'sp_size' and hasattr(args, 'sequence_parallel_size'):
                kwargs[attr] = args.sequence_parallel_size
            elif attr == 'flow_shift' and hasattr(args, 'shift'):
                kwargs[attr] = args.shift
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                default_value = getattr(cls, attr, None)
                kwargs[attr] = getattr(args, attr, default_value)
        
        return cls(**kwargs)


    def check_inference_args(self):
        """Validate inference arguments for consistency"""
        
        # Validate VAE spatial parallelism with VAE tiling
        if self.vae_sp and not self.vae_tiling:
            raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")
        

def prepare_inference_args(argv: List[str]) -> InferenceArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    InferenceArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    inference_args = InferenceArgs.from_cli_args(raw_args)
    inference_args.check_inference_args()
    return inference_args


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise ValueError(self.help)