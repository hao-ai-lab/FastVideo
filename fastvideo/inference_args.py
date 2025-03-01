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
import logging
import random
from typing import List, Optional, Dict, Union

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferenceArgs:
    # Model and tokenizer
    model_path: str
    load_format: str = "auto"
    dtype: str = "auto"
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None
    device: str = "cuda"

    # Other runtime options
    tp_size: int = 1
    sp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    
    # Memory management
    mem_fraction_static: Optional[float] = None
    
    # Inference parameters
    num_frames: int = 125
    num_height: int = 720
    num_width: int = 1280
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    embedded_cfg_scale: float = 6.0
    flow_shift: int = 17
    flow_reverse: bool = False
    shift: int = 7
    
    # Scheduler options
    scheduler_type: str = "euler"
    linear_threshold: float = 0.1
    linear_range: float = 0.75
    
    # Load balancing
    load_balance_method: str = "round_robin"
    
    # Watchdog
    watchdog_timeout: float = 60.0
    
    # LoRA options
    lora_paths: Optional[Union[List[str], Dict[str, str]]] = None
    max_loras_per_batch: int = 1
    
    # GPU options
    base_gpu_id: int = 0
    gpu_id_step: int = 1
    
    # Optimization flags
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    enable_dp_attention: bool = False
    enable_teacache: bool = False
    enable_torch_compile: bool = False
    vae_sp: bool = False
    cpu_offload: bool = False

    random_seed: Optional[int] = None
    dist_timeout: Optional[int] = None  # timeout for torch.distributed

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    show_time_cost: bool = False
    enable_metrics: bool = False
    decode_log_interval: int = 40

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Kernel backend
    attention_backend: Optional[str] = None
    sampling_backend: str = "pytorch"

    # Model-specific paths
    dit_weight: Optional[str] = None
    
    # Inference parameters
    prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"
    seed: int = 1024
    
    # Video generation parameters
    height: int = 720
    width: int = 1280
    
    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    rel_l1_thresh: float = 0.15
    
    # StepVideo specific parameters
    model_dir: Optional[str] = None
    vae_url: Optional[str] = None
    caption_url: Optional[str] = None
    time_shift: float = 13.0
    cfg_scale: float = 9.0
    infer_steps: int = 50

    def __post_init__(self):
        # Set missing default values
        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        # Set mem fraction static, which depends on the tensor parallelism size
        if self.mem_fraction_static is None:
            if self.tp_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.tp_size >= 8:
                self.mem_fraction_static = 0.81
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        # Choose kernel backends
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

        # if self.attention_backend is None:
        #     self.attention_backend = (
        #         "flashinfer" if is_flashinfer_available() else "triton"
        #     )

        # if self.attention_backend == "torch_native":
        #     logger.warning(
        #         "Cuda graph is disabled because of using torch native attention backend"
        #     )
        #     self.disable_cuda_graph = True

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=InferenceArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--quantization-param-path",
            type=str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=InferenceArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
                "gguf",
                "modelopt",
                "w8a8_int8",
                "nf4",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "xpu", "hpu", "cpu"],
            help="The device type.",
        )

        # Other runtime options
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=InferenceArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=InferenceArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=InferenceArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=InferenceArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=InferenceArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=InferenceArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log the inputs and outputs of all requests.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log prometheus metrics.",
        )
        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=InferenceArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=InferenceArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=InferenceArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=InferenceArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
            ],
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatbility. This will be removed in the future.
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=InferenceArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=InferenceArgs.node_rank, help="The node rank."
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=["flashinfer", "triton", "torch_native"],
            default=InferenceArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        
        # Video generation parameters
        parser.add_argument(
            "--num-frames",
            type=int,
            default=InferenceArgs.num_frames,
            help="Number of frames to generate",
        )
        parser.add_argument(
            "--height",
            "--num-height",
            type=int,
            default=InferenceArgs.num_height,
            help="Height of generated video",
        )
        parser.add_argument(
            "--width",
            "--num-width",
            type=int,
            default=InferenceArgs.num_width,
            help="Width of generated video",
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
        
        # Optimization flags
        parser.add_argument(
            "--vae-sp",
            action="store_true",
            help="Enable VAE spatial parallelism",
        )
        parser.add_argument(
            "--cpu-offload",
            action="store_true",
            help="Enable CPU offloading for memory efficiency",
        )
        parser.add_argument(
            "--enable-teacache",
            action="store_true",
            help="Enable TeaCache optimization",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Enable torch.compile for optimization",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable CUDA graph optimization",
        )
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable radix cache optimization",
        )
        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enable data parallel attention",
        )
        
        # LoRA options
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="+",
            default=None,
            help="Paths to LoRA adapters",
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=InferenceArgs.max_loras_per_batch,
            help="Maximum number of LoRAs per batch",
        )
        
        # GPU options
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=InferenceArgs.base_gpu_id,
            help="Base GPU ID for multi-GPU setup",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=InferenceArgs.gpu_id_step,
            help="Step size for GPU ID assignment",
        )

        # Model-specific paths
        parser.add_argument(
            "--dit-weight",
            type=str,
            help="Path to the DiT model weights",
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
        
        # StepVideo specific parameters
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )
        parser.add_argument(
            "--vae-url",
            type=str,
            help="URL for VAE server (StepVideo)",
        )
        parser.add_argument(
            "--caption-url",
            type=str,
            help="URL for caption server (StepVideo)",
        )
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
        parser.add_argument(
            "--infer-steps",
            type=int,
            default=InferenceArgs.infer_steps,
            help="Number of inference steps for StepVideo",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.dp_size = args.data_parallel_size
        args.ep_size = args.expert_parallel_size
        args.num_height = args.height
        args.num_width = args.width
        args.flow_shift = getattr(args, "shift", args.flow_shift)
        
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def check_inference_args(self):
        """Validate inference arguments for consistency"""
        assert (
            self.tp_size % self.nnodes == 0
        ), "tp_size must be divisible by number of nodes"
        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"
        assert (
            self.max_loras_per_batch > 0
            # FIXME
            and (self.lora_paths is None or self.disable_cuda_graph)
            and (self.lora_paths is None or self.disable_radix_cache)
        ), "compatibility of lora and cuda graph and radix attention is in progress"
        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"


def prepare_inference_args(argv: List[str]) -> InferenceArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = argparse.ArgumentParser()
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
