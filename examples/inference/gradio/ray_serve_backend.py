import time
import os
import torch
import base64
import io
from copy import deepcopy
from typing import Dict, Any, Optional, List

import ray
from ray import serve
from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import imageio
from ray.serve.handle import DeploymentHandle

NUM_GPUS = 8
SUPPORTED_MODELS = [
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", 
    "FastVideo/FastWan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
]


class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    use_negative_prompt: bool = False
    seed: int = 42
    guidance_scale: float = 7.5
    num_frames: int = 21
    height: int = 448
    width: int = 832
    # num_inference_steps: int = 20
    randomize_seed: bool = False
    return_frames: bool = False  # Whether to return base64 encoded frames
    image_path: Optional[str] = None  # Path to input image for I2V
    model_type: str = "t2v"  # "t2v" or "i2v" to specify which model to use
    model_path: Optional[str] = None  # Specific model path to use


class VideoGenerationResponse(BaseModel):
    video_data: Optional[str] = None  # Base64 encoded video data
    seed: int
    success: bool
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    # Detailed timing information
    model_load_time: Optional[float] = None
    inference_time: Optional[float] = None
    encoding_time: Optional[float] = None
    total_time: Optional[float] = None
    stage_names: Optional[List[str]] = None
    stage_execution_times: Optional[List[float]] = None


def encode_video_to_base64(frames: List[np.ndarray], fps: int = 24) -> str:
    """Convert numpy frames to base64-encoded MP4 video"""
    if not frames:
        return ""
    
    try:
        # Save frames to bytes buffer as MP4
        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, fps=fps, format="mp4")
        buffer.seek(0)
        
        # Encode to base64
        video_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:video/mp4;base64,{video_base64}"
        
    except Exception as e:
        print(f"Warning: Failed to encode video: {e}")
        return ""





# ---------------------------------------------------------------------------
# Model-Specific Deployments (one per model type)
# These deployments each load a single model and expose a `generate_video`   
# method that can be invoked via a DeploymentHandle. Each deployment can be
# scaled independently by configuring `num_replicas` when binding.
# ---------------------------------------------------------------------------


@serve.deployment(  # T2V 1.3B model deployment
    ray_actor_options={"num_cpus": 10, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
)
class T2VModelDeployment:
    """Serve deployment wrapping the 1.3 B text-to-video model."""

    def __init__(self, t2v_model_path: str, output_path: str = "outputs"):
        self.model_path = t2v_model_path
        self.output_path = output_path

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Delay helps avoid GPU contention when many replicas start at once
        time.sleep(5)

        # Ensure correct attention backend for FastVideo
        if "FastVideo" in self.model_path:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

        # Lazy import to keep the deployment import-safe on head node
        from fastvideo.entrypoints.video_generator import VideoGenerator
        from fastvideo.configs.sample.base import SamplingParam

        print(f"Initializing T2V model: {self.model_path}")
        self.generator = VideoGenerator.from_pretrained(
            model_path=self.model_path,
            num_gpus=1,
            use_fsdp_inference=True,
            text_encoder_cpu_offload=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            VSA_sparsity=0.8,
            enable_stage_verification=False,
        )
        self.default_params = SamplingParam.from_pretrained(self.model_path)
        print("✅ T2V model initialized successfully")

    def generate_video(self, video_request: "VideoGenerationRequest") -> "VideoGenerationResponse":
        """Generate a video for the given request and return an encoded response."""
        import time
        
        total_start_time = time.time()
        
        # Deep-copy default sampling params and override with request values
        params = deepcopy(self.default_params)
        params.prompt = video_request.prompt
        if video_request.use_negative_prompt:
            params.negative_prompt = video_request.negative_prompt

        params.seed = video_request.seed if not video_request.randomize_seed else torch.randint(0, 1_000_000, (1,)).item()
        # Ensure the generator respects the chosen seed strategy
        params.randomize_seed = video_request.randomize_seed
        params.guidance_scale = video_request.guidance_scale
        params.num_frames = video_request.num_frames
        params.height = video_request.height
        params.width = video_request.width
        # params.num_inference_steps = video_request.num_inference_steps

        # Do not write to disk when called via API
        params.save_video = False
        params.return_frames = False

        # Track inference time
        inference_start_time = time.time()
        
        # Generate video frames
        result = self.generator.generate_video(
            prompt=video_request.prompt,
            sampling_param=params,
            save_video=False,
            return_frames=False,
        )
        
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        frames = result if isinstance(result, list) else result.get("frames", [])
        generation_time = result.get("generation_time", 0.0) if isinstance(result, dict) else 0.0


        logging_info = result.get("logging_info", None)
        if logging_info:
            stage_names = logging_info.get_execution_order()
            stage_execution_times = [logging_info.get_stage_info(stage_name).get("execution_time", 0.0) for stage_name in stage_names]
        else:
            stage_names = []
            stage_execution_times = []

        # Track encoding time
        encoding_start_time = time.time()
        
        # Encode outputs
        video_data = encode_video_to_base64(frames, fps=16)
        
        encoding_end_time = time.time()
        encoding_time = encoding_end_time - encoding_start_time
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        return VideoGenerationResponse(
            video_data=video_data,
            seed=params.seed,
            success=True,
            generation_time=generation_time,
            inference_time=inference_time,
            encoding_time=encoding_time,
            total_time=total_time,
            stage_names=stage_names,
            stage_execution_times=stage_execution_times,
        )


# @serve.deployment(  # I2V 14B model deployment - I2V functionality commented out
#     ray_actor_options={"num_cpus": 10, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
# )
# class I2VModelDeployment:
#     """Serve deployment wrapping the 14 B image-to-video model."""
# 
#     def __init__(self, i2v_model_path: str, output_path: str = "outputs"):
#         self.model_path = i2v_model_path
#         self.output_path = output_path
# 
#         os.makedirs(self.output_path, exist_ok=True)
#         time.sleep(10)
# 
#         os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
# 
#         from fastvideo.entrypoints.video_generator import VideoGenerator
#         from fastvideo.configs.sample.base import SamplingParam
# 
#         print(f"Initializing I2V model: {self.model_path}")
#         self.generator = VideoGenerator.from_pretrained(
#             model_path=self.model_path,
#             num_gpus=1,
#             use_fsdp_inference=True,
#             text_encoder_cpu_offload=False,
#             dit_cpu_offload=False,
#             vae_cpu_offload=False,
#             VSA_sparsity=0.8,
#         )
#         self.default_params = SamplingParam.from_pretrained(self.model_path)
#         print("✅ I2V model initialized successfully")
# 
#     def generate_video(self, video_request: "VideoGenerationRequest") -> "VideoGenerationResponse":
#         import time
#         
#         total_start_time = time.time()
#         
#         params = deepcopy(self.default_params)
# 
#         params.prompt = video_request.prompt
#         if video_request.use_negative_prompt:
#             params.negative_prompt = video_request.negative_prompt
# 
#         params.seed = video_request.seed if not video_request.randomize_seed else torch.randint(0, 1_000_000, (1,)).item()
#         params.guidance_scale = video_request.guidance_scale
#         params.num_frames = video_request.num_frames
#         params.height = video_request.height
#         params.width = video_request.width
#         params.num_inference_steps = video_request.num_inference_steps
# 
#         if video_request.image_path:
#             params.image_path = video_request.image_path
# 
#         params.save_video = False
#         params.return_frames = True
# 
#         # Track inference time
#         inference_start_time = time.time()
#         
#         result = self.generator.generate_video(
#             prompt=video_request.prompt,
#             sampling_param=params,
#             save_video=False,
#             return_frames=True,
#         )
#         
#         inference_end_time = time.time()
#         inference_time = inference_end_time - inference_start_time
# 
#         frames = result if isinstance(result, list) else result.get("frames", [])
#         generation_time = result.get("generation_time", 0.0) if isinstance(result, dict) else 0.0
# 
#         # Track encoding time
#         encoding_start_time = time.time()
#         
#         video_data = encode_video_to_base64(frames, fps=24)
#         encoded_frames = encode_frames_to_base64(frames) if video_request.return_frames and frames else None
#         
#         encoding_end_time = time.time()
#         encoding_time = encoding_end_time - encoding_start_time
#         
#         total_end_time = time.time()
#         total_time = total_end_time - total_start_time
# 
#         return VideoGenerationResponse(
#             video_data=video_data,
#             frames=encoded_frames,
#             seed=params.seed,
#             success=True,
#             generation_time=generation_time,
#             inference_time=inference_time,
#             encoding_time=encoding_time,
#             total_time=total_time,
#         )


@serve.deployment(  # T2V 14B model deployment with optimized settings
    ray_actor_options={"num_cpus": 16, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
)
class T2V14BModelDeployment:
    """Serve deployment wrapping the 14B text-to-video model with optimized settings."""

    def __init__(self, t2v_14b_model_path: str, output_path: str = "outputs"):
        self.model_path = t2v_14b_model_path
        self.output_path = output_path

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Delay helps avoid GPU contention when many replicas start at once
        time.sleep(10)

        # Ensure correct attention backend for FastVideo
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

        # Lazy import to keep the deployment import-safe on head node
        from fastvideo.entrypoints.video_generator import VideoGenerator
        from fastvideo.configs.sample.base import SamplingParam

        print(f"Initializing T2V 14B model: {self.model_path}")
        self.generator = VideoGenerator.from_pretrained(
            model_path=self.model_path,
            num_gpus=1,
            use_fsdp_inference=True,
            text_encoder_cpu_offload=True,  # Enable CPU offload for 14B model
            dit_cpu_offload=True,           # Enable CPU offload for 14B model
            vae_cpu_offload=False,
            VSA_sparsity=0.9,               # Higher sparsity for 14B model
            enable_stage_verification=False,
        )
        self.default_params = SamplingParam.from_pretrained(self.model_path)
        print("✅ T2V 14B model initialized successfully")

    def generate_video(self, video_request: "VideoGenerationRequest") -> "VideoGenerationResponse":
        """Generate a video for the given request and return an encoded response."""
        import time
        
        total_start_time = time.time()
        
        # Deep-copy default sampling params and override with request values
        params = deepcopy(self.default_params)
        params.prompt = video_request.prompt
        if video_request.use_negative_prompt:
            params.negative_prompt = video_request.negative_prompt

        params.seed = video_request.seed if not video_request.randomize_seed else torch.randint(0, 1_000_000, (1,)).item()
        # Ensure the generator respects the chosen seed strategy
        params.randomize_seed = video_request.randomize_seed
        params.guidance_scale = video_request.guidance_scale
        params.num_frames = video_request.num_frames
        params.height = video_request.height
        params.width = video_request.width
        # params.num_inference_steps = video_request.num_inference_steps

        # Do not write to disk when called via API
        params.save_video = False
        params.return_frames = False

        # Track inference time
        inference_start_time = time.time()
        
        # Generate video frames
        result = self.generator.generate_video(
            prompt=video_request.prompt,
            sampling_param=params,
            save_video=False,
            return_frames=False,
        )
        
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        frames = result if isinstance(result, list) else result.get("frames", [])
        generation_time = result.get("generation_time", 0.0) if isinstance(result, dict) else 0.0
        logging_info = result.get("logging_info", None)
        if logging_info:
            stage_names = logging_info.get_execution_order()
            stage_execution_times = [logging_info.get_stage_info(stage_name).get("execution_time", 0.0) for stage_name in stage_names]
        else:
            stage_names = []
            stage_execution_times = []

        # Track encoding time
        encoding_start_time = time.time()
        
        # Encode outputs
        video_data = encode_video_to_base64(frames, fps=16)
        
        encoding_end_time = time.time()
        encoding_time = encoding_end_time - encoding_start_time
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        return VideoGenerationResponse(
            video_data=video_data,
            seed=params.seed,
            success=True,
            generation_time=generation_time,
            inference_time=inference_time,
            encoding_time=encoding_time,
            total_time=total_time,
            stage_names=stage_names,
            stage_execution_times=stage_execution_times,
        )


# Create FastAPI app with rate limiting
app = FastAPI()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@serve.deployment(ray_actor_options={"num_cpus": 2})
@serve.ingress(app)
class FastVideoAPI:
    """Ingress deployment that routes requests to either the T2V or I2V model deployment."""

    def __init__(self, t2v_deployments: Dict[str, DeploymentHandle]):  # Removed i2v_deployment
        self.t2v_deployments = t2v_deployments
        # self.t2v_14b_handle = t2v_14b_deployment
        # self.i2v_handle = i2v_deployment  # I2V functionality commented out
    
    @app.post("/generate_video", response_model=VideoGenerationResponse)
    @limiter.limit("50/minute")  # Allow 50 requests per minute per IP
    async def generate_video(self, request: Request, video_request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Route the request to the appropriate model deployment based on `model_type` and `model_path`."""
        try:
            # assert video_request.model_type in self.t2v_deployments, f"Model {video_request.model_type} not found"
            if video_request.model_type not in self.t2v_deployments:
                raise ValueError(f"Model {video_request.model_type} not found")
            response_ref = self.t2v_deployments[video_request.model_type].generate_video.remote(video_request)
            # if video_request.model_type.lower() == "i2v":  # I2V functionality commented out
            #     response_ref = self.i2v_handle.generate_video.remote(video_request)
            # if video_request.model_type.lower() == "t2v":
            #     # Route T2V requests based on model path
            #     if video_request.model_path and "14b" in video_request.model_path.lower():
            #         response_ref = self.t2v_14b_handle.generate_video.remote(video_request)
            #     else:
            #         # Default to 1.3B model
            #         response_ref = self.t2v_handle.generate_video.remote(video_request)
            # else:
            #     # Default to 1.3B T2V model
            #     response_ref = self.t2v_handle.generate_video.remote(video_request)

            # Await the remote response
            response = await response_ref
            
            return response

        except Exception as e:
            return VideoGenerationResponse(
                video_data=None,
                seed=video_request.seed,
                success=False,
                error_message=str(e),
            )
    
    @app.get("/health")
    @limiter.limit("10/minute")  # Allow 10 health checks per minute per IP
    async def health_check(self, request: Request):
        return {"status": "healthy"}
    



def start_ray_serve(
    *,
    t2v_model_paths: str,
    t2v_model_replicas: str,
    # t2v_14b_model_path: str = "FastVideo/FastWan2.1-T2V-14B-Diffusers",
    # i2v_model_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",  # I2V functionality commented out
    output_path: str = "outputs",
    host: str = "0.0.0.0",
    port: int = 8000,
    # t2v_replicas: int = 4,
    # t2v_14b_replicas: int = 4,  # Reduced replicas for 14B model due to higher resource requirements
    # i2v_replicas: int = 4,  # I2V functionality commented out
):
    """Start Ray Serve with independently scalable deployments for each model."""

    if not ray.is_initialized():
        ray.init()

    # Bind model deployments with configurable replica counts
    t2v_deps = {}
    for model_path, replicas in zip(t2v_model_paths.split(","), t2v_model_replicas.split(",")):
        t2v_dep = T2VModelDeployment.options(num_replicas=int(replicas)).bind(model_path, output_path)
        t2v_deps[model_path] = t2v_dep
    # i2v_dep = I2VModelDeployment.options(num_replicas=i2v_replicas).bind(i2v_model_path, output_path)  # I2V functionality commented out

    # Ingress
    api = FastVideoAPI.bind(t2v_deps)  # Removed i2v_dep

    serve.run(api, route_prefix="/", name="fast_video")

    print(f"Ray Serve backend started at http://{host}:{port}")
    for model_path, replicas in zip(t2v_model_paths.split(","), t2v_model_replicas.split(",")):
        print(f"T2V Model: {model_path} | Replicas: {replicas}")
    # print(f"T2V 14B Model: {t2v_14b_model_path} | Replicas: {t2v_14b_replicas}")
    # print(f"I2V Model: {i2v_model_path} | Replicas: {i2v_replicas}")  # I2V functionality commented out
    print(f"Health check: http://{host}:{port}/health")
    print(f"Video generation endpoint: http://{host}:{port}/generate_video")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve Backend")
    parser.add_argument("--t2v_model_paths",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.1-T2V-14B-Diffusers",
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--t2v_model_replicas",
                        type=str,
                        default="4,4",
                        help="Comma separated list of number of replicas for the T2V model(s)")
    # parser.add_argument("--t2v_14b_model_path",
    #                     type=str,
    #                     default="FastVideo/FastWan2.1-T2V-14B-Diffusers",
    #                     help="Path to the T2V 14B model")
    # parser.add_argument("--i2v_model_path",  # I2V functionality commented out
    #                     type=str,
    #                     default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    #                     help="Path to the I2V model")
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs",
                        help="Path to save generated videos")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Port to bind to")
    # parser.add_argument("--t2v_replicas",
    #                     type=int,
    #                     default=4,
    #                     help="Number of replicas for the T2V model deployment")
    # parser.add_argument("--t2v_14b_replicas",
    #                     type=int,
    #                     default=4,  # Reduced default for 14B model due to higher resource requirements
    #                     help="Number of replicas for the T2V 14B model deployment")
    # parser.add_argument("--i2v_replicas",  # I2V functionality commented out
    #                     type=int,
    #                     default=4,  # Reduced default for 14B model due to higher resource requirements
    #                     help="Number of replicas for the I2V model deployment")
    
    args = parser.parse_args()

    split_models = args.t2v_model_paths.split(",")
    split_replicas = [int(replica) for replica in args.t2v_model_replicas.split(",")]
    assert len(split_models) == len(split_replicas), "Number of models and replicas must match"
    assert sum(split_replicas) <= NUM_GPUS, "Total number of replicas must be less than or equal to 16"
    for model, replicas in zip(split_models, split_replicas):
        assert model in SUPPORTED_MODELS, f"Model {model} not supported"
        assert replicas > 0, f"Replicas must be greater than 0"

    
    start_ray_serve(
        t2v_model_paths=args.t2v_model_paths,
        t2v_model_replicas=args.t2v_model_replicas,
        # t2v_14b_model_path=args.t2v_14b_model_path,
        # i2v_model_path=args.i2v_model_path,  # I2V functionality commented out
        output_path=args.output_path,
        host=args.host,
        port=args.port,
        # t2v_replicas=args.t2v_replicas,
        # t2v_14b_replicas=args.t2v_14b_replicas,
        # i2v_replicas=args.i2v_replicas,  # I2V functionality commented out
    )

    # ---- keep the process alive ---------------------------------
    import signal, sys, time
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))   # Ctrl-C
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # docker stop etc.

    print("✅ FastVideo backend is running. Press Ctrl-C to stop.")
    while True:
        time.sleep(3600) 