import time
import os
import torch
import base64
import io
from copy import deepcopy
from typing import Dict, Any, Optional, List
import signal
import sys

import ray
from ray import serve
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import imageio
from ray.serve.handle import DeploymentHandle
from prometheus_client import Counter, Histogram, generate_latest

NUM_GPUS = 16
DEFAULT_FPS = 16
SEED_RANGE_MAX = 1_000_000
SUPPORTED_MODELS = [
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", 
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
]

MODEL_CONFIGS = {
    "1.3B": {
        "num_cpus": 2,
        "text_encoder_cpu_offload": False,
        "dit_cpu_offload": False,
        "vae_cpu_offload": False,
        "VSA_sparsity": 0.8,
    },
    "14B": {
        "num_cpus": 16,
        "text_encoder_cpu_offload": True,
        "dit_cpu_offload": True,
        "vae_cpu_offload": False,
        "VSA_sparsity": 0.9,
    }
}


class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    use_negative_prompt: bool = False
    seed: int = 42
    guidance_scale: float = 7.5
    num_frames: int = 21
    height: int = 448
    width: int = 832
    randomize_seed: bool = False
    return_frames: bool = False
    model_path: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    video_data: Optional[str] = None
    seed: int
    success: bool
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    model_load_time: Optional[float] = None
    inference_time: Optional[float] = None
    encoding_time: Optional[float] = None
    total_time: Optional[float] = None
    stage_names: Optional[List[str]] = None
    stage_execution_times: Optional[List[float]] = None


def encode_video_to_base64(frames: List[np.ndarray], fps: int = DEFAULT_FPS) -> str:
    if not frames:
        return ""
    
    try:
        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, fps=fps, format="mp4")
        buffer.seek(0)
        
        video_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:video/mp4;base64,{video_base64}"
        
    except Exception as e:
        print(f"Warning: Failed to encode video: {e}")
        return ""


def setup_model_environment(model_path: str) -> None:
    if "fullattn" in model_path.lower():
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    else:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"


def process_generation_result(result: Any) -> tuple[List[np.ndarray], float, List[str], List[float]]:
    frames = result if isinstance(result, list) else result.get("frames", [])
    generation_time = result.get("generation_time", 0.0) if isinstance(result, dict) else 0.0
    
    logging_info = result.get("logging_info", None)
    if logging_info:
        stage_names = logging_info.get_execution_order()
        stage_execution_times = [
            logging_info.get_stage_info(stage_name).get("execution_time", 0.0) 
            for stage_name in stage_names
        ]
    else:
        stage_names = []
        stage_execution_times = []
    
    return frames, generation_time, stage_names, stage_execution_times


def prepare_sampling_params(video_request: VideoGenerationRequest, default_params: Any) -> Any:
    params = deepcopy(default_params)
    params.prompt = video_request.prompt
    
    if video_request.use_negative_prompt:
        params.negative_prompt = video_request.negative_prompt

    params.seed = (video_request.seed if not video_request.randomize_seed 
                  else torch.randint(0, SEED_RANGE_MAX, (1,)).item())
    params.randomize_seed = video_request.randomize_seed
    params.guidance_scale = video_request.guidance_scale
    params.num_frames = video_request.num_frames
    params.height = video_request.height
    params.width = video_request.width
    params.save_video = False
    params.return_frames = False
    
    return params


class BaseModelDeployment:
    def __init__(self, model_path: str, output_path: str = "outputs"):
        self.model_path = model_path
        self.output_path = output_path
        self.generator = None
        self.default_params = None

        os.makedirs(self.output_path, exist_ok=True)
        setup_model_environment(self.model_path)

    def _initialize_generator(self, config: Dict[str, Any]) -> None:
        from fastvideo.entrypoints.video_generator import VideoGenerator
        from fastvideo.configs.sample.base import SamplingParam

        print(f"Initializing model: {self.model_path}")
        self.generator = VideoGenerator.from_pretrained(
            model_path=self.model_path,
            num_gpus=1,
            use_fsdp_inference=True,
            text_encoder_cpu_offload=config["text_encoder_cpu_offload"],
            dit_cpu_offload=config["dit_cpu_offload"],
            vae_cpu_offload=config["vae_cpu_offload"],
            VSA_sparsity=config["VSA_sparsity"],
            enable_stage_verification=False,
        )
        self.default_params = SamplingParam.from_pretrained(self.model_path)

    def generate_video(self, video_request: VideoGenerationRequest) -> VideoGenerationResponse:
        total_start_time = time.time()
        
        params = prepare_sampling_params(video_request, self.default_params)

        inference_start_time = time.time()
        result = self.generator.generate_video(
            prompt=video_request.prompt,
            sampling_param=params,
            save_video=False,
            return_frames=False,
        )
        inference_time = time.time() - inference_start_time

        frames, generation_time, stage_names, stage_execution_times = process_generation_result(result)

        encoding_start_time = time.time()
        video_data = encode_video_to_base64(frames, fps=DEFAULT_FPS)
        encoding_time = time.time() - encoding_start_time
        
        total_time = time.time() - total_start_time

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


@serve.deployment(
    ray_actor_options={"num_cpus": 2, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
)
class T2VModelDeployment(BaseModelDeployment):
    def __init__(self, t2v_model_path: str, output_path: str = "outputs"):
        super().__init__(t2v_model_path, output_path)
        self._initialize_generator(MODEL_CONFIGS["1.3B"])
        print("✅ T2V model initialized successfully")


@serve.deployment(
    ray_actor_options={"num_cpus": 16, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
)
class T2V14BModelDeployment(BaseModelDeployment):
    def __init__(self, t2v_14b_model_path: str, output_path: str = "outputs"):
        super().__init__(t2v_14b_model_path, output_path)
        # Override environment for 14B model
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        self._initialize_generator(MODEL_CONFIGS["14B"])
        print("✅ T2V 14B model initialized successfully")


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@serve.deployment(num_replicas=50, ray_actor_options={"num_cpus": 2})
@serve.ingress(app)
class FastVideoAPI:

    def __init__(self, t2v_deployments: Dict[str, DeploymentHandle]):
        self.t2v_deployments = t2v_deployments
        
        # Initialize Prometheus metrics
        self.request_count = Counter('fastvideo_requests_total', 'Total FastVideo requests', ['model_type', 'status'])
        self.request_duration = Histogram('fastvideo_request_duration_seconds', 'FastVideo request duration', ['model_type'])
        self.video_generation_time = Histogram('fastvideo_video_generation_seconds', 'Video generation time', ['model_type'])
    
    def _get_model_name(self, model_path: Optional[str]) -> str:
        return model_path.split('/')[-1] if model_path else "unknown"
    
    def _record_metrics(self, model_name: str, status: str, duration: float, response: Optional[VideoGenerationResponse] = None) -> None:
        self.request_count.labels(model_type=model_name, status=status).inc()
        self.request_duration.labels(model_type=model_name).observe(duration)
        
        if response and hasattr(response, 'generation_time') and response.generation_time:
            self.video_generation_time.labels(model_type=model_name).observe(response.generation_time)
    
    @app.post("/generate_video", response_model=VideoGenerationResponse)
    @limiter.limit("10/minute")
    async def generate_video(self, request: Request, video_request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Route the request to the appropriate model deployment based on model_path."""
        start_time = time.time()
        model_name = self._get_model_name(video_request.model_path)
        
        try:
            if video_request.model_path not in self.t2v_deployments:
                raise ValueError(f"Model {video_request.model_path} not found")
            
            response_ref = self.t2v_deployments[video_request.model_path].generate_video.remote(video_request)
            response = await response_ref
            
            self._record_metrics(model_name, "success", time.time() - start_time, response)
            return response

        except Exception as e:
            self._record_metrics(model_name, "error", time.time() - start_time)
            
            return VideoGenerationResponse(
                video_data=None,
                seed=video_request.seed,
                success=False,
                error_message=str(e),
                generation_time=0,
                inference_time=0,
                encoding_time=0,
                total_time=0,
            )
    
    @app.get("/health")
    @limiter.limit("10/minute")
    async def health_check(self, request: Request) -> Dict[str, str]:
        return {"status": "healthy"}
    
    @app.get("/metrics")
    async def metrics(self) -> Response:
        return Response(generate_latest(), media_type="text/plain")


def validate_configuration(model_paths: List[str], replicas: List[int]) -> None:
    assert len(model_paths) == len(replicas), "Number of models and replicas must match"
    assert sum(replicas) <= NUM_GPUS, f"Total replicas ({sum(replicas)}) must be <= {NUM_GPUS}"
    
    for model, replica_count in zip(model_paths, replicas):
        assert model in SUPPORTED_MODELS, f"Model {model} not supported"
        assert replica_count > 0, f"Replicas must be greater than 0"


def start_ray_serve(
    *,
    t2v_model_paths: str,
    t2v_model_replicas: str,
    output_path: str = "outputs",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    if not ray.is_initialized():
        ray.init()

    model_paths = t2v_model_paths.split(",")
    replicas = [int(r) for r in t2v_model_replicas.split(",")]
    validate_configuration(model_paths, replicas)

    t2v_deps = {}
    for model_path, replica_count in zip(model_paths, replicas):
        t2v_dep = T2VModelDeployment.options(num_replicas=replica_count).bind(model_path, output_path)
        t2v_deps[model_path] = t2v_dep

    api = FastVideoAPI.bind(t2v_deps)
    serve.run(api, route_prefix="/", name="fast_video")

    print(f"Ray Serve backend started at http://{host}:{port}")
    for model_path, replica_count in zip(model_paths, replicas):
        print(f"T2V Model: {model_path} | Replicas: {replica_count}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Video generation endpoint: http://{host}:{port}/generate_video")


def setup_signal_handlers() -> None:
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve Backend")
    parser.add_argument("--t2v_model_paths",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--t2v_model_replicas",
                        type=str,
                        default="4,4",
                        help="Comma separated list of number of replicas for the T2V model(s)")
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
    
    args = parser.parse_args()

    model_paths = args.t2v_model_paths.split(",")
    replicas = [int(r) for r in args.t2v_model_replicas.split(",")]
    validate_configuration(model_paths, replicas)
    
    start_ray_serve(
        t2v_model_paths=args.t2v_model_paths,
        t2v_model_replicas=args.t2v_model_replicas,
        output_path=args.output_path,
        host=args.host,
        port=args.port,
    )

    setup_signal_handlers()
    print("✅ FastVideo backend is running. Press Ctrl-C to stop.")
    while True:
        time.sleep(3600) 