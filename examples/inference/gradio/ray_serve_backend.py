import os
import torch
from copy import deepcopy
from typing import Dict, Any, Optional

import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel


class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    use_negative_prompt: bool = False
    seed: int = 42
    guidance_scale: float = 7.5
    num_frames: int = 21
    height: int = 448
    width: int = 832
    num_inference_steps: int = 20
    randomize_seed: bool = False


class VideoGenerationResponse(BaseModel):
    output_path: str
    seed: int
    success: bool
    error_message: Optional[str] = None


# Create FastAPI app
app = FastAPI()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
)
@serve.ingress(app)
class FastVideoAPI:
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        
        # Initialize the video generator
        self.generator = None # Initialize to None
        self.default_params = None # Initialize to None
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        self._initialize_model() # Ensure model is initialized
    
    def _initialize_model(self):
        if self.generator is None:
            # Set VSA environment variable
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
            
            # Import only when needed - use direct imports to avoid module-level execution
            from fastvideo.entrypoints.video_generator import VideoGenerator
            from fastvideo.configs.sample.base import SamplingParam
            
            self.generator = VideoGenerator.from_pretrained(
                model_path=self.model_path, 
                num_gpus=1,
                use_fsdp_inference=True,
                # Adjust these offload parameters if you have < 32GB of VRAM
                text_encoder_cpu_offload=False,
                dit_cpu_offload=False,
                vae_cpu_offload=False,
                VSA_sparsity=0.8,
            )
            self.default_params = SamplingParam.from_pretrained(self.model_path)
    
    @app.post("/generate_video", response_model=VideoGenerationResponse)
    async def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        try:
            # Create a copy of default parameters
            params = deepcopy(self.default_params)
            
            # Update parameters with request values
            params.prompt = request.prompt
            # Only override negative prompt if user explicitly opts in
            if request.use_negative_prompt:
                params.negative_prompt = request.negative_prompt

            params.seed = request.seed
            params.guidance_scale = request.guidance_scale
            params.num_frames = request.num_frames
            params.height = request.height
            params.width = request.width
            params.num_inference_steps = request.num_inference_steps
            
            # Handle seed randomization
            if request.randomize_seed:
                params.seed = torch.randint(0, 1000000, (1,)).item()
            
            # Ensure negative_prompt is a non-None string; FastVideo validation disallows None
            if params.negative_prompt is None:
                params.negative_prompt = ""  # empty string satisfies validator
            
            # Set up output path and video saving
            params.save_video = True
            params.output_path = self.output_path
            # params.return_frames = False  # avoid keeping frames in memory
            
            # Create a clean filename from the prompt
            safe_prompt = request.prompt[:100].replace(' ', '_').replace('/', '_').replace('\\', '_')
            # Store desired video name inside the SamplingParam to avoid unknown kwarg errors
            setattr(params, "output_video_name", safe_prompt)
            # Generate the video with proper output path and filename
            self.generator.generate_video(
                prompt=request.prompt,
                sampling_param=params,
            )
            
            # The actual output path where the video was saved
            output_path = os.path.join(self.output_path, f"{safe_prompt}.mp4")
            
            # Verify the file exists
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Video was not saved to expected location: {output_path}")
            
            
            response = VideoGenerationResponse(
                output_path=output_path,
                seed=params.seed,
                success=True
            )
            
            # Memory cleanup to avoid OOM in repeated generations
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response
        except Exception as e:
            return VideoGenerationResponse(
                output_path="",
                seed=request.seed,
                success=False,
                error_message=str(e)
            )
    
    @app.get("/health")
    async def health_check(self):
        return {"status": "healthy"}


def start_ray_serve(
    model_path: str = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    output_path: str = "outputs",
    host: str = "0.0.0.0",
    port: int = 8000
):
    """Start the Ray Serve backend"""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Deploy the API
    api = FastVideoAPI.bind(model_path, output_path)
    serve.run(api, route_prefix="/", name="fast_video")  # detach
    
    print(f"Ray Serve backend started at http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Video generation endpoint: http://{host}:{port}/generate_video")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve Backend")
    parser.add_argument("--model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the model")
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
    
    start_ray_serve(
        model_path=args.model_path,
        output_path=args.output_path,
        host=args.host,
        port=args.port,
    )

    # ---- keep the process alive ---------------------------------
    import signal, sys, time
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))   # Ctrl-C
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # docker stop etc.

    print("✅ FastVideo backend is running. Press Ctrl-C to stop.")
    while True:
        time.sleep(3600) 