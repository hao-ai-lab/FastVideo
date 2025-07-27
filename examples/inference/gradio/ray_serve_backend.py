import time
import os
import torch
import base64
import io
from copy import deepcopy
from typing import Dict, Any, Optional, List

import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import numpy as np


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
    return_frames: bool = False  # Whether to return base64 encoded frames


class VideoGenerationResponse(BaseModel):
    output_path: str
    seed: int
    success: bool
    error_message: Optional[str] = None
    frames: Optional[List[str]] = None  # Base64 encoded frames


def encode_frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """Convert numpy frames (0-255) to base64-encoded PNG images"""
    if not frames:
        return []
    
    encoded_frames = []
    
    for i, frame in enumerate(frames):
        try:
            # Ensure frame is numpy array
            if not isinstance(frame, np.ndarray):
                print(f"Warning: Frame {i} is not a numpy array, skipping")
                continue
                
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Clip values to 0-255 range and convert to uint8
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Convert numpy array to PIL Image
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(frame, mode='RGB')
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA image
                pil_image = Image.fromarray(frame, mode='RGBA')
            elif len(frame.shape) == 2:
                # Grayscale image
                pil_image = Image.fromarray(frame, mode='L')
            else:
                print(f"Warning: Frame {i} has unsupported shape {frame.shape}, skipping")
                continue
            
            # Save to bytes buffer as PNG
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            encoded_frames.append(f"data:image/png;base64,{img_base64}")
            
        except Exception as e:
            print(f"Warning: Failed to encode frame {i}: {e}")
            continue
    
    return encoded_frames


# Create FastAPI app
app = FastAPI()


@serve.deployment(
    num_replicas=16,
    # ray_actor_options={"num_cpus": 10, "num_gpus": 1, "runtime_env": {"conda": "fv", "working_dir": "/mnt/fast-disks/nfs/hao_lab/FastVideo"}},
    ray_actor_options={"num_cpus": 1, "num_gpus": 1, "runtime_env": {"conda": "fv"}},
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
        time.sleep(10)
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
            result = self.generator.generate_video(
                prompt=request.prompt,
                sampling_param=params,
                save_video=False,  # Match the params.save_video setting
            )
            
            # The actual output path where the video was saved
            # output_path = os.path.join(self.output_path, f"{safe_prompt}.mp4")
            
            # Verify the file exists
            # if not os.path.exists(output_path):
            #     raise FileNotFoundError(f"Video was not saved to expected location: {output_path}")
            
            frames = result.get("frames", [])
            
            # Encode frames to base64 for web transmission only if requested
            encoded_frames = None
            if request.return_frames and frames:
                try:
                    encoded_frames = encode_frames_to_base64(frames)
                except Exception as e:
                    print(f"Warning: Failed to encode frames: {e}")
                    encoded_frames = None
            
            response = VideoGenerationResponse(
                output_path="",
                frames=encoded_frames,
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