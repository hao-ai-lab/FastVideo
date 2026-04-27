import argparse
import os
from pathlib import Path

import gradio as gr

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.layers.quantization.fp4_config import FP4Config
from fastvideo.utils import maybe_download_model

from .config import (
    GENERATED_CLIP_ROOT,
    MODEL_ID,
    apply_ltx2_defaults,
    resolve_model_path,
    resolve_refine_upsampler_path,
    setup_model_environment,
)
from .ui import create_gradio_interface

def main():
    parser = argparse.ArgumentParser(description="FastVideo Gradio Local Demo")
    parser.add_argument("--t2v_model_paths", type=str,
                        default=MODEL_ID,
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to bind to")
    args = parser.parse_args()
    gradio_temp_dir = os.path.abspath("outputs/gradio_tmp")
    os.makedirs(gradio_temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = gradio_temp_dir
    generators = {}
    default_params = {}
    model_paths = args.t2v_model_paths.split(",")
    for model_path in model_paths:
        print(f"Loading model: {model_path}")
        setup_model_environment(model_path)
        resolved_model_input = str(resolve_model_path(model_path))
        model_root = maybe_download_model(resolved_model_input)
        resolved_model_path = Path(model_root)

        pipeline_config = PipelineConfig.from_pretrained(str(resolved_model_path))
        pipeline_config.dit_config.quant_config = FP4Config()
        refine_upsampler_path = resolve_refine_upsampler_path(resolved_model_path)
        print(f"Using refine upsampler: {refine_upsampler_path}")

        generators[model_path] = VideoGenerator.from_pretrained(
            str(resolved_model_path),
            num_gpus=1,
            ltx2_refine_enabled=True,
            ltx2_refine_upsampler_path=str(refine_upsampler_path),
            ltx2_refine_lora_path="",  # disable refine LoRA for distilled model
            ltx2_refine_num_inference_steps=2,
            ltx2_refine_guidance_scale=1.0,
            ltx2_refine_add_noise=True,
            pipeline_config=pipeline_config,
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=True,
            torch_compile_kwargs={
                "backend": "inductor",
                "fullgraph": True,
                "mode": "max-autotune-no-cudagraphs",
                "dynamic": False,
            },
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            ltx2_vae_tiling=False,
        )
        default_params[model_path] = apply_ltx2_defaults(
            SamplingParam.from_pretrained(str(resolved_model_path))
        )
    demo = create_gradio_interface(default_params, generators)
    print(f"Starting Gradio frontend at http://{args.host}:{args.port}")
    print(f"T2V Models: {args.t2v_model_paths}")
    
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/logo.png")
    def get_logo():
        return FileResponse(
            "assets/full.svg",
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )

    @app.get("/nvidia.png")
    def get_nvidia_logo():
        return FileResponse(
            "assets/nv.png",
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    @app.get("/favicon.ico")
    def get_favicon():
        favicon_path = "assets/icon-simple.svg"
        
        if os.path.exists(favicon_path):
            return FileResponse(
                favicon_path, 
                media_type="image/svg+xml",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Favicon not found")

    @app.get("/generated-clips/{clip_path:path}")
    def get_generated_clip(clip_path: str):
        root = GENERATED_CLIP_ROOT.resolve()
        resolved_path = (root / clip_path).resolve()

        if root not in resolved_path.parents or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Clip not found")

        return FileResponse(
            resolved_path,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )
    
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        base_url = str(request.base_url).rstrip('/')
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            
            <title>FastLTX-2.3</title>
            <meta name="title" content="FastLTX-2.3">
            <meta name="description" content="Make video generation go blurrrrrrr">
            <meta name="keywords" content="FastVideo, video generation, AI, machine learning, FastLTX-2.3">
            
            <meta property="og:type" content="website">
            <meta property="og:url" content="{base_url}/">
            <meta property="og:title" content="FastLTX-2.3">
            <meta property="og:description" content="Make video generation go blurrrrrrr">
            <meta property="og:image" content="{base_url}/logo.png">
            <meta property="og:image:width" content="1200">
            <meta property="og:image:height" content="630">
            <meta property="og:site_name" content="FastLTX-2.3">
            
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="{base_url}/">
            <meta property="twitter:title" content="FastLTX-2.3">
            <meta property="twitter:description" content="Make video generation go blurrrrrrr">
            <meta property="twitter:image" content="{base_url}/logo.png">
            <link rel="icon" type="image/png" sizes="32x32" href="/favicon.ico">
            <link rel="icon" type="image/png" sizes="16x16" href="/favicon.ico">
            <link rel="apple-touch-icon" href="/favicon.ico">
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    min-height: 100%;
                    width: 100%;
                    background: #000;
                    background-color: #000;
                    background-image: none;
                    overscroll-behavior-y: auto;
                    scroll-behavior: smooth;
                }}
                body {{
                    position: relative;
                }}
                body::before {{
                    content: "";
                    position: fixed;
                    inset: 0;
                    background: #000;
                    pointer-events: none;
                    z-index: -1;
                }}
                iframe {{
                    display: block;
                    width: 100%;
                    height: 100vh;
                    background: #000;
                    background-color: #000;
                    background-image: none;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <iframe src="/gradio" width="100%" height="100%" style="border: none;"></iframe>
        </body>
        </html>
        """
    
    app = gr.mount_gradio_app(
        app, 
        demo, 
        path="/gradio",
        allowed_paths=[
            os.path.abspath("outputs"),
            os.path.abspath("outputs_video"),
            os.path.abspath("fastvideo-logos"),
        ]
    )
    
    uvicorn.run(app, host=args.host, port=args.port)
