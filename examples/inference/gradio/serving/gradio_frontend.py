import argparse
import os
import requests
import base64
import time
import json
from pathlib import Path
import tempfile

import gradio as gr

from fastvideo.configs.sample.base import SamplingParam


MODEL_PATH_MAPPING = {
    "FastWan2.1-T2V-1.3B": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "FastWan2.2-TI2V-5B-FullAttn": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    "CausalWan2.2-I2V-A14B-Preview": "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
}


class RayServeClient:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        try:
            response = self.session.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_video(self, request_data: dict) -> dict:
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            
            response = self.session.post(
                f"{self.backend_url}/generate_video",
                json=request_data,
                headers=headers,
                timeout=900  # 15 minutes timeout for longer video generation
            )
            
            round_trip_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                backend_total = result.get("total_time", 0)
                network_time = round_trip_time - backend_total
                result["network_time"] = network_time
                return result
            else:
                return {"success": False, "error_message": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error_message": f"Request failed: {str(e)}"}


def save_video_from_base64(video_data: str, output_dir: str, prompt: str) -> str:
    if not video_data:
        return None
    
    try:
        if video_data.startswith('data:video/'):
            video_data = video_data.split(',')[1]
        
        video_bytes = base64.b64decode(video_data)
        
        safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '_')
        video_filename = f"{safe_prompt}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        
        return video_path
        
    except Exception as e:
        print(f"Failed to save video: {e}")
        return None


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine image type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{image_base64}"
        
    except Exception as e:
        print(f"Failed to encode image: {e}")
        return None


# def create_timing_display(inference_time, encoding_time, network_time, total_time, stage_execution_times, num_frames):
#     dit_denoising_time = f"{stage_execution_times[5]:.2f}s" if len(stage_execution_times) > 5 else "N/A"
#     
#     timing_html = f"""
#     <div style="margin: 10px 0;">
#         <h3 style="text-align: center; margin-bottom: 10px;">⏱️ Timing Breakdown</h3>
#         <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 10px;">
#             <div class="timing-card timing-card-highlight">
#                 <div style="font-size: 20px;">🚀</div>
#                 <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">DiT Denoising</div>
#                 <div style="font-size: 16px; color: #ffa200; font-weight: bold;">{dit_denoising_time}</div>
#             </div>
#             <div class="timing-card">
#                 <div style="font-size: 20px;">🧠</div>
#                 <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">E2E (w. vae/text encoder)</div>
#                 <div style="font-size: 16px; color: #2563eb;">{inference_time:.2f}s</div>
#             </div>
#             <div class="timing-card">
#                 <div style="font-size: 20px;">🎬</div>
#                 <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Video Encoding</div>
#                 <div style="font-size: 16px; color: #dc2626;">{encoding_time:.2f}s</div>
#             </div>
#             <div class="timing-card">
#                 <div style="font-size: 20px;">🌐</div>
#                 <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Network Transfer</div>
#                 <div style="font-size: 16px; color: #059669;">{network_time:.2f}s</div>
#             </div>
#             <div class="timing-card">
#                 <div style="font-size: 20px;">📊</div>
#                 <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Total Processing</div>
#                 <div style="font-size: 18px; color: #0277bd;">{total_time:.2f}s</div>
#             </div>
#         </div>"""
#     
#     if inference_time > 0:
#         fps = num_frames / inference_time
#         timing_html += f"""
#         <div class="performance-card" style="margin-top: 15px;">
#             <span style="font-weight: bold;">Generation Speed: </span>
#             <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
#         </div>"""
#     
#     return timing_html + "</div>"


def load_example_prompts():
    def contains_chinese(text):
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def load_from_file(filepath):
        prompts, labels = [], []
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not contains_chinese(line):
                        label = line[:100] + "..." if len(line) > 100 else line
                        labels.append(label)
                        prompts.append(line)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
        return prompts, labels
    
    # Load prompts from prompts.txt
    examples, example_labels = load_from_file("examples/inference/gradio/serving/prompts.txt")
    
    if not examples:
        examples = ["A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background."]
        example_labels = ["Crowded rooftop bar at night"]
    
    # Load image mappings from JSON file
    prompt_to_image = {}
    # Try to find the JSON file relative to project root
    possible_json_paths = [
        Path("prompts/mixkit_i2v.jsonl"),
        Path(__file__).parent.parent.parent.parent / "prompts" / "mixkit_i2v.jsonl",
    ]
    json_path = None
    for path in possible_json_paths:
        if path.exists():
            json_path = path
            break
    
    if json_path and json_path.exists():
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                # Get the project root directory (parent of prompts directory)
                project_root = json_path.parent.parent
                for item in data:
                    prompt_text = item.get("prompt", "").strip()
                    image_path = item.get("image_path", "")
                    if prompt_text and image_path:
                        # Resolve image path relative to project root
                        full_image_path = project_root / image_path
                        if full_image_path.exists():
                            prompt_to_image[prompt_text] = str(full_image_path.absolute())
        except Exception as e:
            print(f"Warning: Could not load image mappings from {json_path}: {e}")
    
    # Create image paths list matching the prompts
    example_images = []
    for prompt in examples:
        # Try exact match first
        image_path = prompt_to_image.get(prompt)
        if not image_path:
            # Try fuzzy match (case-insensitive, whitespace normalized)
            normalized_prompt = " ".join(prompt.split())
            for json_prompt, img_path in prompt_to_image.items():
                normalized_json = " ".join(json_prompt.split())
                if normalized_prompt.lower() == normalized_json.lower():
                    image_path = img_path
                    break
        example_images.append(image_path if image_path and os.path.exists(image_path) else None)
    
    return examples, example_labels, example_images


def create_gradio_interface(backend_url: str, default_params: dict[str, SamplingParam]):
    
    client = RayServeClient(backend_url)
    
    def is_i2v_model(model_name: str) -> bool:
        """Check if the model is an I2V model."""
        return "I2V" in model_name
    
    def generate_video(
        prompt, negative_prompt, use_negative_prompt, guidance_scale,
        num_frames, height, width, model_selection, input_image, progress
    ):
        # Use default seed value (randomize_seed disabled)
        seed = 1000
        randomize_seed = False
        if not client.check_health():
            return None, f"Backend is not available. Please check if Ray Serve is running at {backend_url}", ""
        
        # Check if I2V model requires an image
        if is_i2v_model(model_selection) and not input_image:
            return None, "I2V models require an input image. Please upload an image.", ""
        
        # Validate dimensions
        max_pixels = 720 * 1280
        if height * width > max_pixels:
            return None, f"Video dimensions too large. Maximum: 720x1280 pixels", ""
        
        if progress:
            progress(0.1, desc="Checking backend health...")
        
        # Encode image if provided
        image_data = None
        if input_image:
            if progress:
                progress(0.2, desc="Encoding input image...")
            image_data = encode_image_to_base64(input_image)
            if not image_data:
                return None, "Failed to encode input image", ""
        
        request_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_negative_prompt": use_negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "randomize_seed": randomize_seed,
            "return_frames": False,
            "image_data": image_data,
            "model_path": MODEL_PATH_MAPPING.get(model_selection, "FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
        }
        
        if progress:
            progress(0.4, desc="Generating video...")
        
        response = client.generate_video(request_data)
        
        if progress:
            progress(0.8, desc="Processing response...")
        
        if response.get("success", False):
            video_data = response.get("video_data", "")
            used_seed = response.get("seed", seed)
            # inference_time = response.get("inference_time", 0.0)
            # encoding_time = response.get("encoding_time", 0.0)
            # total_time = response.get("total_time", 0.0)
            # network_time = response.get("network_time", 0.0)
            # stage_execution_times = response.get("stage_execution_times", [])
            
            # timing_details = create_timing_display(
            #     inference_time, encoding_time, network_time, total_time, 
            #     stage_execution_times, num_frames
            # )
            
            if video_data:
                if progress:
                    progress(0.9, desc="Saving video...")
                
                video_path = save_video_from_base64(video_data, "outputs", prompt)
                
                if progress:
                    progress(1.0, desc="Generation complete!")
                
                if video_path and os.path.exists(video_path):
                    return video_path, used_seed, ""
                else:
                    return None, "Failed to save video", ""
            else:
                return None, "No video data received from backend", ""
        else:
            error_msg = response.get("error_message", "Unknown error occurred")
            return None, f"Generation failed: {error_msg}", ""
    
    examples, example_labels, example_images = load_example_prompts()
    
    theme = gr.themes.Base().set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        slider_color="#2563eb",
        checkbox_background_color_selected="#2563eb",
    )
    
    def get_default_values(model_name):
        # model_path = MODEL_PATH_MAPPING.get(model_name)
        # if model_path and model_path in default_params:
        #     params = default_params[model_path]
        #     return {
        #         'height': params.height,
        #         'width': params.width,
        #         'num_frames': params.num_frames,
        #         'guidance_scale': params.guidance_scale,
        #     }
        
        return {
            'height': 480,
            'width': 832,
            'num_frames': 73,
        }
    
    # Get available models based on what's loaded
    available_models = []
    for model_name, model_path in MODEL_PATH_MAPPING.items():
        if model_path in default_params:
            available_models.append(model_name)
    
    # Select first available model as default
    default_model = available_models[0] if available_models else "FastWan2.1-T2V-1.3B"
    initial_values = get_default_values(default_model)
    initial_show_image = is_i2v_model(default_model)
    
    with gr.Blocks(title="CausalWan", theme=theme) as demo:
        gr.Image("assets/logos/logo.svg", show_label=False, container=False, height=80)
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <p style="font-size: 18px;"> Make Video Generation Go Blurrrrrrr </p>
            <p style="font-size: 18px;"> <a href="https://github.com/hao-ai-lab/FastVideo/tree/main" target="_blank">Code</a> | <a href="https://hao-ai-lab.github.io/blogs/fastvideo_causalwan_preview/" target="_blank">Blog</a> | <a href="https://hao-ai-lab.github.io/FastVideo/" target="_blank">Docs</a>  </p>
        </div>
        """)
        
        with gr.Accordion("🎥 What Is FastVideo?", open=False):
            gr.HTML("""
            <div style="padding: 20px; line-height: 1.6;">
                <p style="font-size: 16px; margin-bottom: 15px;">
                    FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.
                </p>
            </div>
            """)
        
        with gr.Row():
            model_selection = gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="Select Model",
                interactive=True
            )

        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=example_labels,
                label="Example Prompts",
                value=None,
                interactive=True,
                allow_custom_value=False
            )
        
        with gr.Row():
            with gr.Column(scale=6):
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=3,
                    placeholder="Describe your scene...",
                    container=False,
                    lines=3,
                    autofocus=True,
                )
            with gr.Column(scale=1, min_width=120, elem_classes="center-button"):
                run_button = gr.Button("Run", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                error_output = gr.Text(label="Error", visible=False)
                # timing_display = gr.Markdown(label="Timing Breakdown", visible=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Input Image", visible=initial_show_image) as image_tab:
                        gr.Markdown("**Please make sure you upload a 480x832 image**")
                        input_image = gr.Image(
                            label="",
                            type="filepath",
                            height=400,
                        )
                    
                    with gr.Tab("Advanced Options"):
                        with gr.Group():
                            with gr.Row():
                                height = gr.Number(
                                    label="Height",
                                    value=initial_values['height'],
                                    interactive=False,
                                    container=True
                                )
                                width = gr.Number(
                                    label="Width",
                                    value=initial_values['width'],
                                    interactive=False,
                                    container=True
                                )
                            
                            with gr.Row():
                                num_frames = gr.Number(
                                    label="Number of Frames",
                                    value=initial_values['num_frames'],
                                    interactive=False,
                                    container=True
                                )
                                guidance_scale = gr.Number(
                                    label="Guidance Scale",
                                    value=1.0,
                                    interactive=False,
                                    container=True
                                )
                            
                            with gr.Row():
                                use_negative_prompt = gr.Checkbox(
                                    label="Use negative prompt", value=False)
                                negative_prompt = gr.Text(
                                    label="Negative prompt",
                                    max_lines=3,
                                    lines=3,
                                    placeholder="Enter a negative prompt",
                                    visible=False,
                                )

                            # randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                            seed_output = gr.Number(label="Used Seed", value=1000)
        
            with gr.Column(scale=1):
                result = gr.Video(
                    label="Generated Video", 
                    show_label=True,
                    height=500,
                    container=True,
                    autoplay=True,
                )
        
        gr.HTML("""
        <style>
        .center-button {
            display: flex !important;
            justify-content: center !important;
            height: 100% !important;
            padding-top: 1.4em !important;
        }
        
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        
        .gr-number input[readonly] {
            background-color: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color-subdued) !important;
            cursor: default !important;
            text-align: center !important;
            font-weight: 500 !important;
        }
        </style>
        """)
        
        def on_example_select(example_label):
            if example_label and example_label in example_labels:
                index = example_labels.index(example_label)
                selected_prompt = examples[index]
                selected_image = example_images[index] if index < len(example_images) else None
                return selected_prompt, selected_image
            return "", None
        
        example_dropdown.change(
            fn=on_example_select,
            inputs=example_dropdown,
            outputs=[prompt, input_image],
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 15px;">
            <p style="font-size: 16px; margin: 0;">The compute for this demo is generously provided by <a href="https://www.gmicloud.ai/" target="_blank">GMI Cloud</a>.  Note that this demo is meant as a preview of our distilled I2V model. Outside of few-step distillation, we have not yet fully optimized it for speed. Stay tuned for updates!</p>
        </div>
        """)
        
        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )
        
        def on_model_selection_change(selected_model):
            if not selected_model:
                selected_model = "FastWan2.1-T2V-1.3B"
            
            model_path = MODEL_PATH_MAPPING.get(selected_model)
            show_image_input = is_i2v_model(selected_model)
            
            if model_path and model_path in default_params:
                params = default_params[model_path]
                return (
                    gr.update(value=params.height),
                    gr.update(value=params.width),
                    gr.update(value=params.num_frames),
                    gr.update(value=params.guidance_scale),
                    gr.update(visible=show_image_input),
                )
            
            return (
                gr.update(value=448),
                gr.update(value=832),
                gr.update(value=20),
                gr.update(value=3.0),
                gr.update(visible=show_image_input),
            )
        
        model_selection.change(
            fn=on_model_selection_change,
            inputs=model_selection,
            outputs=[height, width, num_frames, guidance_scale, image_tab],
        )
        
        def handle_generation(*args, progress=None, request: gr.Request = None):
            model_selection, prompt, negative_prompt, use_negative_prompt, guidance_scale, num_frames, height, width, input_image = args
            
            result_path, seed_or_error, _ = generate_video(
                prompt, negative_prompt, use_negative_prompt, guidance_scale, 
                num_frames, height, width, model_selection, input_image, progress
            )
            
            if result_path and os.path.exists(result_path):
                return (
                    result_path, 
                    seed_or_error, 
                    gr.update(visible=False),
                )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),
                )
        
        run_button.click(
            fn=handle_generation,
            inputs=[
                model_selection,
                prompt,
                negative_prompt,
                use_negative_prompt,
                guidance_scale,
                num_frames,
                height,
                width,
                # randomize_seed,
                input_image,
            ],
            outputs=[result, seed_output, error_output],  # timing_display removed
            concurrency_limit=20,
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="FastVideo Gradio Frontend")
    parser.add_argument("--backend_url", type=str, default="http://localhost:8000",
                        help="URL of the Ray Serve backend")
    parser.add_argument("--t2v_model_paths", type=str,
                        default="",
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--i2v_model_paths", type=str,
                        default="FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
                        help="Comma separated list of paths to the I2V model(s)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to bind to")
    
    args = parser.parse_args()
    
    default_params = {}
    
    # Load T2V model params
    t2v_paths = [p.strip() for p in args.t2v_model_paths.split(",") if p.strip()]
    for model_path in t2v_paths:
        default_params[model_path] = SamplingParam.from_pretrained(model_path)
    
    # Load I2V model params
    i2v_paths = [p.strip() for p in args.i2v_model_paths.split(",") if p.strip()]
    for model_path in i2v_paths:
        default_params[model_path] = SamplingParam.from_pretrained(model_path)
    
    demo = create_gradio_interface(args.backend_url, default_params)
    
    print(f"Starting Gradio frontend at http://{args.host}:{args.port}")
    print(f"Backend URL: {args.backend_url}")
    print(f"T2V Models: {args.t2v_model_paths}")
    if args.i2v_model_paths:
        print(f"I2V Models: {args.i2v_model_paths}")
    
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/logo.svg")
    def get_logo():
        return FileResponse(
            "assets/logos/logo.svg", 
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    @app.get("/favicon.ico")
    def get_favicon():
        favicon_path = "assets/logos/icon_simple.svg"
        
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
    
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        base_url = str(request.base_url).rstrip('/')
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            
            <title>CausalWan</title>
            <meta name="title" content="CausalWan">
            <meta name="description" content="Make video generation go blurrrrrrr">
            <meta name="keywords" content="FastVideo, video generation, AI, machine learning, CausalWan">
            
            <meta property="og:type" content="website">
            <meta property="og:url" content="{base_url}/">
            <meta property="og:title" content="CausalWan">
            <meta property="og:description" content="Make video generation go blurrrrrrr">
            <meta property="og:image" content="{base_url}/logo.svg">
            <meta property="og:image:width" content="1200">
            <meta property="og:image:height" content="630">
            <meta property="og:site_name" content="CausalWan">
            
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="{base_url}/">
            <meta property="twitter:title" content="CausalWan">
            <meta property="twitter:description" content="Make video generation go blurrrrrrr">
            <meta property="twitter:image" content="{base_url}/logo.svg">
            <link rel="icon" type="image/png" sizes="32x32" href="/favicon.ico">
            <link rel="icon" type="image/png" sizes="16x16" href="/favicon.ico">
            <link rel="apple-touch-icon" href="/favicon.ico">
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }}
                iframe {{
                    width: 100%;
                    height: 100vh;
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
            os.path.abspath("fastvideo-logos"),
            os.path.abspath("prompts"),
            os.path.abspath("images"),
            os.path.abspath(tempfile.gettempdir()),
            os.path.abspath(os.path.join(tempfile.gettempdir(), "gradio")),
        ]
    )
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 