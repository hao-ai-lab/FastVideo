import argparse
import os
import requests
import base64
import time

import gradio as gr

from fastvideo.configs.sample.base import SamplingParam


MODEL_PATH_MAPPING = {
    "FastWan2.1-T2V-1.3B": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "FastWan2.2-TI2V-5B-FullAttn": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
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
                timeout=300
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


def create_timing_display(inference_time, encoding_time, network_time, total_time, stage_execution_times, num_frames):
    dit_denoising_time = f"{stage_execution_times[5]:.2f}s" if len(stage_execution_times) > 5 else "N/A"
    
    timing_html = f"""
    <div style="margin: 10px 0;">
        <h3 style="text-align: center; margin-bottom: 10px;">⏱️ Timing Breakdown</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="timing-card timing-card-highlight">
                <div style="font-size: 20px;">🚀</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">DiT Denoising</div>
                <div style="font-size: 16px; color: #ffa200; font-weight: bold;">{dit_denoising_time}</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🧠</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">E2E (w. vae/text encoder)</div>
                <div style="font-size: 16px; color: #2563eb;">{inference_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🎬</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Video Encoding</div>
                <div style="font-size: 16px; color: #dc2626;">{encoding_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🌐</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Network Transfer</div>
                <div style="font-size: 16px; color: #059669;">{network_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">📊</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Total Processing</div>
                <div style="font-size: 18px; color: #0277bd;">{total_time:.2f}s</div>
            </div>
        </div>"""
    
    if inference_time > 0:
        fps = num_frames / inference_time
        timing_html += f"""
        <div class="performance-card" style="margin-top: 15px;">
            <span style="font-weight: bold;">Generation Speed: </span>
            <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
        </div>"""
    
    return timing_html + "</div>"


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
    
    examples, example_labels = load_from_file("prompts/prompts_final.txt")
    
    if not examples:
        examples = ["A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background."]
        example_labels = ["Crowded rooftop bar at night"]
    
    return examples, example_labels


def create_gradio_interface(backend_url: str, default_params: dict[str, SamplingParam]):
    
    client = RayServeClient(backend_url)
    
    def generate_video(
        prompt, negative_prompt, use_negative_prompt, seed, guidance_scale,
        num_frames, height, width, randomize_seed, model_selection, progress
    ):
        if not client.check_health():
            return None, f"Backend is not available. Please check if Ray Serve is running at {backend_url}", ""
        
        # Validate dimensions
        max_pixels = 720 * 1280
        if height * width > max_pixels:
            return None, f"Video dimensions too large. Maximum: 720x1280 pixels", ""
        
        if progress:
            progress(0.1, desc="Checking backend health...")
        
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
            "image_path": None,
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
            inference_time = response.get("inference_time", 0.0)
            encoding_time = response.get("encoding_time", 0.0)
            total_time = response.get("total_time", 0.0)
            network_time = response.get("network_time", 0.0)
            stage_execution_times = response.get("stage_execution_times", [])
            
            timing_details = create_timing_display(
                inference_time, encoding_time, network_time, total_time, 
                stage_execution_times, num_frames
            )
            
            if video_data:
                if progress:
                    progress(0.9, desc="Saving video...")
                
                video_path = save_video_from_base64(video_data, "outputs", prompt)
                
                if progress:
                    progress(1.0, desc="Generation complete!")
                
                if video_path and os.path.exists(video_path):
                    return video_path, used_seed, timing_details
                else:
                    return None, "Failed to save video", ""
            else:
                return None, "No video data received from backend", ""
        else:
            error_msg = response.get("error_message", "Unknown error occurred")
            return None, f"Generation failed: {error_msg}", ""
    
    examples, example_labels = load_example_prompts()
    
    theme = gr.themes.Base().set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        slider_color="#2563eb",
        checkbox_background_color_selected="#2563eb",
    )
    
    def get_default_values(model_name):
        model_path = MODEL_PATH_MAPPING.get(model_name)
        if model_path and model_path in default_params:
            params = default_params[model_path]
            return {
                'height': params.height,
                'width': params.width,
                'num_frames': params.num_frames,
                'guidance_scale': params.guidance_scale,
                'seed': params.seed,
            }
        
        return {
            'height': 448,
            'width': 832,
            'num_frames': 61,
            'guidance_scale': 3.0,
            'seed': 1024,
        }
    
    initial_values = get_default_values("FastWan2.1-T2V-1.3B")
    
    with gr.Blocks(title="FastWan", theme=theme) as demo:
        gr.Image("assets/logos/logo.svg", show_label=False, container=False, height=80)
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <p style="font-size: 18px;"> Make Video Generation Go Blurrrrrrr </p>
            <p style="font-size: 18px;"> <a href="https://github.com/hao-ai-lab/FastVideo/tree/main" target="_blank">Code</a> | <a href="https://hao-ai-lab.github.io/blogs/fastvideo_post_training/" target="_blank">Blog</a> | <a href="https://hao-ai-lab.github.io/FastVideo/" target="_blank">Docs</a>  </p>
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
                choices=list(MODEL_PATH_MAPPING.keys()),
                value="FastWan2.1-T2V-1.3B",
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
                timing_display = gr.Markdown(label="Timing Breakdown", visible=False)

        with gr.Row(equal_height=True, elem_classes="main-content-row"):
            with gr.Column(scale=1, elem_classes="advanced-options-column"):
                with gr.Group():
                    gr.HTML("<div style='margin: 0 0 15px 0; text-align: center; font-size: 16px;'>Advanced Options</div>")
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
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=12,
                            value=initial_values['guidance_scale'],
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

                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=1000000,
                        step=1,
                        value=initial_values['seed'],
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    seed_output = gr.Number(label="Used Seed")
        
            with gr.Column(scale=1, elem_classes="video-column"):
                result = gr.Video(
                    label="Generated Video", 
                    show_label=True,
                    height=466,
                    width=600,
                    container=True,
                    elem_classes="video-component"
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
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .main {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .gr-form, .gr-box, .gr-group {
            max-width: 1200px !important;
        }
        
        .gr-video {
            max-width: 500px !important;
            margin: 0 auto !important;
        }
        
        .main-content-row {
            display: flex !important;
            align-items: flex-start !important;
            min-height: 500px !important;
            gap: 20px !important;
        }
        
        .advanced-options-column,
        .video-column {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 !important;
            min-height: 400px !important;
            align-items: stretch !important;
        }
        
        .video-column > * {
            margin-top: 0 !important;
        }
        
        .video-column .gr-video,
        .video-component {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .video-column .gr-video .gr-form {
            margin-top: 0 !important;
        }
        
        .advanced-options-column .gr-group,
        .video-column .gr-video {
            margin-top: 0 !important;
            vertical-align: top !important;
        }
        
        .advanced-options-column > *:last-child,
        .video-column > *:last-child {
            flex-grow: 0 !important;
        }
        
        @media (max-width: 1400px) {
            .main-content-row {
                min-height: 600px !important;
            }
            
            .advanced-options-column,
            .video-column {
                min-height: 600px !important;
            }
        }
        
        @media (max-width: 1200px) {
            .main-content-row {
                flex-direction: column !important;
                align-items: stretch !important;
            }
            
            .advanced-options-column,
            .video-column {
                min-height: auto !important;
                width: 100% !important;
            }
        }
        
        .timing-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .timing-card-highlight {
            background: var(--background-fill-primary) !important;
            border: 2px solid var(--color-accent) !important;
        }
        
        .performance-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
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
                return examples[index]
            return ""
        
        example_dropdown.change(
            fn=on_example_select,
            inputs=example_dropdown,
            outputs=prompt,
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 15px;">
            <p style="font-size: 16px; margin: 0;">The compute for this demo is generously provided by <a href="https://www.gmicloud.ai/" target="_blank">GMI Cloud</a>. Note that this demo is meant to showcase FastWan's quality and that under a large number of requests, generation speed may be affected. We are also rate-limiting users to 3 requests per minute.</p>
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
            
            if model_path and model_path in default_params:
                params = default_params[model_path]
                return (
                    gr.update(value=params.height),
                    gr.update(value=params.width),
                    gr.update(value=params.num_frames),
                    gr.update(value=params.guidance_scale),
                    gr.update(value=params.seed),
                )
            
            return (
                gr.update(value=448),
                gr.update(value=832),
                gr.update(value=61),
                gr.update(value=3.0),
                gr.update(value=1024),
            )
        
        model_selection.change(
            fn=on_model_selection_change,
            inputs=model_selection,
            outputs=[height, width, num_frames, guidance_scale, seed],
        )
        
        def handle_generation(*args, progress=None, request: gr.Request = None):
            model_selection, prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, num_frames, height, width, randomize_seed = args
            
            result_path, seed_or_error, timing_details = generate_video(
                prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, 
                num_frames, height, width, randomize_seed, model_selection, progress
            )
            
            if result_path and os.path.exists(result_path):
                return (
                    result_path, 
                    seed_or_error, 
                    gr.update(visible=False),
                    gr.update(visible=True, value=timing_details),
                )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),
                    gr.update(visible=False),
                )
        
        run_button.click(
            fn=handle_generation,
            inputs=[
                model_selection,
                prompt,
                negative_prompt,
                use_negative_prompt,
                seed,
                guidance_scale,
                num_frames,
                height,
                width,
                randomize_seed,
            ],
            outputs=[result, seed_output, error_output, timing_display],
            concurrency_limit=20,
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="FastVideo Gradio Frontend")
    parser.add_argument("--backend_url", type=str, default="http://localhost:8000",
                        help="URL of the Ray Serve backend")
    parser.add_argument("--t2v_model_paths", type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.1-T2V-14B-Diffusers",
                        help="Comma separated list of paths to the T2V model(s)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to bind to")
    
    args = parser.parse_args()
    
    default_params = {}
    model_paths = args.t2v_model_paths.split(",")
    for model_path in model_paths:
        default_params[model_path] = SamplingParam.from_pretrained(model_path)
    
    demo = create_gradio_interface(args.backend_url, default_params)
    
    print(f"Starting Gradio frontend at http://{args.host}:{args.port}")
    print(f"Backend URL: {args.backend_url}")
    print(f"T2V Models: {args.t2v_model_paths}")
    
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
            
            <title>FastWan</title>
            <meta name="title" content="FastWan">
            <meta name="description" content="Make video generation go blurrrrrrr">
            <meta name="keywords" content="FastVideo, video generation, AI, machine learning, FastWan">
            
            <meta property="og:type" content="website">
            <meta property="og:url" content="{base_url}/">
            <meta property="og:title" content="FastWan">
            <meta property="og:description" content="Make video generation go blurrrrrrr">
            <meta property="og:image" content="{base_url}/logo.svg">
            <meta property="og:image:width" content="1200">
            <meta property="og:image:height" content="630">
            <meta property="og:site_name" content="FastWan">
            
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="{base_url}/">
            <meta property="twitter:title" content="FastWan">
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
        allowed_paths=[os.path.abspath("outputs"), os.path.abspath("fastvideo-logos")]
    )
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 