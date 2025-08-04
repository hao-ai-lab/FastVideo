import argparse
import os
import requests
import json
import base64
import io
from typing import Optional

import gradio as gr
import torch
import imageio
from PIL import Image
import numpy as np
import time

from fastvideo.configs.sample.base import SamplingParam


class RayServeClient:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if the backend is healthy"""
        try:
            response = self.session.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_video(self, request_data: dict) -> dict:
        """Generate video using the backend API"""
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            
            response = self.session.post(
                f"{self.backend_url}/generate_video",
                json=request_data,
                headers=headers,
                timeout=300  # 5 minutes timeout
            )
            
            end_time = time.time()
            round_trip_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                # Calculate network time by subtracting backend total time
                backend_total = result.get("total_time", 0)
                network_time = round_trip_time - backend_total
                result["network_time"] = network_time
                return result
            else:
                return {"success": False, "error_message": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error_message": f"Request failed: {str(e)}"}
    



def save_video_from_base64(video_data: str, output_dir: str, prompt: str) -> str:
    """Save base64-encoded video data to a file"""
    if not video_data:
        return "No video data to save"
    
    try:
        # Remove the data URL prefix if present
        if video_data.startswith('data:video/'):
            video_data = video_data.split(',')[1]
        
        # Decode base64 to bytes
        video_bytes = base64.b64decode(video_data)
        
        # Create safe filename from prompt
        safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '_')
        video_filename = f"{safe_prompt}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save video bytes to file
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        
        return f"Saved video to: {video_path}", video_path
        
    except Exception as e:
        return f"Failed to save video: {str(e)}", ""


def create_gradio_interface(backend_url: str, default_params: dict[str, SamplingParam]):
    """Create the Gradio interface"""
    
    # Initialize the Ray Serve client
    client = RayServeClient(backend_url)
    
    def generate_video(
        prompt,
        negative_prompt,
        use_negative_prompt,
        seed,
        guidance_scale,
        num_frames,
        height,
        width,
        randomize_seed=False,
        input_image=None,
        model_selection="FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)",
        progress=None,
        request: gr.Request = None,
    ):
        # Check backend health first
        if not client.check_health():
            return None, f"Backend is not available. Please check if Ray Serve is running at {backend_url}", ""
        
        # Validate video dimensions
        max_pixels = 720 * 1280
        if height * width > max_pixels:
            return None, f"Video dimensions too large. Maximum allowed: 720x1280 pixels. Current: {height}x{width} = {height*width} pixels", ""
        
        # Update progress
        if progress:
            progress(0.1, desc="Checking backend health...")
        
        # Determine if this is I2V based on model selection - I2V functionality commented out
        # is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
        
        # Handle input image for I2V - I2V functionality commented out
        # image_path = None
        # if is_i2v and input_image is not None:
        #     if progress:
        #         progress(0.2, desc="Processing input image...")
        #     try:
        #         # Save the uploaded image to a temporary file
        #         import tempfile
        #         temp_dir = "temp_images"
        #         os.makedirs(temp_dir, exist_ok=True)
        #         
        #         # Generate a unique filename with appropriate extension
        #         import uuid
        #         # Determine the best format to preserve quality
        #         if hasattr(input_image, 'format') and input_image.format:
        #             # Use original format if available
        #             ext = input_image.format.lower()
        #             if ext == 'jpeg':
        #                 ext = 'jpg'
        #         else:
        #             # Default to PNG for lossless quality
        #             ext = 'png'
        #         
        #         image_filename = f"input_image_{uuid.uuid4().hex[:8]}.{ext}"
        #         image_path = os.path.abspath(os.path.join(temp_dir, image_filename))
        # 
        #         # Save the image preserving original quality
        #         if ext == 'png':
        #             # Use PNG for lossless compression
        #             input_image.save(image_path, "PNG", optimize=False)
        #         elif ext == 'jpg':
        #             # Use high quality JPEG with minimal compression
        #             input_image.convert("RGB").save(image_path, "JPEG", quality=95, optimize=False)
        #         else:
        #             # For other formats, save as PNG to preserve quality
        #             input_image.save(image_path, "PNG", optimize=False)
        #         
        #         print(f"Saved input image to: {image_path}")
        #     except Exception as e:
        #         print(f"Warning: Failed to save input image: {e}")
        #         image_path = None
        
        # Prepare request data
        if progress:
            progress(0.3, desc="Preparing request...")
        
        # Map clean model names to full paths
        model_path_mapping = {
            "FastWan2.1-T2V-1.3B": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            "FastWan2.1-T2V-14B": "FastVideo/FastWan2.1-T2V-14B-Diffusers",
            "FastWan2.2-TI2V-5B": "FastVideo/FastWan2.2-TI2V-5B-Diffusers"
        }
        
        request_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_negative_prompt": use_negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            # "num_inference_steps": 20,  # Use default value
            "randomize_seed": randomize_seed,
            "return_frames": False,  # We'll get video data directly
            "image_path": None, # For T2V, we pass None as input_image
            # "model_type": "i2v" if "I2V" in model_selection or "Image-to-Video" in model_selection else "t2v",  # Use model type selection
            "model_type": model_path_mapping[model_selection],
            "model_path": model_selection.split(" (")[0] if model_selection else None  # Extract model path from selection
        }
        
        # Send request to backend
        if progress:
            progress(0.4, desc="Sending request to backend...")
        
        response = client.generate_video(request_data)
        
        if progress:
            progress(0.8, desc="Processing response...")
        
        # Clean up temporary image file after processing
        # if image_path and os.path.exists(image_path):
        #     try:
        #         os.remove(image_path)
        #         print(f"Cleaned up temporary image: {image_path}")
        #     except Exception as e:
        #         print(f"Warning: Failed to clean up temporary image {image_path}: {e}")
        
        if response.get("success", False):
            video_data = response.get("video_data", "")
            used_seed = response.get("seed", seed)
            generation_time = response.get("generation_time", 0.0)
            inference_time = response.get("inference_time", 0.0)
            encoding_time = response.get("encoding_time", 0.0)
            total_time = response.get("total_time", 0.0)
            network_time = response.get("network_time", 0.0)
            stage_names = response.get("stage_names", [])
            stage_execution_times = response.get("stage_execution_times", [])
            
            print(f"Used seed: {used_seed}")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Encoding time: {encoding_time:.2f}s")
            print(f"Network transfer: {network_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print(f"Stage names: {stage_names}")
            print(f"Stage execution times: {stage_execution_times}")
            
            # Create detailed timing message with all cards in a single row
            timing_details = f"""
            <div style="margin: 10px 0;">
                <h3 style="text-align: center; margin-bottom: 10px;">⏱️ Timing Breakdown</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px;">
                    <div class="timing-card timing-card-highlight">
                        <div style="font-size: 20px;">🧠</div>
                        <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Model Inference</div>
                        <div style="font-size: 16px; color: #2563eb; font-weight: bold;">{inference_time:.2f}s</div>
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
            
            # timing_details += f"""
            #     <div style="margin-top: 15px;">
            #         <h4 style="text-align: center; margin-bottom: 10px;">🔄 Processing Stages</h4>
            #         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
            # """
            # 
            # # Add individual stage timing cards
            # for stage_name, stage_time in zip(stage_names, stage_execution_times):
            #     if stage_name.strip() and stage_time > 0:  # Only show non-empty stages with valid times
            #         timing_details += f"""
            #             <div class="stage-card">
            #                 <div style="font-weight: bold; font-size: 14px; margin-bottom: 5px;">{stage_name.strip()}</div>
            #                 <div style="font-size: 16px; color: #7c3aed; font-weight: bold;">{stage_time:.2f}s</div>
            #             </div>
            #         """
            # 
            # timing_details += """
            #         </div>
            #     </div>
            # """
            
            # Add performance insights
            if inference_time > 0:
                fps = num_frames / inference_time
                timing_details += f"""
                <div class="performance-card" style="margin-top: 15px;">
                    <span style="font-weight: bold;">Generation Speed: </span>
                    <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
                </div>
                """
            
            timing_details += "</div>"
            
            # Save video data to file for Gradio to display
            if video_data:
                try:
                    if progress:
                        progress(0.9, desc="Saving video...")
                    
                    output_dir = "outputs"
                    save_status, video_path = save_video_from_base64(video_data, output_dir, prompt)
                    print(f"Video save status: {save_status}")
                    
                    if progress:
                        progress(1.0, desc="Generation complete!")
                    
                    if video_path and os.path.exists(video_path):
                        return video_path, used_seed, timing_details
                    else:
                        return None, f"Video generated but failed to save: {save_status}", ""
                except Exception as e:
                    return None, f"Failed to save video: {str(e)}", ""
            else:
                return None, "No video data received from backend", ""
        else:
            error_msg = response.get("error_message", "Unknown error occurred")
            return None, f"Generation failed: {error_msg}", ""
    
    # Example prompts
    examples = []
    example_labels = []
    
    def contains_chinese(text):
        """Check if text contains Chinese characters"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs (Chinese characters)
                return True
        return False
    
    # Load prompts from all text files in the prompts directory
    prompts_dir = "prompts"
    if os.path.exists(prompts_dir):
        for filename in os.listdir(prompts_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(prompts_dir, filename)
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line and not contains_chinese(line):  # Skip empty lines and lines with Chinese text
                                # Create a label from the first 100 characters
                                label = line[:100] + "..." if len(line) > 100 else line
                                example_labels.append(label)
                                examples.append(line)
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    
    # Fallback to example_prompts.txt if prompts directory is empty or doesn't exist
    if not examples:
        try:
            with open("example_prompts.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        example_labels.append(line[:100])
                        examples.append(line)
        except Exception as e:
            print(f"Warning: Could not read example_prompts.txt: {e}")
            # Add a default example if all else fails
            examples = ["A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze."]
            example_labels = ["Crowded rooftop bar at night"]
    
    # Create a custom theme with blue styling to match the logo
    theme = gr.themes.Base().set(
        button_primary_background_fill="#2563eb",  # Blue color
        button_primary_background_fill_hover="#1d4ed8",  # Darker blue on hover
        button_primary_text_color="white",
        slider_color="#2563eb",  # Blue slider
        checkbox_background_color_selected="#2563eb",  # Blue checkbox when selected
    )
    
    def get_default_values_for_model(model_selection_value):
        """Get default parameter values for the specified model"""
        model_path = model_selection_value.split(" (")[0] if model_selection_value else None
        if model_path and model_path in default_params:
            params = default_params[model_path]
            return {
                'height': params.height,
                'width': params.width,
                'num_frames': params.num_frames,
                'guidance_scale': params.guidance_scale,
                'seed': params.seed,
            }
        else:
            # Fallback defaults if model not found
            return {
                'height': 448,
                'width': 832,
                'num_frames': 61,
                'guidance_scale': 3.0,
                'seed': 1024,
            }
    
    # Get initial values for the default model
    default_model = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)"
    initial_values = get_default_values_for_model(default_model)
    
    # Create Gradio interface
    with gr.Blocks(title="FastWan", theme=theme) as demo:
        
        # Logo using Gradio's Image component
        gr.Image("fastvideo-logos/main/png/full.png", show_label=False, container=False, height=80)
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <p style="font-size: 18px;"> Make Video Generation Go Blurrrrrrr </p>
            <p style="font-size: 18px;"> Twitter | <a href="https://github.com/hao-ai-lab/FastVideo/tree/main" target="_blank">Code</a> | Blog | <a href="https://hao-ai-lab.github.io/FastVideo/" target="_blank">Docs</a>  </p>
        </div>
        """)
        
        # What is FastVideo accordion
        with gr.Accordion("🎥 What Is FastVideo?", open=False):
            gr.HTML("""
            <div style="padding: 20px; line-height: 1.6;">
                <p style="font-size: 16px; margin-bottom: 15px;">
                    It features a clean, consistent API that works across popular video models, making it easier for developers to author new models and incorporate system- or kernel-level optimizations. With FastVideo's optimizations, you can achieve more than 3x inference improvement compared to other systems.
                </p>
            </div>
            """)
        
        # Backend status indicator
        # status_text = gr.Text(
        #     label="Backend Status",
        #     value="Checking backend status...",
        #     interactive=False
        # )
        
        def update_status():
            if client.check_health():
                return "✅ Backend is healthy and ready"
            else:
                return "❌ Backend is not available"
        
        # Model selection dropdown
        with gr.Row():
            model_selection = gr.Dropdown(
                choices=[
                    "FastWan2.1-T2V-1.3B",
                    "FastWan2.1-T2V-14B",
                    "FastWan2.2-TI2V-5B",
                    # "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (Image-to-Video)"  # I2V functionality commented out
                ],
                value="FastWan2.1-T2V-1.3B",
                label="Select Model",
                interactive=True
            )

        # Examples dropdown
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=example_labels,
                label="Example Prompts",
                value=None,
                interactive=True,
                allow_custom_value=False
            )
        
        # Main interface
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
        
        # Status and timing information
        with gr.Row():
            with gr.Column():
                error_output = gr.Text(label="Error", visible=False)
                # frames_output = gr.Text(label="Generation Status", visible=False)
                timing_display = gr.Markdown(label="Timing Breakdown", visible=False)

        # Two-column layout: Advanced options on left, Video on right
        with gr.Row(equal_height=True, elem_classes="main-content-row"):
            # Left column - Advanced options
            with gr.Column(scale=1, elem_classes="advanced-options-column"):
                with gr.Group():
                    gr.HTML("<div style='margin: 0 0 15px 0; text-align: center; font-size: 16px;'>Advanced Options</div>")
                    with gr.Row():
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1280,
                            step=32,
                            value=initial_values['height'],
                        )
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1280,
                            step=32,
                            value=initial_values['width']
                        )
                    
                    with gr.Row():
                        num_frames = gr.Slider(
                            label="Number of Frames",
                            minimum=16,
                            maximum=121,
                            step=16,
                            value=initial_values['num_frames'],
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
        
            # Right column - Video result
            with gr.Column(scale=1, elem_classes="video-column"):
                result = gr.Video(
                    label="Generated Video", 
                    show_label=True,
                    height=436,  # Adjusted height for better vertical alignment
                    width=600,   # Limit video width
                    container=True
                )
        
        # Add CSS to position the button and constrain width
        gr.HTML("""
        <style>
        .center-button {
            display: flex !important;
            justify-content: center !important;
            height: 100% !important;
            padding-top: 1.4em !important;
        }
        
        /* Constrain overall width */
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .main {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        /* Constrain individual components */
        .gr-form, .gr-box, .gr-group {
            max-width: 1200px !important;
        }
        
        /* Make video component smaller */
        .gr-video {
            max-width: 500px !important;
            margin: 0 auto !important;
        }
        
        /* Ensure equal height columns */
        .main-content-row {
            display: flex !important;
            align-items: flex-start !important;
            min-height: 500px !important;
        }
        
        .advanced-options-column,
        .video-column {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 !important;
            min-height: 400px !important;
        }
        
        /* Force equal heights regardless of content */
        .advanced-options-column > *:last-child,
        .video-column > *:last-child {
            flex-grow: 0 !important;
        }
        
        /* Responsive alignment for split screen */
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
        
        /* Theme-agnostic timing cards */
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
        
        .stage-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .performance-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        /* Dark mode support */
        .dark .timing-card {
            background: var(--background-fill-secondary) !important;
            border-color: var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
        }
        
        .dark .timing-card-highlight {
            background: var(--background-fill-primary) !important;
            border-color: var(--color-accent) !important;
        }
        
        .dark .stage-card,
        .dark .performance-card {
            background: var(--background-fill-secondary) !important;
            border-color: var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
        }
        </style>
        """)
        
        # Function to update prompt when example is selected
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
        
        # Disclaimer text
        gr.HTML("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 15px;">
            <p style="font-size: 16px; margin: 0;">The compute for this demo is generously provided by <a href="https://www.gmicloud.ai/" target="_blank">GMI Cloud</a>. Note that this demo is meant to showcase FastWan2.1's quality and that under a large number of requests, generation speed may be affected.</p>
        </div>
        """)
        
        # Event handlers
        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )
        
        # Model selection change handler - I2V functionality commented out
        # def on_model_selection_change(model_selection):
        #     is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
        #     prompt_placeholder = "Describe how the image should animate" if is_i2v else "Enter your prompt"
        #     return gr.update(visible=is_i2v), gr.update(placeholder=prompt_placeholder)
        # 
        # model_selection.change(
        #     fn=on_model_selection_change,
        #     inputs=model_selection,
        #     outputs=[input_image, prompt],
        # )
        
        def on_model_selection_change(selected_model):
            """Update advanced options based on selected model's default parameters"""
            if not selected_model:
                return {}, {}, {}, {}, {}  # Return empty updates if no model selected
            
            # Extract model path from selection (remove the description part)
            model_path = selected_model.split(" ")[0] if selected_model else None
            
            if model_path and model_path in default_params:
                params = default_params[model_path]
                
                # Update each component with the model's default values
                return (
                    gr.update(value=params.height),  # height
                    gr.update(value=params.width),   # width  
                    gr.update(value=params.num_frames),  # num_frames
                    gr.update(value=params.guidance_scale),  # guidance_scale
                    gr.update(value=params.seed),  # seed
                )
            else:
                # If model not found in default_params, return current values (no change)
                return {}, {}, {}, {}, {}
        
        model_selection.change(
            fn=on_model_selection_change,
            inputs=model_selection,
            outputs=[height, width, num_frames, guidance_scale, seed],
        )
        
        def handle_generation(*args, progress=None, request: gr.Request = None):
            # Extract model selection and input image from args - I2V functionality commented out
            model_selection, prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, num_frames, height, width, randomize_seed = args
            
            # Determine if this is I2V based on model selection - I2V functionality commented out
            # is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
            
            # For T2V, we pass None as input_image - I2V functionality commented out
            # if not is_i2v:
            #     input_image = None
            
            # Call the generate_video function with progress tracking
            result_path, seed_or_error, timing_details = generate_video(
                prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, 
                num_frames, height, width, randomize_seed, None, model_selection, progress, request
            )
            
            if result_path and os.path.exists(result_path):
                return (
                    result_path, 
                    seed_or_error, 
                    gr.update(visible=False),  # error_output
                    # gr.update(visible=True, value="Generation completed successfully!"),  # frames_output
                    gr.update(visible=True, value=timing_details),  # timing_display
                )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),  # error_output
                    # gr.update(visible=False),  # frames_output
                    gr.update(visible=False),  # timing_display
                )
        
        # Unified event handler
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
                # input_image, # Removed input_image from inputs
            ],
            outputs=[result, seed_output, error_output, timing_display],
            concurrency_limit=20,
        )
        
        # Update status periodically
        # demo.load(update_status, outputs=status_text)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="FastVideo Gradio Frontend")
    parser.add_argument("--backend_url",
                        type=str,
                        default="http://localhost:8000",
                        help="URL of the Ray Serve backend")
    parser.add_argument("--t2v_model_paths",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers,FastVideo/FastWan2.1-T2V-14B-Diffusers",
                        help="Comma separated list of paths to the T2V model(s)")
    # parser.add_argument("--t2v_model_replicas", type=int,
    #                     default="4,4",
    #                     help="Comma separated list of number of replicas for the T2V model(s)")
    # parser.add_argument("--i2v_model_path",  # I2V functionality commented out
    #                     type=str,
    #                     default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    #                     help="Path to the I2V model (for default parameters)")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port",
                        type=int,
                        default=7860,
                        help="Port to bind to")
    
    args = parser.parse_args()
    
    # Load default parameters from the models
    # try:
    default_params = {}
    model_paths = args.t2v_model_paths.split(",")
    for model_path in model_paths:
        default_params[model_path] = SamplingParam.from_pretrained(model_path)
    # except Exception as e:
    #     print(f"Warning: Could not load default parameters from {args.t2v_model_path}: {e}")
    #     print("Using fallback default parameters...")
    #     # Create fallback default parameters
    # default_params = SamplingParam()
    # default_params.height = 448
    # default_params.width = 832
    # default_params.num_frames = 21
    # default_params.guidance_scale = 7.5
    # default_params.num_inference_steps = 20
    # default_params.seed = 1024
    
    # Create and launch the interface
    demo = create_gradio_interface(args.backend_url, default_params)
    
    print(f"Starting Gradio frontend at http://{args.host}:{args.port}")
    print(f"Backend URL: {args.backend_url}")
    print(f"T2V Models: {args.t2v_model_paths}")
    # print(f"T2V Model Replicas: {args.t2v_model_replicas}")
    # print(f"I2V Model: {args.i2v_model_path}") # I2V functionality commented out
    
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        favicon_path="fastvideo-logos/main/png/icon-simple.png",
        allowed_paths=[os.path.abspath("outputs"), os.path.abspath("temp_images"), os.path.abspath("fastvideo-logos")]
    )


if __name__ == "__main__":
    main() 