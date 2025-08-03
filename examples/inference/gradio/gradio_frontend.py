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
            response = self.session.post(
                f"{self.backend_url}/generate_video",
                json=request_data,
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
    
    def get_gallery(self) -> dict:
        """Get user's video gallery"""
        try:
            response = self.session.get(f"{self.backend_url}/gallery", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}


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


def decode_and_save_video_from_frames(frames_b64: list, output_dir: str, prompt: str, fps: int = 24) -> str:
    """Decode base64 frames and save them as a video file"""
    if not frames_b64:
        return "No frames to save"
    
    # Create safe filename from prompt
    safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '_')
    video_filename = f"{safe_prompt}_frames.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    try:
        # Decode frames from base64
        decoded_frames = []
        
        for i, frame_b64 in enumerate(frames_b64):
            try:
                # Remove the data URL prefix if present
                if frame_b64.startswith('data:image/'):
                    frame_b64 = frame_b64.split(',')[1]
                
                # Decode base64 to bytes
                frame_bytes = base64.b64decode(frame_b64)
                
                # Create PIL Image from bytes
                image = Image.open(io.BytesIO(frame_bytes))
                
                # Convert PIL Image to numpy array (same format as video_generator.py)
                frame_array = np.array(image)
                decoded_frames.append(frame_array)
                
            except Exception as e:
                print(f"Warning: Failed to decode frame {i}: {e}")
                continue
        
        if not decoded_frames:
            return "Failed to decode any frames", ""
        
        # Save as video using imageio (same as video_generator.py)
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(video_path, decoded_frames, fps=fps, format="mp4")
        
        return f"Saved {len(decoded_frames)} frames as video: {video_path}", video_path
        
    except Exception as e:
        return f"Failed to save video: {str(e)}", ""


def create_gradio_interface(backend_url: str, default_params: SamplingParam):
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
        
        request_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_negative_prompt": use_negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": 20,  # Use default value
            "randomize_seed": randomize_seed,
            "return_frames": False,  # We'll get video data directly
            "image_path": None, # For T2V, we pass None as input_image
            "model_type": "i2v" if "I2V" in model_selection or "Image-to-Video" in model_selection else "t2v",  # Use model type selection
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
            # stage_names = response.get("stage_names", "").split(",")
            # stage_execution_times = [float(time) for time in response.get("stage_execution_times", "").split(",")]
            
            print(f"Used seed: {used_seed}")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Encoding time: {encoding_time:.2f}s")
            print(f"Network transfer: {network_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            # print(f"Stage names: {stage_names}")
            # print(f"Stage execution times: {stage_execution_times}")
            
            # Create detailed timing message with separate boxes
            timing_details = f"""
            <div style="margin: 20px 0;">
                <h3 style="text-align: center; margin-bottom: 15px;">⏱️ Timing Breakdown</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 15px;">
                    <div style="background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
                        <div style="font-size: 24px;">🧠</div>
                        <div style="font-weight: bold; margin: 5px 0;">Model Inference</div>
                        <div style="font-size: 18px; color: #2563eb;">{inference_time:.2f}s</div>
                    </div>
                    <div style="background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
                        <div style="font-size: 24px;">🎬</div>
                        <div style="font-weight: bold; margin: 5px 0;">Video Encoding</div>
                        <div style="font-size: 18px; color: #dc2626;">{encoding_time:.2f}s</div>
                    </div>
                    <div style="background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
                        <div style="font-size: 24px;">🌐</div>
                        <div style="font-weight: bold; margin: 5px 0;">Network Transfer</div>
                        <div style="font-size: 18px; color: #059669;">{network_time:.2f}s</div>
                    </div>
                    <div style="background: #e0f2fe; padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #0277bd;">
                        <div style="font-size: 24px;">📊</div>
                        <div style="font-weight: bold; margin: 5px 0;">Total Processing</div>
                        <div style="font-size: 20px; font-weight: bold; color: #0277bd;">{total_time:.2f}s</div>
                    </div>
                </div>"""
            
            timing_details += f"""
                <div style="margin-top: 15px;">
                    <h4 style="text-align: center; margin-bottom: 10px;">🔄 Processing Stages</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
            """
            
            # # Add individual stage timing cards
            # for stage_name, stage_time in zip(stage_names, stage_execution_times):
            #     if stage_name.strip() and stage_time > 0:  # Only show non-empty stages with valid times
            #         timing_details += f"""
            #             <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #e9ecef;">
            #                 <div style="font-weight: bold; font-size: 14px; margin-bottom: 5px;">{stage_name.strip()}</div>
            #                 <div style="font-size: 16px; color: #7c3aed; font-weight: bold;">{stage_time:.2f}s</div>
            #             </div>
            #         """
            
            # timing_details += """
            #         </div>
            #     </div>
            # """
            
            timing_details += "</div>"
            
            # Add performance insights
            if inference_time > 0:
                fps = num_frames / inference_time
                timing_details += f"""
                <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef;">
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
    examples = [
        # "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks. The cars glint in the late afternoon sun, their deep crimson and gold accents lending a touch of elegance. The tracks carve a precarious path along the cliffside, revealing glimpses of a roaring river far below. Inside, passengers peer out the large windows, their faces lit with awe as the landscape unfolds.",
        "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze.",
    ]
    
    example_labels = [
        # "Hand wrapping dough with plastic",
        # "Vintage train through mountains",
        "Crowded rooftop bar at night"
    ]

    with open("example_prompts.txt", "r") as f:
        for line in f:
            example_labels.append(line.strip()[:100])
            examples.append(line.strip())
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Image("assets/logo.jpg", show_label=False, container=False, height=100)
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <p style="font-size: 18px;"> ⚡️ Make Video Generation Go Blurrrrrrr ⚡️ </p>
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
                    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)",
                    "FastVideo/FastWan2.1-T2V-14B-Diffusers (Text-to-Video)",
                    # "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (Image-to-Video)"  # I2V functionality commented out
                ],
                value="FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)",
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
                    placeholder="Enter your prompt",
                    container=False,
                    lines=3,
                )
            with gr.Column(scale=1, min_width=120, elem_classes="center-button"):
                run_button = gr.Button("Run", variant="primary", size="lg")
        
        # Two-column layout: Advanced options on left, Video on right
        with gr.Row(equal_height=True):
            # Left column - Advanced options
            with gr.Column(scale=1):
                gr.HTML("<h3>Advanced Options</h3>")
                with gr.Group():
                    with gr.Row():
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1280,
                            step=32,
                            value=448,
                        )
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1280,
                            step=32,
                            value=832
                        )
                    
                    with gr.Row():
                        num_frames = gr.Slider(
                            label="Number of Frames",
                            minimum=16,
                            maximum=121,
                            step=16,
                            value=61,
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=12,
                            value=3.0,
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
                        value=1024
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    seed_output = gr.Number(label="Used Seed")
            
            # Right column - Video result
            with gr.Column(scale=1):
                # Add spacing to align with advanced options
                gr.HTML("<div style='height: 1.4em;'></div>")
                result = gr.Video(
                    label="Generated Video", 
                    show_label=True,
                    height=400,  # Restore original height
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
        .gr-row {
            align-items: stretch !important;
        }
        
        .gr-column {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        </style>
        """)
        
        # Progress tracking
        progress_bar = gr.Progress()
        status_updates = gr.Text(label="Status Updates", visible=False)
        
        # Centered status and timing information
        with gr.Row():
            with gr.Column():
                error_output = gr.Text(label="Error", visible=False)
                frames_output = gr.Text(label="Generation Status", visible=False)
                timing_display = gr.Markdown(label="Timing Breakdown", visible=False)
        
        # Gallery section
        with gr.Accordion("Your Video Gallery", open=False):
            gallery_refresh_btn = gr.Button("🔄 Refresh Gallery", size="sm")
            gallery_display = gr.HTML("Click 'Refresh Gallery' to see your generated videos")
            
            def refresh_gallery():
                try:
                    print("Refreshing gallery...")
                    gallery_data = client.get_gallery()
                    print(f"Gallery response: {gallery_data}")
                    
                    if gallery_data.get("success") and gallery_data.get("videos"):
                        videos = gallery_data["videos"]
                        video_count = len(videos)
                        print(f"Found {video_count} videos for this user")
                        
                        if video_count == 0:
                            return "<p style='text-align: center; color: #666;'>No videos found in your personal gallery yet. Generate some videos to see them here!</p>"
                        
                        html_content = f"<h4>Your Personal Gallery ({video_count} videos)</h4>"
                        html_content += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;'>"
                        
                        for video in videos:
                            # Create a simple video player for each video
                            # Convert absolute path to relative path for Gradio file serving
                            relative_path = os.path.relpath(video['filepath'], start='.')
                            file_size_mb = video['size'] / (1024 * 1024)
                            
                            video_html = f"""
                            <div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;'>
                                <video controls style='width: 100%; height: 200px; object-fit: cover;'>
                                    <source src="file/{relative_path}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                                <div style='margin-top: 10px;'>
                                    <p style='font-size: 12px; color: #666; margin: 5px 0;'>
                                        <strong>Prompt:</strong> {video['prompt']}
                                    </p>
                                    <p style='font-size: 10px; color: #999; margin: 5px 0;'>
                                        Generated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(video['timestamp']))} | Size: {file_size_mb:.1f} MB
                                    </p>
                                    <p style='font-size: 10px; color: #999; margin: 5px 0;'>
                                        File: {video['filename']}
                                    </p>
                                </div>
                            </div>
                            """
                            html_content += video_html
                        
                        html_content += "</div>"
                        return html_content
                    else:
                        error_msg = gallery_data.get("error", "Unknown error")
                        print(f"Gallery request failed: {error_msg}")
                        return f"<p style='text-align: center; color: #666;'>No videos found in your personal gallery yet. Generate some videos to see them here!<br/><small>({error_msg})</small></p>"
                except Exception as e:
                    print(f"Exception in refresh_gallery: {str(e)}")
                    return f"<p style='text-align: center; color: #red;'>Error loading gallery: {str(e)}</p>"
            
            gallery_refresh_btn.click(
                fn=refresh_gallery,
                outputs=gallery_display
            )
        
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
        <div style="text-align: center; margin-top: 20px;">
            <p style="font-size: 16px;">The compute for this demo is generously provided by <a href="https://www.gmicloud.ai/" target="_blank">GMI Cloud</a>. Note that this demo is meant to showcase FastWan2.1's quality and that under a large number of requests, generation speed may be affected.</p>
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
        
        def handle_generation(*args, progress=None):
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
                num_frames, height, width, randomize_seed, None, model_selection, progress
            )
            
            if result_path and os.path.exists(result_path):
                return (
                    result_path, 
                    seed_or_error, 
                    gr.update(visible=False),  # error_output
                    gr.update(visible=True, value="Generation completed successfully!"),  # frames_output
                    gr.update(visible=True, value=timing_details),  # timing_display
                )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),  # error_output
                    gr.update(visible=False),  # frames_output
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
            outputs=[result, seed_output, error_output, frames_output, timing_display],
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
    parser.add_argument("--t2v_model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the T2V model (for default parameters)")
    parser.add_argument("--t2v_14b_model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-14B-Diffusers",
                        help="Path to the T2V 14B model (for default parameters)")
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
    default_params = SamplingParam.from_pretrained(args.t2v_model_path)
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
    print(f"T2V 1.3B Model: {args.t2v_model_path}")
    print(f"T2V 14B Model: {args.t2v_14b_model_path}")
    # print(f"I2V Model: {args.i2v_model_path}") # I2V functionality commented out
    
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[os.path.abspath("outputs"), os.path.abspath("temp_images")]
    )


if __name__ == "__main__":
    main() 