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

from fastvideo.configs.sample.base import SamplingParam


class RayServeClient:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip('/')
        self.generate_endpoint = f"{self.backend_url}/generate_video"
        self.health_endpoint = f"{self.backend_url}/health"

        # Default request timeout in seconds. Increase if generation may run longer.
        self.request_timeout_s = int(os.getenv("FASTVIDEO_GENERATION_TIMEOUT", "900"))  # 15 minutes default
    
    def check_health(self) -> bool:
        """Check if the backend is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    

    
    def generate_video(self, request_data: dict) -> dict:
        """Send video generation request to the backend"""
        import time
        
        request_start_time = time.time()
        
        try:
            response = requests.post(
                self.generate_endpoint,
                json=request_data,
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            
            request_end_time = time.time()
            network_time = request_end_time - request_start_time
            
            response_data = response.json()
            response_data["network_time"] = network_time
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_message": f"Backend request failed: {str(e)}",
                "video_data": None,
                "seed": request_data.get("seed", 42),
                "network_time": time.time() - request_start_time
            }


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
        num_inference_steps,
        randomize_seed=False,
        input_image=None,
        model_selection="FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)",
        progress=None,
    ):
        # Check backend health first
        if not client.check_health():
            return None, f"Backend is not available. Please check if Ray Serve is running at {backend_url}", ""
        
        # Update progress
        if progress:
            progress(0.1, desc="Checking backend health...")
        
        # Determine if this is I2V based on model selection
        is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
        
        # Handle input image for I2V
        image_path = None
        if is_i2v and input_image is not None:
            if progress:
                progress(0.2, desc="Processing input image...")
            try:
                # Save the uploaded image to a temporary file
                import tempfile
                temp_dir = "temp_images"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate a unique filename with appropriate extension
                import uuid
                # Determine the best format to preserve quality
                if hasattr(input_image, 'format') and input_image.format:
                    # Use original format if available
                    ext = input_image.format.lower()
                    if ext == 'jpeg':
                        ext = 'jpg'
                else:
                    # Default to PNG for lossless quality
                    ext = 'png'
                
                image_filename = f"input_image_{uuid.uuid4().hex[:8]}.{ext}"
                image_path = os.path.abspath(os.path.join(temp_dir, image_filename))

                # Save the image preserving original quality
                if ext == 'png':
                    # Use PNG for lossless compression
                    input_image.save(image_path, "PNG", optimize=False)
                elif ext == 'jpg':
                    # Use high quality JPEG with minimal compression
                    input_image.convert("RGB").save(image_path, "JPEG", quality=95, optimize=False)
                else:
                    # For other formats, save as PNG to preserve quality
                    input_image.save(image_path, "PNG", optimize=False)
                
                print(f"Saved input image to: {image_path}")
            except Exception as e:
                print(f"Warning: Failed to save input image: {e}")
                image_path = None
        
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
            "num_inference_steps": num_inference_steps,
            "randomize_seed": randomize_seed,
            "return_frames": False,  # We'll get video data directly
            "image_path": image_path,
            "model_type": "i2v" if is_i2v else "t2v",  # Use model type selection
            "model_path": model_selection.split(" (")[0] if model_selection else None  # Extract model path from selection
        }
        
        # Send request to backend
        if progress:
            progress(0.4, desc="Sending request to backend...")
        
        response = client.generate_video(request_data)
        
        if progress:
            progress(0.8, desc="Processing response...")
        
        # Clean up temporary image file after processing
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Cleaned up temporary image: {image_path}")
            except Exception as e:
                print(f"Warning: Failed to clean up temporary image {image_path}: {e}")
        
        if response.get("success", False):
            video_data = response.get("video_data", "")
            used_seed = response.get("seed", seed)
            generation_time = response.get("generation_time", 0.0)
            inference_time = response.get("inference_time", 0.0)
            encoding_time = response.get("encoding_time", 0.0)
            total_time = response.get("total_time", 0.0)
            network_time = response.get("network_time", 0.0)
            
            print(f"Used seed: {used_seed}")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Encoding time: {encoding_time:.2f}s")
            print(f"Network transfer: {network_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            
            # Create detailed timing message
            timing_details = f"⏱️ **Timing Breakdown**\n\n"
            timing_details += f"| Phase | Time |\n"
            timing_details += f"|-------|------|\n"
            timing_details += f"| 🧠 Model Inference | {inference_time:.2f}s |\n"
            timing_details += f"| 🎬 Video Encoding | {encoding_time:.2f}s |\n"
            timing_details += f"| 🌐 Network Transfer | {network_time:.2f}s |\n"
            timing_details += f"| **📊 Total Processing** | **{total_time:.2f}s** |\n\n"
            
            # Add performance insights
            if inference_time > 0:
                fps = num_frames / inference_time
                timing_details += f"**Performance Metrics:**\n"
                timing_details += f"• Generation Speed: {fps:.1f} frames/second\n"
                timing_details += f"• Model Efficiency: {inference_time/total_time*100:.1f}% of total time\n"
            
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
        "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand's movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough.",
        "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks. The cars glint in the late afternoon sun, their deep crimson and gold accents lending a touch of elegance. The tracks carve a precarious path along the cliffside, revealing glimpses of a roaring river far below. Inside, passengers peer out the large windows, their faces lit with awe as the landscape unfolds.",
        "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze.",
    ]
    
    example_labels = [
        "Hand wrapping dough with plastic",
        "Vintage train through mountains", 
        "Crowded rooftop bar at night"
    ]
    

    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# FastVideo Inference Demo (Ray Serve Backend)")
        
        # Backend status indicator
        status_text = gr.Text(
            label="Backend Status",
            value="Checking backend status...",
            interactive=False
        )
        
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
                    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (Image-to-Video)"
                ],
                value="FastVideo/FastWan2.1-T2V-1.3B-Diffusers (Text-to-Video)",
                label="Select Model",
                interactive=True
            )
        
        # Main interface
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=3,
                    placeholder="Enter your prompt",
                    container=False,
                    lines=3,
                    scale=4,
                )
                run_button = gr.Button("Run", scale=1)
            
            # Input image (only visible for I2V models)
            input_image = gr.Image(
                label="Input Image (required for Image-to-Video models)",
                type="pil",
                show_label=True,
                container=True,
                visible=False,
            )
            
            # Results
            result = gr.Video(label="Result", show_label=False)
            error_output = gr.Text(label="Error", visible=False)
            frames_output = gr.Text(label="Generation Status", visible=False)
            timing_display = gr.Markdown(label="Timing Breakdown", visible=False)
            download_file = gr.File(visible=False)
            
            # Progress tracking
            progress_bar = gr.Progress()
            status_updates = gr.Text(label="Status Updates", visible=False)
        
        # Examples dropdown
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=example_labels,
                label="Example Prompts",
                value=None,
                interactive=True,
                allow_custom_value=False
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
        
        # Shared advanced options
        with gr.Accordion("Advanced options", open=False):
            with gr.Group():
                with gr.Row():
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=448,
                    )
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=832
                    )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=16,
                        maximum=160,
                        step=16,
                        value=61,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=12,
                        value=3.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=3,
                        maximum=100,
                        value=3,
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
        
        # Event handlers
        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )
        
        # Model selection change handler
        def on_model_selection_change(model_selection):
            is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
            prompt_placeholder = "Describe how the image should animate" if is_i2v else "Enter your prompt"
            return gr.update(visible=is_i2v), gr.update(placeholder=prompt_placeholder)
        
        model_selection.change(
            fn=on_model_selection_change,
            inputs=model_selection,
            outputs=[input_image, prompt],
        )
        
        def handle_generation(*args, progress=None):
            # Extract model selection and input image from args
            model_selection, prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, num_frames, height, width, num_inference_steps, randomize_seed, input_image = args
            
            # Determine if this is I2V based on model selection
            is_i2v = "I2V" in model_selection or "Image-to-Video" in model_selection
            
            # For T2V, we pass None as input_image
            if not is_i2v:
                input_image = None
            
            # Call the generate_video function with progress tracking
            result_path, seed_or_error, timing_details = generate_video(
                prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, 
                num_frames, height, width, num_inference_steps, randomize_seed, input_image, model_selection, progress
            )
            
            if result_path and os.path.exists(result_path):
                return (
                    result_path, 
                    seed_or_error, 
                    gr.update(visible=False),  # error_output
                    gr.update(visible=True, value="Generation completed successfully!"),  # frames_output
                    gr.update(visible=True, value=timing_details),  # timing_display
                    gr.update(visible=True, value=result_path),  # download_file
                )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),  # error_output
                    gr.update(visible=False),  # frames_output
                    gr.update(visible=False),  # timing_display
                    gr.update(visible=False),  # download_file
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
                num_inference_steps,
                randomize_seed,
                input_image,
            ],
            outputs=[result, seed_output, error_output, frames_output, timing_display, download_file],
            concurrency_limit=20,
        )
        
        # Update status periodically
        demo.load(update_status, outputs=status_text)
    
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
    parser.add_argument("--i2v_model_path",
                        type=str,
                        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                        help="Path to the I2V model (for default parameters)")
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
    print(f"I2V Model: {args.i2v_model_path}")
    
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[os.path.abspath("outputs"), os.path.abspath("temp_images")]
    )


if __name__ == "__main__":
    main() 