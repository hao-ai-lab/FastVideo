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
        try:
            response = requests.post(
                self.generate_endpoint,
                json=request_data,
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_message": f"Backend request failed: {str(e)}",
                "output_path": "",
                "seed": request_data.get("seed", 42)
            }


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
    ):
        # Check backend health first
        if not client.check_health():
            return None, f"Backend is not available. Please check if Ray Serve is running at {backend_url}", ""
        
        # Prepare request data - always request frames for video creation
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
            "return_frames": True  # Always request frames
        }
        
        # Send request to backend
        response = client.generate_video(request_data)
        
        if response.get("success", False):
            output_path = response.get("output_path", "")
            used_seed = response.get("seed", seed)
            frames_b64 = response.get("frames", [])
            
            print(f"Used seed: {used_seed}")
            print(f"Output path: {output_path}")
            
            # Handle frame extraction and video creation
            frames_status = ""
            if frames_b64:
                try:
                    # Get the output directory from the video path
                    output_dir = os.path.dirname(output_path) if output_path else "outputs"
                    frames_status, video_path = decode_and_save_video_from_frames(frames_b64, output_dir, prompt)
                    print(f"Frames: {frames_status}")
                except Exception as e:
                    frames_status = f"Failed to save frames video: {str(e)}"
                    print(f"Frame extraction error: {e}")
            else:
                frames_status = "No frames returned from backend"
            
            # Check if the video file exists
            if os.path.exists(video_path):
                return video_path, used_seed, frames_status
            else:
                return None, f"Video generated but file not found at {video_path} {frames_status}", frames_status
        else:
            error_msg = response.get("error_message", "Unknown error occurred")
            return None, f"Generation failed: {error_msg}", ""
    
    # Example prompts
    examples = [
        "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand's movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough.",
        "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks. The cars glint in the late afternoon sun, their deep crimson and gold accents lending a touch of elegance. The tracks carve a precarious path along the cliffside, revealing glimpses of a roaring river far below. Inside, passengers peer out the large windows, their faces lit with awe as the landscape unfolds.",
        "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze.",
    ]
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# FastVideo Inference Demo (Ray Serve Backend)")
        gr.Markdown(f"**Backend URL:** {backend_url}")
        
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
        
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            
            result = gr.Video(label="Result", show_label=False)
            error_output = gr.Text(label="Error", visible=False)
            frames_output = gr.Text(label="Frame Video Status", visible=False)
        
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
                        max_lines=1,
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
        
        gr.Examples(examples=examples, inputs=prompt)
        
        # Event handlers
        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )
        
        def handle_generation(*args):
            result_path, seed_or_error, frames_status = generate_video(*args)
            
            if result_path and os.path.exists(result_path):
                # Show frame status if available
                if frames_status:
                    return (
                        result_path, 
                        seed_or_error, 
                        gr.update(visible=False),  # error_output
                        gr.update(visible=True, value=frames_status)  # frames_output
                    )
                else:
                    return (
                        result_path, 
                        seed_or_error, 
                        gr.update(visible=False),  # error_output
                        gr.update(visible=False)  # frames_output
                    )
            else:
                return (
                    None, 
                    seed_or_error, 
                    gr.update(visible=True, value=seed_or_error),  # error_output
                    gr.update(visible=False)  # frames_output
                )
        
        run_button.click(
            fn=handle_generation,
            inputs=[
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
            ],
            outputs=[result, seed_output, error_output, frames_output],
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
    parser.add_argument("--model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the model (for default parameters)")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port",
                        type=int,
                        default=7860,
                        help="Port to bind to")
    
    args = parser.parse_args()
    
    # Load default parameters from the model
    # try:
    #     default_params = SamplingParam.from_pretrained(args.model_path)
    # except Exception as e:
    #     print(f"Warning: Could not load default parameters from {args.model_path}: {e}")
    #     print("Using fallback default parameters...")
    #     # Create fallback default parameters
    default_params = SamplingParam()
    default_params.height = 448
    default_params.width = 832
    default_params.num_frames = 21
    default_params.guidance_scale = 7.5
    default_params.num_inference_steps = 20
    default_params.seed = 1024
    
    # Create and launch the interface
    demo = create_gradio_interface(args.backend_url, default_params)
    
    print(f"Starting Gradio frontend at http://{args.host}:{args.port}")
    print(f"Backend URL: {args.backend_url}")
    
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[os.path.abspath("outputs")]
    )


if __name__ == "__main__":
    main() 