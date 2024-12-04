import gradio as gr
import torch
from fastvideo.model.pipeline_mochi import MochiPipeline
from fastvideo.model.modeling_mochi import MochiTransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from fastvideo.distill.solver import PCMFMScheduler
import tempfile
import os
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs='+', default=[])
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--scheduler_type", type=str, default="euler")
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument("--linear_threshold", type=float, default=0.025)
    parser.add_argument("--linear_range", type=float, default=0.5)
    return parser.parse_args()

args = init_args()

examples = examples = args.prompts if args.prompts else [
    "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand’s movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough."
]

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        scheduler = PCMFMScheduler(1000, args.shift, args.num_euler_timesteps, False, args.linear_threshold, args.linear_range)
    
    if args.transformer_path:
        transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path)
    else:
        transformer = MochiTransformer3DModel.from_pretrained(args.model_path, subfolder='transformer/')
    
    pipe = MochiPipeline.from_pretrained(args.model_path, transformer=transformer, scheduler=scheduler)
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload(device)
    return pipe

def generate_video(prompt, guidance_scale, num_frames, height, width, num_inference_steps, seed, randomize_seed=False):
    if randomize_seed:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    pipe = load_model(args)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipe(
            prompt=[prompt],
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    
    output_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    export_to_video(output, output_path, fps=30)
    return output_path, seed

with gr.Blocks() as demo:
    gr.Markdown("# Mochi Video Generation Demo")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            with gr.Row():
                num_frames = gr.Slider(minimum=8, maximum=256, value=args.num_frames, step=1, label="Number of Frames")
                guidance_scale = gr.Slider(minimum=1, maximum=20, value=args.guidance_scale, step=0.5, label="Guidance Scale")
            with gr.Row():
                height = gr.Slider(minimum=256, maximum=1024, value=args.height, step=64, label="Height")
                width = gr.Slider(minimum=256, maximum=1024, value=args.width, step=64, label="Width")
            with gr.Row():
                num_inference_steps = gr.Slider(minimum=10, maximum=100, value=args.num_inference_steps, step=1, label="Inference Steps")
                seed = gr.Slider(minimum=0, maximum=1000000, value=args.seed, step=1, label="Seed")
            
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            result = gr.Video(label="Generated Video")
            seed_output = gr.Number(label="Used Seed")
    
    gr.Examples(examples=examples, inputs=prompt)
    
    generate_btn.click(
        fn=generate_video,
        inputs=[prompt, guidance_scale, num_frames, height, width,
                num_inference_steps, seed, randomize_seed],
        outputs=[result, seed_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )