import gradio as gr
import torch
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from fastvideo.models.mochi_genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmDiTJoint
from fastvideo.models.mochi_genmo.mochi_preview.vae.models import Decoder, Encoder
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from fastvideo.distill.solver import PCMFMScheduler
import tempfile
import os
import argparse
from safetensors.torch import load_file

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs='+', default=[])
    parser.add_argument("--num_frames", type=int, default=139)
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
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.scheduler_type == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        scheduler = PCMFMScheduler(1000, args.shift, args.num_euler_timesteps, False, args.linear_threshold, args.linear_range)
    
    mochi_genmo = False
    if mochi_genmo:
        vae_encoder_path = "/root/data/fastmochi_genmo/encoder.safetensors"
        vae_decoder_path = "/root/data/fastmochi_genmo/decoder.safetensors"
        dit_path = "/root/data/fastmochi_genmo/dit.safetensors"
        
        vae_encoder_state_dict = load_file(vae_encoder_path)
        vae_decoder_state_dict = load_file(vae_decoder_path)
        dit_state_dict = load_file(dit_path)
        
        vae_encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate",
        )
        vae_decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )
        transformer = AsymmDiTJoint()
        
        vae_encoder.load_state_dict(vae_encoder_state_dict)
        vae_decoder.load_state_dict(vae_decoder_state_dict)
        transformer.load_state_dict(dit_state_dict)
        
        transformer.config.in_channels = 12
    else:
        if args.transformer_path:
            transformer = MochiTransformer3DModel.from_pretrained(args.transformer_path)
        else:
            transformer = MochiTransformer3DModel.from_pretrained(args.model_path, subfolder='transformer/')
    
    pipe = MochiPipeline.from_pretrained(args.model_path, transformer=transformer, scheduler=scheduler)
    # from IPython import embed
    # embed()
    # del pipe.vae.encoder
    # del pipe.vae.decoder
    # pipe.vae.encoder = vae_encoder
    # pipe.vae.decoder = vae_decoder
    
    
    pipe.enable_vae_tiling()
    pipe.to(device)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    return pipe

def generate_video(prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, 
                  num_frames, height, width, num_inference_steps, randomize_seed=False):
    if randomize_seed:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    pipe = load_model(args)
    print("load model successfully")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    if not use_negative_prompt:
        negative_prompt = None
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipe(
            prompt=[prompt],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    output_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    export_to_video(output, output_path, fps=30)
    return output_path, seed

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
examples = [
    "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand’s movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough.",
    "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks. The cars glint in the late afternoon sun, their deep crimson and gold accents lending a touch of elegance. The tracks carve a precarious path along the cliffside, revealing glimpses of a roaring river far below. Inside, passengers peer out the large windows, their faces lit with awe as the landscape unfolds.",
    "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze.",
<<<<<<< Updated upstream
]

args = init_args()
pipe = load_model(args)
print("load model successfully")
with gr.Blocks() as demo:
    gr.Markdown("# Fastvideo Mochi Video Generation Demo")

=======
    "Will Smith eats noodles.",
    "A short clip of the blonde woman taking a sip from her whiskey glass, her eyes locking with the camera as she smirks playfully. The background shows a group of people laughing and enjoying the party, with vibrant neon signs illuminating the space. The    shot is taken in a way that conveys the feeling of a tipsy, carefree night out. The camera then zooms in on her face as she winks, creating a cheeky, flirtatious vibe.",
    "A superintelligent humanoid robot waking up. The robot has a sleek metallic body with futuristic design features. Its glowing red eyes are the focal point, emanating a sharp, intense light as it powers on. The scene is set in a dimly lit, high-tech laboratory filled with glowing control panels, robotic arms, and holographic screens. The setting emphasizes advanced technology and an atmosphere of mystery. The ambiance is eerie and dramatic, highlighting the moment of awakening and the robot's immense intelligence. Photorealistic style with a cinematic, dark sci-fi aesthetic. Aspect ratio: 16:9 --v 6.1"
]

args = init_args()

with gr.Blocks() as demo:
    gr.Markdown("# Mochi Video Generation Demo")
    
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row():
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    step=32,
                    value=args.height,
                )
                width = gr.Slider(
                    label="Width", minimum=256, maximum=1024, step=32, value=args.width
                )

            with gr.Row():
                num_frames = gr.Slider(
                    label="Number of Frames",
                    minimum=21,
                    maximum=163,
                    value=args.num_frames,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=12,
                    value=args.guidance_scale,
                )
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=4,
                    maximum=100,
                    value=args.num_inference_steps,
                )

            with gr.Row():
                use_negative_prompt = gr.Checkbox(
                    label="Use negative prompt", value=False
                )
=======
    
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row():
                height = gr.Slider(label="Height", minimum=256, maximum=1024, step=1, value=args.height)
                width = gr.Slider(label="Width", minimum=256, maximum=1024, step=1, value=args.width)
            
            with gr.Row():
                num_frames = gr.Slider(label="Number of Frames", minimum=8, maximum=256, step=1, value=args.num_frames)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, step=0.5, value=args.guidance_scale)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=args.num_inference_steps)
            
            with gr.Row():
                use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
>>>>>>> Stashed changes
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
<<<<<<< Updated upstream
                visible=False,
            )

            seed = gr.Slider(
                label="Seed", minimum=0, maximum=1000000, step=1, value=args.seed
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            seed_output = gr.Number(label="Used Seed")

    gr.Examples(examples=examples, inputs=prompt)

=======
                visible=False
            )
            
            seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, step=1, value=args.seed)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            seed_output = gr.Number(label="Used Seed")
    
    gr.Examples(examples=examples, inputs=prompt)
    
>>>>>>> Stashed changes
    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
    )

    run_button.click(
        fn=generate_video,
<<<<<<< Updated upstream
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
        outputs=[result, seed_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
=======
        inputs=[prompt, negative_prompt, use_negative_prompt, seed, guidance_scale, 
                num_frames, height, width, num_inference_steps, randomize_seed],
        outputs=[result, seed_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
>>>>>>> Stashed changes
