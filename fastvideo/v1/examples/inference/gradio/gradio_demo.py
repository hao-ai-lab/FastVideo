import os
import gradio as gr
import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo import VideoGenerator


if __name__ == "__main__":
    args = FastVideoArgs(model_path="FastVideo/FastHunyuan-Diffusers", num_gpus=2)

    generator = VideoGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus
    )

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
        if randomize_seed:
            seed = torch.randint(0, 1000000, (1, )).item()

        if not use_negative_prompt:
            negative_prompt = None

        generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=seed
        )

        output_path = os.path.join(args.output_path, f"{prompt[:100]}.mp4")

        return output_path, seed

    examples = [
    "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand’s movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough.",
    "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks. The cars glint in the late afternoon sun, their deep crimson and gold accents lending a touch of elegance. The tracks carve a precarious path along the cliffside, revealing glimpses of a roaring river far below. Inside, passengers peer out the large windows, their faces lit with awe as the landscape unfolds.",
    "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background. Strings of fairy lights hang above, casting a warm, golden glow over the scene. Groups of people gather around high tables, their laughter blending with the soft rhythm of live jazz. The aroma of freshly mixed cocktails and charred appetizers wafts through the air, mingling with the cool night breeze.",
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# FastVideo Inference Demo")

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
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, step=32, value=args.width)

                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=21,
                        maximum=163,
                        value=45,
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
                        value=6,
                    )

                with gr.Row():
                    use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                    visible=False,
                )

                seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, step=1, value=args.seed)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                seed_output = gr.Number(label="Used Seed")

        gr.Examples(examples=examples, inputs=prompt)

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )

        run_button.click(
            fn=generate_video,
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

    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860)
