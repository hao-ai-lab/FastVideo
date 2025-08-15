import argparse
import os
from copy import deepcopy

import gradio as gr
import torch

from fastvideo import VideoGenerator
from fastvideo.configs.sample.base import SamplingParam

if __name__ == "__main__":
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    parser = argparse.ArgumentParser(description="FastVideo Gradio Demo")
    parser.add_argument("--model_path",
                        type=str,
                        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
                        help="Path to the model")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--output_path",
                        type=str,
                        default="my_videos/",
                        help="Path to save generated videos")
    parsed_args = parser.parse_args()


    generator = VideoGenerator.from_pretrained(
        model_path=parsed_args.model_path, num_gpus=parsed_args.num_gpus)

    default_params = SamplingParam.from_pretrained(parsed_args.model_path)

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
        params = deepcopy(default_params)
        params.prompt = prompt
        params.negative_prompt = negative_prompt
        params.seed = seed
        params.guidance_scale = guidance_scale
        params.num_frames = num_frames
        params.height = height
        params.width = width
        params.num_inference_steps = num_inference_steps

        if randomize_seed:
            params.seed = torch.randint(0, 1000000, (1, )).item()

        if use_negative_prompt and negative_prompt:
            params.negative_prompt = negative_prompt
        else:
            params.negative_prompt = default_params.negative_prompt

        generator.generate_video(prompt=prompt, sampling_param=params,output_path=parsed_args.output_path,save_video=True)

        output_path = os.path.join(parsed_args.output_path,
                                   f"{params.prompt[:100]}.mp4")

        return output_path, params.seed

    examples = [
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest.",
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
                        value=default_params.height,
                    )
                    width = gr.Slider(label="Width",
                                      minimum=256,
                                      maximum=1024,
                                      step=32,
                                      value=default_params.width)

                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=21,
                        maximum=163,
                        value=default_params.num_frames,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=12,
                        value=default_params.guidance_scale,
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=3,
                        maximum=100,
                        value=default_params.num_inference_steps,
                    )

                with gr.Row():
                    use_negative_prompt = gr.Checkbox(
                        label="Use negative prompt", value=False)
                    negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=1,
                        placeholder="Enter your negative prompt",
                        visible=False,
                    )

                seed = gr.Slider(label="Seed",
                                 minimum=0,
                                 maximum=1000000,
                                 step=1,
                                 value=default_params.seed)
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

    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=8888)
