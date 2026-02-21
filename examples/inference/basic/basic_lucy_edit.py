"""
Lucy-Edit-Dev video editing example.

This script demonstrates how to use FastVideo with the Lucy-Edit-Dev model
for instruction-guided video editing. Lucy-Edit takes an input video and
a text instruction to produce an edited video.

Reference: https://huggingface.co/decart-ai/Lucy-Edit-Dev

Usage:
    python examples/inference/basic/basic_lucy_edit.py
"""

from fastvideo import VideoGenerator
from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples"


def main():
    model_id = "decart-ai/Lucy-Edit-Dev"

    generator = VideoGenerator.from_pretrained(
        model_id,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    sampling_param = SamplingParam.from_pretrained(model_id)
    sampling_param.num_frames = 81
    sampling_param.height = 480
    sampling_param.width = 832

    # Provide an input video path for editing
    # Lucy-Edit takes a video and an editing instruction prompt
    sampling_param.video_path = "path/to/your/input_video.mp4"

    # The prompt should be an editing instruction
    prompt = (
        "Change the shirt to a bright red leather jacket with a glossy finish, "
        "add aviator sunglasses, keep the same pose and background."
    )

    video = generator.generate_video(
        prompt,
        sampling_param=sampling_param,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
