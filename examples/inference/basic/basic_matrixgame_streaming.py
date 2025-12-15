from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.utils import (
    collect_actions_interactively, get_default_actions, StreamingCallback
)

import torch


OUTPUT_PATH = "outputs/matrixgame_streaming"


def main():
    generator = VideoGenerator.from_pretrained(
        "/workspace/Matrix-Game-2.0-Diffusers/base_distilled_model",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        pipeline_config=MatrixGameI2V480PConfig(),
    )

    action_sequence = collect_actions_interactively()
    num_blocks = len(action_sequence)
    num_output_latents = num_blocks * 3
    num_frames = (num_output_latents - 1) * 4 + 1
    grid_sizes = torch.tensor([num_output_latents, 44, 80])

    generator.generate_video(
        prompt="",
        image_path="https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0002.png",
        mouse_cond=torch.zeros((1, num_frames, 2)),
        keyboard_cond=torch.zeros((1, num_frames, 4)),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        output_path=OUTPUT_PATH,
        save_video=True,
        streaming_action_callback=StreamingCallback(action_sequence),
    )


if __name__ == "__main__":
    main()
