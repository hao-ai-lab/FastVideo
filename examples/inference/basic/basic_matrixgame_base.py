from fastvideo import VideoGenerator
from fastvideo.models.dits.matrix_game.utils import create_action_presets

import torch


OUTPUT_PATH = "video_samples_matrixgame2"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "/workspace/Matrix-Game-2.0-Base-Diffusers",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )

    num_frames = 57
    actions = create_action_presets(num_frames, keyboard_dim=6)
    grid_sizes = torch.tensor([15, 44, 80])

    generator.generate_video(
        prompt="",
        image_path="https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
