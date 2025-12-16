from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.utils import create_action_presets

import random
import torch

OUTPUT_PATH = "outputs/matrixgame"
SEED = 42


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    generator = VideoGenerator.from_pretrained(
        "Skywork/Matrix-Game-2.0",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        pipeline_config=MatrixGameI2V480PConfig())

    num_frames = 597
    actions = create_action_presets(num_frames, keyboard_dim=4)
    grid_sizes = torch.tensor([150, 44, 80])

    generator.generate_video(
        prompt="",
        image_path=
        "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0002.png",
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        seed=SEED,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
