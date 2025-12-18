from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.utils import create_action_presets

import torch

# Available variants: "base_distilled_model", "gta_distilled_model", "templerun_distilled_model"
# Each variant has different keyboard_dim:
#   - base_distilled_model: keyboard_dim=4
#   - gta_distilled_model: keyboard_dim=2
#   - templerun_distilled_model: keyboard_dim=7 (keyboard only, no mouse)
MODEL_VARIANT = "base_distilled_model"

# Variant-specific settings
VARIANT_CONFIG = {
    "base_distilled_model": {
        "model_path": "H1yori233/Matrix-Game-2.0-Diffusers-Base",
        "keyboard_dim": 4,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
    },
    "gta_distilled_model": {
        "model_path": "H1yori233/Matrix-Game-2.0-Diffusers-GTA",
        "keyboard_dim": 2,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/gta_drive/0000.png",
    },
    "templerun_distilled_model": {
        "model_path": "H1yori233/Matrix-Game-2.0-Diffusers-TempleRun",
        "keyboard_dim": 7,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/temple_run/0002.png",
    },
}


OUTPUT_PATH = "video_samples_matrixgame2"
def main():
    config = VARIANT_CONFIG[MODEL_VARIANT]

    generator = VideoGenerator.from_pretrained(
        config["model_path"],
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        pipeline_config=MatrixGameI2V480PConfig())

    num_frames = 597
    actions = create_action_presets(num_frames, keyboard_dim=config["keyboard_dim"])
    grid_sizes = torch.tensor([150, 44, 80])

    generator.generate_video(
        prompt="",
        image_path=config["image_url"],
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
