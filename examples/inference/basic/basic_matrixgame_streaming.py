from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.wan import MatrixGameI2V480PConfig
from fastvideo.models.dits.matrix_game.utils import (
    input_collector, StreamingCallback
)

import os
import tempfile
import threading
import torch


# Available variants: "base_distilled_model", "gta_distilled_model", "templerun_distilled_model"
# Each variant has different keyboard_dim:
#   - base_distilled_model: keyboard_dim=4
#   - gta_distilled_model: keyboard_dim=2
#   - templerun_distilled_model: keyboard_dim=7 (keyboard only, no mouse)
MODEL_VARIANT = "base_distilled_model"
MAX_BLOCKS = 50

# Variant-specific settings
VARIANT_CONFIG = {
    "base_distilled_model": {
        "keyboard_dim": 4,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
    },
    "gta_distilled_model": {
        "keyboard_dim": 2,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/gta_drive/0000.png",
    },
    "templerun_distilled_model": {
        "keyboard_dim": 7,
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/temple_run/0002.png",
    },
}


OUTPUT_PATH = "video_samples_matrixgame2_streaming"
def main():
    config = VARIANT_CONFIG[MODEL_VARIANT]
    keyboard_dim = config["keyboard_dim"]

    generator = VideoGenerator.from_pretrained(
        "Skywork/Matrix-Game-2.0",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        model_variant=MODEL_VARIANT,
        pipeline_config=MatrixGameI2V480PConfig(),
    )

    action_file = tempfile.mktemp(suffix='.json')
    request_file = action_file + '.request'
    stop_event = threading.Event()

    input_thread = threading.Thread(
        target=input_collector,
        args=(action_file, request_file, keyboard_dim, MAX_BLOCKS, stop_event),
        daemon=True
    )
    input_thread.start()

    num_output_latents = MAX_BLOCKS * 3
    num_frames = (num_output_latents - 1) * 4 + 1
    grid_sizes = torch.tensor([num_output_latents, 44, 80])

    try:
        generator.generate_video(
            prompt="",
            image_path=config["image_url"],
            mouse_cond=torch.zeros((1, num_frames, 2)),
            keyboard_cond=torch.zeros((1, num_frames, keyboard_dim)),
            grid_sizes=grid_sizes,
            num_frames=num_frames,
            height=352,
            width=640,
            num_inference_steps=50,
            output_path=OUTPUT_PATH,
            save_video=True,
            streaming_action_callback=StreamingCallback(action_file, keyboard_dim),
            allow_early_stop=True,
        )
    finally:
        stop_event.set()
        for f in [action_file, request_file]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
