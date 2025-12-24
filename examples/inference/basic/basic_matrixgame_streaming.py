from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
from fastvideo.models.dits.matrix_game.utils import get_current_action

import os
import torch
import asyncio

# Available variants: "base_distilled_model", "gta_distilled_model", "templerun_distilled_model"
# Each variant has different keyboard_dim:
#   - base_distilled_model: keyboard_dim=4
#   - gta_distilled_model: keyboard_dim=2
#   - templerun_distilled_model: keyboard_dim=7 (keyboard only, no mouse)
MODEL_VARIANT = "base_distilled_model"

# Variant-specific settings
VARIANT_CONFIG = {
    "base_distilled_model": {
        "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
        "keyboard_dim": 4,
        "mode": "universal",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
    },
    "gta_distilled_model": {
        "model_path": "FastVideo/Matrix-Game-2.0-GTA-Diffusers",
        "keyboard_dim": 2,
        "mode": "gta_drive",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/gta_drive/0000.png",
    },
    "templerun_distilled_model": {
        "model_path": "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers",
        "keyboard_dim": 7,
        "mode": "templerun",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/temple_run/0000.png",
    },
}


OUTPUT_PATH = "video_samples_matrixgame2"

def expand_action_to_frames(action: dict, num_frames: int) -> dict:
    """Expand a single action to cover multiple frames."""
    result = {}
    for key, tensor in action.items():
        result[key] = tensor.unsqueeze(0).repeat(num_frames, 1)
    return result


async def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    config = VARIANT_CONFIG[MODEL_VARIANT]

    generator = StreamingVideoGenerator.from_pretrained(
        config["model_path"],
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

    # Streaming parameters
    num_latent_frames_per_block = 3
    max_blocks = 10
    total_latent_frames = num_latent_frames_per_block * max_blocks
    num_frames = (total_latent_frames - 1) * 4 + 1
    frames_per_block = num_latent_frames_per_block * 4
    
    actions = {
        "keyboard": torch.zeros((num_frames, config["keyboard_dim"])),
        "mouse": torch.zeros((num_frames, 2))
    }
    grid_sizes = torch.tensor([150, 44, 80])

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    video_path = os.path.join(OUTPUT_PATH, "video.mp4")
    mode = config["mode"]

    generator.reset(
        prompt="",
        image_path=config["image_url"],
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=50,
        output_path=video_path,
    )
    print("Initialization complete.")

    print(f"\n=== Block 1/{max_blocks} ===")
    action = get_current_action(mode)
    expanded = expand_action_to_frames(action, frames_per_block)
    keyboard_cond = expanded["keyboard"].unsqueeze(0)
    mouse_cond = expanded.get("mouse", torch.zeros(frames_per_block, 2).cuda()).unsqueeze(0)
    await generator.step_async(keyboard_cond, mouse_cond)

    if input("\nContinue? (y/n): ").lower() == 'n':
        generator.shutdown()
        return

    for block_idx in range(1, max_blocks):
        print(f"\n=== Block {block_idx + 1}/{max_blocks} ===")
        
        action = get_current_action(mode)
        expanded = expand_action_to_frames(action, frames_per_block)
        keyboard_cond = expanded["keyboard"].unsqueeze(0)
        mouse_cond = expanded.get("mouse", torch.zeros(frames_per_block, 2).cuda()).unsqueeze(0)
        await generator.step_async(keyboard_cond, mouse_cond)
        
        if input("\nContinue? (y/n): ").lower() == 'n':
            break

    # Save final video
    generator.finalize()
    generator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
