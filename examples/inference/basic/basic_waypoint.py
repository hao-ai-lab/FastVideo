from fastvideo import VideoGenerator

import torch

# Waypoint-1-Small: 2.3B interactive world model from Overworld.
# Generates 60fps video conditioned on text + controller inputs.
MODEL_PATH = "FastVideo/Waypoint-1-Small-Diffusers"

# Owl-Control keycodes: W=17 (forward), A=30, S=31, D=32, Space=57
KEY_FORWARD = 17

OUTPUT_PATH = "video_samples_waypoint"


def main():
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    num_frames = 60
    keyboard_cond = torch.zeros((num_frames, 256))
    keyboard_cond[:, KEY_FORWARD] = 1.0
    mouse_cond = torch.zeros((num_frames, 2))

    generator.generate_video(
        prompt="A first-person gameplay video exploring a stylized world.",
        mouse_cond=mouse_cond.unsqueeze(0),
        keyboard_cond=keyboard_cond.unsqueeze(0),
        num_frames=num_frames,
        height=368,
        width=640,
        num_inference_steps=4,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
