import torch

from fastvideo import VideoGenerator

MODEL_PATH = "FastVideo/Waypoint-1-Small-Diffusers"
KEY_FORWARD = 17
OUTPUT_PATH = "video_samples_waypoint"


def main() -> None:
    """Generate a four-second forward-moving gameplay clip."""
    num_frames = 240
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    keyboard_cond = torch.zeros((num_frames, 256))
    keyboard_cond[:, KEY_FORWARD] = 1.0
    mouse_cond = torch.zeros((num_frames, 2))
    scroll_cond = torch.zeros(num_frames)

    generator.generate_video(
        prompt="A first-person gameplay video exploring a stylized world.",
        mouse_cond=mouse_cond.unsqueeze(0),
        keyboard_cond=keyboard_cond.unsqueeze(0),
        scroll_cond=scroll_cond.unsqueeze(0),
        num_frames=num_frames,
        height=360,
        width=640,
        num_inference_steps=4,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
