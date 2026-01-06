from fastvideo import VideoGenerator
import torch


OUTPUT_PATH = "finetune2_output"
def main():
    model_path = "/mnt/fast-disks/hao_lab/kaiqin/FastVideo/SkyReels-V2-I2V-1.3B-540P-Diffusers"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    num_frames = 81
    action_patterns = [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
    ]
    keyboard_cond = torch.zeros(num_frames, 3, dtype=torch.bfloat16)
    for i in range(num_frames):
        pattern = action_patterns[i % 3]
        keyboard_cond[i] = torch.tensor(pattern, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([21, 60, 104])
    image_path = "footsies-dataset/validate/episode_020_part_000_first_frame.png"

    generator.generate_video(
        prompt="",
        image_path=image_path,
        keyboard_cond=keyboard_cond.unsqueeze(0),
        grid_sizes=grid_sizes,
        num_frames=num_frames,
        height=480,
        width=832,
        num_inference_steps=50,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
