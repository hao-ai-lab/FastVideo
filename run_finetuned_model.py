from fastvideo import VideoGenerator
import torch


action_patterns = [
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
]

action_map = {
    tuple(action_patterns[0]): "Left  (0)",
    tuple(action_patterns[1]): "Stop  (1)",
    tuple(action_patterns[2]): "Right (2)",
}

# sequence_indices = [0, 0, 2, 0, 0, 2, 0]
sequence_indices = [2, 2, 2, 2, 2, 2, 0]

OUTPUT_PATH = "finetune2_output"
def main():
    model_path = "/mnt/fast-disks/hao_lab/kaiqin/FastVideo/Matrix-Game-2.0-Bidirectional-Diffusers"

    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    num_frames = 81
    full_sequence = []
    for action_idx in sequence_indices:
        action = action_patterns[action_idx]
        full_sequence.extend([action] * 12)

    current_len = len(full_sequence)
    if current_len > num_frames:
        full_sequence = full_sequence[:num_frames]
    else:
        full_sequence.extend([action_patterns[1]] * (num_frames - current_len))
        
    keyboard_cond = torch.tensor(full_sequence, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([21, 60, 104])
    # image_path = "footsies-dataset/validate/0.png"
    image_path = "footsies-dataset/validate/1.png"

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
