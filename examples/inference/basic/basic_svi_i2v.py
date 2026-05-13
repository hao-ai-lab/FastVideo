from fastvideo import VideoGenerator

# Available variants: "shot", "film", "tom"
# Each variant uses a different LoRA + motion-frame conditioning:
#   - shot: 1 motion frame + reference-frame padding (single-prompt demo)
#   - film: 5 motion frames + zero padding (multi-prompt long-story continuity)
#   - tom:  1 motion frame + reference-frame padding (cartoon style)
MODEL_VARIANT = "shot"

VARIANT_CONFIG = {
    "shot": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-shot.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/shot/frame.jpg",
        "prompt": ("A sleek white motor yacht speeds across the turquoise blue sea, "
                   "leaving a dramatic wake of white foam behind it under a clear blue sky."),
        "num_motion_frames": 1,
        "ref_pad_num": -1,
    },
    "film": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-film.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/film/frame.jpg",
        "prompt": ("A Siamese kitten rests snugly inside a straw hat, its head slightly tilted "
                   "as it gazes curiously to the side."),
        "num_motion_frames": 5,
        "ref_pad_num": 0,
    },
    "tom": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-tom.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/tom/frame.png",
        "prompt": ("A static shot of the bright 1950s kitchen, turquoise cabinets and a chrome "
                   "sink glinting; Tom cat hovers over the counter, yellow eyes narrowed, while "
                   "Jerry mouse stands defiantly in a tiny milk puddle near a stack of purple plates."),
        "num_motion_frames": 1,
        "ref_pad_num": -1,
    },
}

OUTPUT_PATH = "video_samples_svi"


def main():
    config = VARIANT_CONFIG[MODEL_VARIANT]

    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        override_pipeline_cls_name="WanSVIImageToVideoPipeline",
        lora_path=config["lora_path"],
        lora_nickname=f"svi-{MODEL_VARIANT}",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
    )

    generator.generate_video(
        prompt=config["prompt"],
        image_path=config["image_url"],
        output_path=OUTPUT_PATH,
        save_video=True,
        height=448,
        width=832,
        num_frames=81,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=42,
        # SVI motion-frame knobs vary per variant; num_clips>1 enables motion-frame chaining.
        svi_num_clips=1,
        svi_num_motion_frames=config["num_motion_frames"],
        svi_ref_pad_num=config["ref_pad_num"],
    )


if __name__ == "__main__":
    main()
