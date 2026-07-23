from fastvideo import VideoGenerator

MODEL_VARIANT = "shot"

VARIANT_CONFIG = {
    "shot": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-shot.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/shot/frame.jpg",
        "prompts": [
            ("A sleek white motor yacht speeds across the turquoise blue sea, "
             "leaving a dramatic wake of white foam behind it under a clear blue sky."),
        ],
        "num_clips": 2,
        "num_motion_frames": 1,
        "ref_pad_num": -1,
        "height": 448,
    },
    "film": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-film-opt-10212025.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/film/frame.jpg",
        "prompts": [
            ("A Siamese kitten rests snugly inside a straw hat, its head slightly tilted "
             "as it gazes curiously to the side."),
            ("The Siamese kitten decides to explore the room and jumps out of the hat "
             "onto the soft carpet below."),
        ],
        "num_clips": 2,
        "num_motion_frames": 5,
        "ref_pad_num": 0,
        "height": 480,
    },
    "tom": {
        "lora_path": "vita-video-gen/svi-model/version-1.0/svi-tom.safetensors",
        "image_url":
        "https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/data/toy_test/tom/frame.png",
        "prompts": [
            ("A static shot of the bright 1950s kitchen, turquoise cabinets and a chrome "
             "sink glinting; Tom cat hovers over the counter, yellow eyes narrowed, while "
             "Jerry mouse stands defiantly in a tiny milk puddle near a stack of purple plates."),
            ("Close-up on Tom cat’s face: a wicked smirk creases his white muzzle; his black "
             "brows angle into a sharp V as he crooks one claw toward Jerry mouse like a "
             "menacing metronome."),
        ],
        "num_clips": 2,
        "num_motion_frames": 1,
        "ref_pad_num": 0,
        "height": 560,
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
        flow_shift=5.0,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
    )

    generator.generate_video(
        prompt=config["prompts"][0],
        image_path=config["image_url"],
        output_path=OUTPUT_PATH,
        save_video=True,
        height=config["height"],
        width=832,
        num_frames=81,
        fps=24,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=0,
        svi_num_clips=config["num_clips"],
        svi_clip_prompts=None if MODEL_VARIANT == "shot" else config["prompts"],
        svi_num_motion_frames=config["num_motion_frames"],
        svi_seed_stride=42,
        svi_ref_pad_num=config["ref_pad_num"],
    )


if __name__ == "__main__":
    main()
