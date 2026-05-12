from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_svi_shot"


def main():
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        override_pipeline_cls_name="WanSVIImageToVideoPipeline",
        lora_path=("./Stable-Video-Infinity/weights/Stable-Video-Infinity/"
                   "version-1.0/svi-shot.safetensors"),
        lora_nickname="svi-shot",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,  # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
    )

    prompt = ("A sleek white motor yacht speeds across the turquoise blue sea, "
              "leaving a dramatic wake of white foam behind it under a clear blue sky.")
    image_path = "./Stable-Video-Infinity/data/toy_test/shot/frame.jpg"

    video = generator.generate_video(
        prompt,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=448,
        width=832,
        num_frames=81,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=42,
        # SVI knobs. Shot/Tom variants use num_motion_frames=1 + ref_pad_num=-1;
        # Film uses num_motion_frames=5 + ref_pad_num=0. num_clips>1 enables motion-frame chaining.
        svi_num_clips=1,
        svi_num_motion_frames=1,
        svi_ref_pad_num=-1,
    )


if __name__ == "__main__":
    main()
