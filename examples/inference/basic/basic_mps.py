from fastvideo import VideoGenerator, PipelineConfig


def main():
    config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config.text_encoder_precisions = ["fp16"]

    generator = VideoGenerator.from_config({
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "engine": {
            "use_fsdp_inference": False,  # Disable FSDP for MPS
            "offload": {
                "dit": True,
                "text_encoder": True,
                "pin_cpu_memory": True,
            },
            "disable_autocast": False,
            "num_gpus": 1,
        },
        "pipeline": {"experimental": {"pipeline_config": config}},
    })

    # Create sampling parameters with reduced number of frames
    sampling = {
        "num_frames": 25,  # Reduce from default 81 to 25 frames bc we have to use the SDPA attn backend for mps
        "height": 256,
        "width": 256,
    }

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
              "wide with interest. The playful yet serene atmosphere is complemented by soft "
              "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")

    video = generator.generate({"prompt": prompt, "sampling": sampling})

    prompt2 = ("A majestic lion strides across the golden savanna, its powerful frame "
               "glistening under the warm afternoon sun. The tall grass ripples gently in "
               "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
               "embodying the raw energy of the wild. Low angle, steady tracking shot, "
               "cinematic.")

    video2 = generator.generate({"prompt": prompt2, "sampling": sampling})


if __name__ == "__main__":
    main()
