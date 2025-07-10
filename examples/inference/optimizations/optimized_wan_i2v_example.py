from fastvideo import VideoGenerator
from fastvideo.v1.configs.sample import SamplingParam


OUTPUT_PATH = "./optimized_output"


def main():
    """Run WanVideo2.1 I2V pipeline with all optimizations enabled."""
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P",
        num_gpus=1,
        skip_layer_guidance=0.2,
        use_normalized_attention=True,
        use_dcm=True,
        use_taylor_seer=True,
    )

    sampling = SamplingParam.from_pretrained("Wan-AI/Wan2.1-I2V-14B-480P")

    prompt = "A lone explorer crosses a vast alien desert under twin moons"
    generator.generate_video(
        prompt,
        sampling_param=sampling,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()

