from fastvideo import VideoGenerator

OUTPUT_PATH = "./multi_lora"


def main():
    # Create a generator for WanVideo2.1 I2V
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P",
        num_gpus=1,
    )

    # Load three LoRA adapters into the pipeline
    generator.set_lora_adapter("lora1", "path/to/first_lora")
    generator.set_lora_adapter("lora2", "path/to/second_lora")
    generator.set_lora_adapter("lora3", "path/to/third_lora")

    # The last call activates lora3. Generate a video with it
    prompt = "An astronaut explores a strange new world, cinematic scene"
    generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)

    # Switch to lora1 and generate another video
    generator.set_lora_adapter("lora1")
    generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)

    # Switch to lora2 and generate one more video
    generator.set_lora_adapter("lora2")
    generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    main()
