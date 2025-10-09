
# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_ltx"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "Lightricks/LTX-Video",
        # # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False,
        # dit_cpu_offload=True,
        # vae_cpu_offload=False,
        # text_encoder_cpu_offload=True,
        # # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        # pin_cpu_memory=True,
        # # image_encoder_cpu_offload=False,
    )

    prompt = "A cute little penguin takes out a book and starts reading it"
    image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"

    video = generator.generate_video(prompt, image_path=image_path, output_path=OUTPUT_PATH, save_video=True, height=512, width=768, num_frames=20)

if __name__ == "__main__":
    main()
