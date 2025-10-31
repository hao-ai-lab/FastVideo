from fastvideo import VideoGenerator
from huggingface_hub import snapshot_download

# from fastvideo.configs.sample import SamplingParam


OUTPUT_PATH = "video_samples_ltx"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    allow = [
        "model_index.json","scheduler/*","tokenizer/*","text_encoder/*","vae/*","transformer/*",
        "ltxv-13b-0.9.8-dev.safetensors",
        "ltxv-spatial-upscaler-0.9.8.safetensors",
    ]

    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"


    snapshot_download(
        "Lightricks/LTX-Video",
        local_dir="data/Lightricks/LTX-Video",
        allow_patterns=allow
        )

    generator = VideoGenerator.from_pretrained(
        model_path="data/Lightricks/LTX-Video",
        # TODO: test with >1 gpus
        num_gpus=1,
        use_fsdp_inference=False,
        pin_cpu_memory=False,
    )

    prompt = "A lone lighthouse in stormy seas, low-flying drone shot, sweeping camera arc, moody clouds, high contrast, cinematic realism"
    # TODO: when i2v pipeline is checked, add test with image_path
    image_path = None

    video = generator.generate_video(prompt, image_path=image_path, neg_prompt=negative_prompt, output_path=OUTPUT_PATH, save_video=True, num_frames=161)

if __name__ == "__main__":
    main()

