import os
import time
from fastvideo import VideoGenerator, SamplingParam

OUTPUT_PATH = "video_samples_causal"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        text_encoder_cpu_offload=False,
        dit_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained(model_name)

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)

if __name__ == "__main__":
    main()
