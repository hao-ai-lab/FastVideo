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
        num_gpus=2,
        use_fsdp_inference=True,
        text_encoder_cpu_offload=False,
        dit_cpu_offload=False,
        num_frame_per_block=4,
    )

    sampling_param = SamplingParam.from_pretrained(model_name)
    # sampling_param.num_frames = 13
    sampling_param.num_frames = 77
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    return
    import time
    start_time = time.perf_counter()
    for i in range(10):
        prompt = (
            "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
            "wide with interest. The playful yet serene atmosphere is complemented by soft "
            "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
        )
        video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    end_time = time.perf_counter()
    print(f"Time taken to generate 10 videos: {end_time - start_time} seconds, average time per video: {(end_time - start_time) / 10} seconds")

if __name__ == "__main__":
    main()
