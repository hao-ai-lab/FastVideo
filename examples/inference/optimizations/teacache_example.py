import time

from fastvideo import VideoGenerator, SamplingParam


def main():
    start_time = time.perf_counter()

    gen = VideoGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        use_cpu_offload=False,
    )
    load_time = time.perf_counter() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")

    gen_start_time = time.perf_counter()

    params = SamplingParam.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    # this controls the threshold for the tea cache
    params.teacache_params.teacache_thresh = 0.08
    gen.generate_video(
        prompt=
        "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
        sampling_param=params,
        height=480,
        width=832,
        num_frames=61,  # 85 ,77 
        num_inference_steps=50,
        enable_teacache=True,
        seed=1024,
        output_path="example_outputs/")
    
    generation_time = time.perf_counter() - gen_start_time
    print(f"Video generation time: {generation_time:.2f} seconds")

    total_time = time.perf_counter() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
