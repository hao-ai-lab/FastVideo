import time
from fastvideo import VideoGenerator

def main():
    start_time = time.time()

    # config = BaseConfig.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers")
    
    gen = VideoGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        # model_path="FastVideo/FastHunyuan-diffusers",
        num_gpus=8,
        use_cpu_offload=False,
        cache_strategy="teacache",
        # flow_shift = 5
        # tp_size=2,
        # sp_size=2,
    )
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    
    gen_start_time = time.time()
    video = gen.generate_video(
        prompt="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
        height=720,
        width=1280,
        num_frames=93, # 85
        num_inference_steps=10,
        # use_cpu_offload=True,
        # num_frames=77,
        # (num_frames - 1) % temporal_compression_ratio  == 0
        # ((num_frames - 1) / temporal_compression_ratio) + 1 % num_gpus == 0
        # num_inference_steps=50,
        # fps=16,
        # guidance_scale=3.0,
        # negative_prompt="low quality, blurry, dark, out of focus",
        # guidance_scale=1,
        # height=720,
        # width=1280,
        # num_frames=125,
        # num_inference_steps=6,
        seed=1024,
        output_path="../example_output_wan_14b/"
    )
    generation_time = time.time() - gen_start_time
    print(f"Video generation time: {generation_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()