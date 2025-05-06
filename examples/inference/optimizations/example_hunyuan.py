import time

from fastvideo import SamplingParam, VideoGenerator


def main():
    start_time = time.time()

    # config = BaseConfig.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers")

    gen = VideoGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        # model_path="FastVideo/FastHunyuan-diffusers",
        num_gpus=1,
        use_cpu_offload=False,
        # vae_precision="fp32",
        # vae_tiling=False,
        # flow_shift = 5
        # tp_size=2,
        # sp_size=2,
    )
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")

    gen_start_time = time.time()

    params = SamplingParam.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers", )
    params.teacache_params.teacache_thresh = 0.20
    params.teacache_params.use_ret_steps = False
    gen.generate_video(
        prompt=
        "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
        sampling_param=params,
        height=720,
        width=1280,
        num_frames=93,  # 85 ,77 
        num_inference_steps=50,
        enable_teacache=True,
        # teacache_params=WanTeaCacheParams(
        #     teacache_thresh=0.20,
        #     use_ret_steps=False,
        # ),
        # use_cpu_offload=True,
        # (num_frames - 1) % temporal_compression_ratio  == 0
        # ((num_frames - 1) / temporal_compression_ratio) + 1 % num_gpus == 0
        seed=1024,
        output_path="../example_output_wan_14b_tea_50_no_ret_steps_1GPU/")

    generation_time = time.time() - gen_start_time
    print(f"Video generation time: {generation_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
