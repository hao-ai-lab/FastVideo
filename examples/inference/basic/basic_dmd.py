import os
import time
from fastvideo import VideoGenerator

from fastvideo.v1.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_dmd2"
def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    # os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    load_start_time = time.perf_counter()
    model_name = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        text_encoder_offload=False,
        use_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    load_end_time = time.perf_counter()
    load_time = load_end_time - load_start_time


    sampling_param = SamplingParam.from_pretrained(model_name)

    start_time = time.perf_counter()
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    end_time = time.perf_counter()
    gen_time1 = end_time - start_time
    e2e_gen_time = end_time - load_start_time

    # Generate another video with a different prompt, without reloading the
    # model!
    gen_time_list = []
    for i in range(10):
        start_time = time.perf_counter()
        prompt2 = (
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
        end_time = time.perf_counter()
        gen_time2 = end_time - start_time
        gen_time_list.append(gen_time2)
        # print(f"Time taken for video {i}: {gen_time2} seconds")

    print(f"Time taken for first video: {gen_time1} seconds")
    print(f"Average time taken for 10 videos: {sum(gen_time_list) / len(gen_time_list)} seconds")
    for i, gen_time in enumerate(gen_time_list):
        print(f"Time taken for video {i}: {gen_time} seconds")

    # print(f"Time taken for second video: {gen_time2} seconds")
    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken for first e2e generation: {e2e_gen_time} seconds")


if __name__ == "__main__":
    main()
