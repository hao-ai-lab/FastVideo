import os
import time
from fastvideo import VideoGenerator

from fastvideo.v1.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples_dmd2"
def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    load_start_time = time.perf_counter()
    model_name = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=True,
        # Adjust these offload parameters if you have < 32GB of VRAM
        text_encoder_offload=False,
        use_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    load_end_time = time.perf_counter()
    load_time = load_end_time - load_start_time


    sampling_param = SamplingParam.from_pretrained(model_name)

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    start_time = time.perf_counter()
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    e2e_gen_time = end_time - start_time

    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken to generate video: {gen_time} seconds")
    print(f"Time taken for e2e generation: {e2e_gen_time} seconds")


if __name__ == "__main__":
    main()
