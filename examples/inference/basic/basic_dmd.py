import os
import time
from fastvideo import VideoGenerator

from fastvideo.configs.sample import SamplingParam

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
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    load_end_time = time.perf_counter()
    load_time = load_end_time - load_start_time


    sampling_param = SamplingParam.from_pretrained(model_name)

    prompt = (
        "A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. The puddles reflect glowing signs in kanji, advertising ramen, karaoke, and VR arcades. A woman in a translucent raincoat walks briskly with an LED umbrella. Steam rises from a street food cart, and a cat darts across the screen. Raindrops are visible on the camera lens, creating a cinematic bokeh effect."
    )
    start_time = time.perf_counter()
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, sampling_param=sampling_param)
    end_time = time.perf_counter()
    gen_time = end_time - start_time

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    start_time = time.perf_counter()
    video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)
    end_time = time.perf_counter()
    gen_time2 = end_time - start_time


    print(f"Time taken to load model: {load_time} seconds")
    print(f"Time taken to generate video: {gen_time} seconds")
    print(f"Time taken to generate video2: {gen_time2} seconds")


if __name__ == "__main__":
    main()
