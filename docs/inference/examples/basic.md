**Source:** [examples/inference/basic](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic)

# Basic Video Generation Tutorial
The `VideoGenerator` class provides the primary Python interface for doing offline video generation, which is interacting with a diffusion pipeline without using a separate inference api server.

## Requirements
- At least a single NVIDIA GPU with CUDA 12.4.
- Python 3.10-3.12

## Installation
If you have not installed FastVideo, please following these [instructions](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html) first.

## Usage
The first script in this example shows the most basic usage of FastVideo. If you are new to Python and FastVideo, you should start here.

```bash
# if you have not cloned the directory:
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

python examples/inference/basic/basic.py
```

For an example on Apple silicon: 
```
python examples/inference/basic/basic_mps.py
```

For an example running DMD+VSA inference:
```
python examples/inference/basic/basic_dmd.py
```

## Basic Walkthrough

All you need to generate videos using multi-gpus from state-of-the-art diffusion pipelines is the following few lines!

```python
from fastvideo import VideoGenerator

def main():
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
    )

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
             "wide with interest. The playful yet serene atmosphere is complemented by soft "
             "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")
    video = generator.generate_video(prompt)

if __name__ == "__main__":
    main()
```


## Additional Files

??? note "basic.py"

    ```py
    from fastvideo import VideoGenerator
    
    # from fastvideo.configs.sample import SamplingParam
    
    OUTPUT_PATH = "video_samples"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            # FastVideo will automatically handle distributed setup
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
            # image_encoder_cpu_offload=False,
        )
    
        # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        # sampling_param.num_frames = 45
        # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        # Generate videos with the same simple API, regardless of GPU count
        prompt = (
            "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
            "wide with interest. The playful yet serene atmosphere is complemented by soft "
            "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
        )
        video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)
        # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")
    
        # Generate another video with a different prompt, without reloading the
        # model!
        prompt2 = (
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)
    
    
    if __name__ == "__main__":
        main()
    
    ```

??? note "basic_dmd.py"

    ```py
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
    
    ```

??? note "basic_mps.py"

    ```py
    from fastvideo import VideoGenerator, PipelineConfig
    from fastvideo.configs.sample import SamplingParam
    
    def main():
        config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        config.text_encoder_precisions = ["fp16"]
        
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            pipeline_config=config,
            use_fsdp_inference=False,      # Disable FSDP for MPS
            dit_cpu_offload=True,          
            text_encoder_cpu_offload=True,    
            pin_cpu_memory=True,           
            disable_autocast=False,        
            num_gpus=1,      
        )
    
        # Create sampling parameters with reduced number of frames
        sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        sampling_param.num_frames = 25  # Reduce from default 81 to 25 frames bc we have to use the SDPA attn backend for mps
        sampling_param.height = 256
        sampling_param.width = 256
    
        prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
                 "wide with interest. The playful yet serene atmosphere is complemented by soft "
                 "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")
        
        video = generator.generate_video(prompt, sampling_param=sampling_param)
    
        prompt2 = ("A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        
        video2 = generator.generate_video(prompt2, sampling_param=sampling_param)
    
    if __name__ == "__main__":
        main()
    
    ```

??? note "basic_ray.py"

    ```py
    from fastvideo import VideoGenerator
    
    # from fastvideo.configs.sample import SamplingParam
    
    OUTPUT_PATH = "video_samples"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            # FastVideo will automatically handle distributed setup
            num_gpus=2,
            use_fsdp_inference=True,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
            distributed_executor_backend="ray",
            # image_encoder_cpu_offload=False,
        )
    
        # Generate videos with the same simple API, regardless of GPU count
        prompt = (
            "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
            "wide with interest. The playful yet serene atmosphere is complemented by soft "
            "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
        )
        video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)
    
        # Generate another video with a different prompt, without reloading the
        # model!
        prompt2 = (
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)
    
    
    if __name__ == "__main__":
        main()
    
    ```

??? note "basic_self_forcing_causal.py"

    ```py
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
    
    ```

??? note "basic_wan2_2.py"

    ```py
    from fastvideo import VideoGenerator
    
    # from fastvideo.configs.sample import SamplingParam
    
    OUTPUT_PATH = "video_samples_wan2_2_14B_t2v"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            # FastVideo will automatically handle distributed setup
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=True, # DiT need to be offloaded for MoE
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
            pin_cpu_memory=True,
            # image_encoder_cpu_offload=False,
        )
    
        # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        # sampling_param.num_frames = 45
        # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        # Generate videos with the same simple API, regardless of GPU count
        prompt = (
            "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
            "wide with interest. The playful yet serene atmosphere is complemented by soft "
            "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
        )
        _ = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, height=720, width=1280, num_frames=81)
        # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")
    
        # Generate another video with a different prompt, without reloading the
        # model!
        prompt2 = (
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        _ = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True, height=720, width=1280, num_frames=81)
    
    
    if __name__ == "__main__":
        main()
    ```

??? note "basic_wan2_2_Fun.py"

    ```py
    from fastvideo import VideoGenerator
    
    # from fastvideo.configs.sample import SamplingParam
    
    OUTPUT_PATH = "video_samples_wan2_1_Fun"
    OUTPUT_NAME = "wan2.1_test"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        generator = VideoGenerator.from_pretrained(
            "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers",
            # "alibaba-pai/Wan2.2-Fun-A14B-Control",
            # FastVideo will automatically handle distributed setup
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=True, # DiT need to be offloaded for MoE
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
            pin_cpu_memory=True,
            # image_encoder_cpu_offload=False,
        )
    
        prompt = "一位年轻女性穿着一件粉色的连衣裙，裙子上有白色的装饰和粉色的纽扣。她的头发是紫色的，头上戴着一个红色的大蝴蝶结，显得非常可爱和精致。她还戴着一个红色的领结，整体造型充满了少女感和活力。她的表情温柔，双手轻轻交叉放在身前，姿态优雅。背景是简单的灰色，没有任何多余的装饰，使得人物更加突出。她的妆容清淡自然，突显了她的清新气质。整体画面给人一种甜美、梦幻的感觉，仿佛置身于童话世界中。"
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        # prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
        # negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
        image_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/8.png"
        control_video_path = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan_fun/asset_Wan2_2/v1.0/pose.mp4"
    
        video = generator.generate_video(prompt, negative_prompt=negative_prompt, image_path=image_path, video_path=control_video_path, output_path=OUTPUT_PATH, output_video_name=OUTPUT_NAME, save_video=True)
    
    if __name__ == "__main__":
        main()
    ```

??? note "basic_wan2_2_i2v.py"

    ```py
    from fastvideo import VideoGenerator
    
    # from fastvideo.configs.sample import SamplingParam
    
    OUTPUT_PATH = "video_samples_wan2_2_14B_i2v"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            # FastVideo will automatically handle distributed setup
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=True, # DiT need to be offloaded for MoE
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
            pin_cpu_memory=True,
            # image_encoder_cpu_offload=False,
        )
    
        prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
        image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    
        video = generator.generate_video(prompt, image_path=image_path, output_path=OUTPUT_PATH, save_video=True, height=832, width=480, num_frames=81)
    
    if __name__ == "__main__":
        main()
    ```

??? note "basic_wan2_2_ti2v.py"

    ```py
    from fastvideo import VideoGenerator
    
    OUTPUT_PATH = "video_samples_wan2_2_5B_ti2v"
    def main():
        # FastVideo will automatically use the optimal default arguments for the
        # model.
        # If a local path is provided, FastVideo will make a best effort
        # attempt to identify the optimal arguments.
        model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        generator = VideoGenerator.from_pretrained(
            model_name,
            # FastVideo will automatically handle distributed setup
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=True,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
            # image_encoder_cpu_offload=False,
        )
    
        # I2V is triggered just by passing in an image_path argument
        prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
        image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
        video = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, image_path=image_path)
    
        # Generate another video with a different prompt, without reloading the
        # model!
    
        # T2V mode
        prompt2 = (
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. The tall grass ripples gently in "
            "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
            "embodying the raw energy of the wild. Low angle, steady tracking shot, "
            "cinematic.")
        video2 = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)
    
    
    if __name__ == "__main__":
        main()
    ```

