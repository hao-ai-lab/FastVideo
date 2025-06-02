from fastvideo import VideoGenerator
from fastvideo.v1.configs.sample import SamplingParam

def main():
    # Initialize VideoGenerator with the Wan model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        lora_path="benjamin-paine/steamboat-willie-1.3b"
    )
    
    # Generate video with LoRA style
    prompt = "steamboat willie style, golden era animation, close-up of a short fluffy monster  kneeling beside a melting red candle. the mood is one of wonder and curiosity,  as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression  convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time.  The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image."
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    video = generator.generate_video(
        prompt,
        # sampling_param=sampling_param,
        output_path="./lora_steamboat_willie",
        save_video=True,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=32
    )

if __name__ == "__main__":
    main()