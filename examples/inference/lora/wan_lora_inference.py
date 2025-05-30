from fastvideo import VideoGenerator
from fastvideo.v1.configs.sample import SamplingParam

def main():
    # Initialize VideoGenerator with the Wan model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        lora_path="benjamin-paine/steamboat-willie-1.3b"
    )

    # Create sampling parameters
    # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # sampling_param.height = 480
    # sampling_param.width = 832
    # sampling_param.num_frames = 81
    # sampling_param.guidance_scale = 5.0
    # sampling_param.num_inference_steps = 32

    # Generate video with LoRA style
    prompt = "steamboat willie style, golden era animation, an anthropomorphic cat character wearing a hat removes it and performs a courteous bow"
    video = generator.generate_video(
        prompt,
        # sampling_param=sampling_param,
        output_path="output.mp4"
    )

if __name__ == "__main__":
    main()