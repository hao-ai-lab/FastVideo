import os

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from fastvideo import VideoGenerator

OUTPUT_PATH = "outputs_14B"
PROMPTS_FILE = "turbodiffusion_prompts.txt"


def load_prompts(path: str) -> list[str]:
    """Load prompts from file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    # TurboDiffusion 14B: 1-4 step video generation using RCM scheduler + SLA attention
    generator = VideoGenerator.from_pretrained(
        "loayrashid/TurboWan2.1-T2V-14B-Diffusers",
        # 14B model needs more GPUs
        num_gpus=2,
        # TurboDiffusion uses a custom pipeline with RCM scheduler
        override_pipeline_cls_name="TurboDiffusionPipeline",
    )

    # Load prompts from file
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    # Generate videos for each prompt
    for i, prompt in enumerate(prompts):
        print(f"Generating video {i+1}/{len(prompts)}: {prompt[:60]}...")
        generator.generate_video(
            prompt,
            output_path=OUTPUT_PATH,
            save_video=True,
            num_inference_steps=4,  # TurboDiffusion uses 1-4 steps
            seed=42,
            guidance_scale=1.0,  # No CFG for TurboDiffusion
        )

    print(f"Done! Videos saved to {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
