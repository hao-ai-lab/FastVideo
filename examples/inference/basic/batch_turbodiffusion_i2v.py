#!/usr/bin/env python3
"""
TurboDiffusion I2V batch inference script.

Generates videos from all images in TurboDiffusion/assets/i2v_inputs/
using the TurboDiffusion I2V pipeline with dual-model switching.
"""

import os

# Set SLA attention backend BEFORE fastvideo imports
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"

from pathlib import Path

from fastvideo import VideoGenerator

# Paths
ASSETS_DIR = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/assets/i2v_inputs"
MODEL_PATH = "loayrashid/TurboWan2.2-I2V-A14B-Diffusers"
OUTPUT_PATH = "video_samples_turbodiffusion_i2v_batch"

# Prompts from prompts.txt (paired with i2v_input_0.jpg through i2v_input_6.jpg)
PROMPTS = [
    # i2v_input_0.jpg - Surfing cat
    "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses stands on a surfboard with a neutral look when the board suddenly whips sideways, throwing cat and camera into the water; the frame dives sharply downward, swallowed by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the camera sinks.",

    # i2v_input_1.jpg - Lunar rover
    "A colorless, rugged, six-wheeled lunar rover—with exposed suspension arms, roll-cage framing, and broad low-gravity tires—glides into view from left to right, kicking up billowing plumes of moon dust that drift slowly in the vacuum. Astronauts in white spacesuits perform light, bouncing lunar strides as they hop aboard the rover's open chassis.",

    # i2v_input_2.jpg - Katana melting
    "Uma Thurman's Beatrix Kiddo holds her razor-sharp katana blade steady in the cinematic lighting. Without warning, the entire metal piece loses rigidity at once, its material trembling like unstable liquid. The surface destabilizes completely—chunks sag off in slow folds, turning into streams of molten silver that ooze downward in drops.",

    # i2v_input_3.jpg - Sailor on boat
    "Close-up on an elderly sailor in a weathered yellow raincoat, seated on the sun-lit deck of a gently rocking catamaran. With each small rise and dip of the hull, the shadows on his face shift subtly. He draws from his pipe, the ember brightening, and the exhale sends a thin ribbon of smoke that wavers and bends as the boat sways.",

    # i2v_input_4.jpg - Watercolor boat
    "Watercolor style. Wet suminagashi inks surge and spread rapidly across the paper, swirling outward as they form island-like shapes with actively shifting, bleeding edges. A tiny paper boat is pulled forward by a faster-moving stream of pigment, gliding swiftly toward the still-wet areas.",

    # i2v_input_5.jpg - Woman looking
    "She looks up, and then looks back.",

    # i2v_input_6.jpg - Tokyo rain
    "A man in a trench coat holding a black umbrella moves at a rapid, urgent pace through the streets of Tokyo on a rainy night, splashing hard through puddles. A handheld follow-cam tracks him from the side and slightly behind with quick, jittery motion, as if struggling to keep up.",
]


def main() -> None:
    print("=" * 60)
    print("TurboDiffusion I2V Batch Inference")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Initialize generator
    print(f"\nLoading model from: {MODEL_PATH}")
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=2,
        override_pipeline_cls_name="TurboDiffusionI2VPipeline",
    )

    # Get all input images
    image_paths = sorted(Path(ASSETS_DIR).glob("i2v_input_*.jpg"))
    print(f"Found {len(image_paths)} input images")

    # Generate video for each image
    for i, image_path in enumerate(image_paths):
        if i >= len(PROMPTS):
            print(f"Skipping {image_path.name}: no prompt defined")
            continue

        prompt = PROMPTS[i]
        output_name = f"turbo_i2v_{i:02d}_{image_path.stem}.mp4"

        print(f"\n[{i+1}/{len(image_paths)}] Generating from: {image_path.name}")
        print(f"  Prompt: {prompt[:80]}...")

        video = generator.generate_video(
            prompt,
            image_path=str(image_path),
            output_path=OUTPUT_PATH,
            save_video=True,
            num_inference_steps=4,
            seed=42,
            guidance_scale=1.0,
        )

        print(f"  Saved: {output_name}")

    print(f"\n{'=' * 60}")
    print(f"Batch generation complete!")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
