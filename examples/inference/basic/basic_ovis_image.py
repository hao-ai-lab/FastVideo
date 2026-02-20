"""
Ovis-Image Text-to-Image Generation Example

This example demonstrates how to use the Ovis-Image-7B model for high-quality
text-to-image generation, especially for text rendering in images.

Ovis-Image excels at:
- Text rendering in posters, banners, logos
- UI mockups with readable text
- Infographics with correct spelling
- Bilingual text rendering
"""

from fastvideo import VideoGenerator

OUTPUT_PATH = "ovis_image_samples"


def main():
    # Load Ovis-Image model
    # Using local path to the downloaded model
    generator = VideoGenerator.from_pretrained(
        "/workspace/FastVideo/official_weights/ovis_image",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,  # Qwen3 encoder
        pin_cpu_memory=True,
    )

    # Example 1: Text rendering in a poster
    prompt1 = (
        'A creative 3D artistic render where the text "OVIS-IMAGE" is written '
        'in a bold, expressive handwritten brush style using thick, wet oil paint. '
        'The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling '
        'together like toothpaste or impasto art. You can see the ridges of the brush '
        'bristles and the glossy, wet texture of the paint. The background is a clean '
        "artist's canvas. Dynamic lighting creates soft shadows behind the floating "
        'paint strokes. Colorful, expressive, tactile texture, 4k detail.'
    )

    print(f"Generating image 1: Text rendering poster...")
    image1 = generator.generate_video(
        prompt1,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=1,  # Single image for T2I
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
    )

    # Example 2: UI mockup with text
    prompt2 = (
        'A modern mobile app interface mockup showing a weather app. '
        'At the top, display "Weather Today" in clean sans-serif font. '
        'Below show the temperature "72°F" in large numbers. '
        'Include labeled sections: "Humidity: 65%", "Wind: 12 mph", '
        'and "Forecast: Sunny". Use a gradient blue background with '
        'white text. Minimalist design, professional UI/UX, high resolution.'
    )

    print(f"Generating image 2: UI mockup...")
    image2 = generator.generate_video(
        prompt2,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=1,  # Single image for T2I
        height=1024,
        width=768,  # Portrait orientation for mobile
        num_inference_steps=50,
        guidance_scale=5.0,
    )

    # Example 3: Logo with text
    prompt3 = (
        'A professional tech startup logo featuring the text "FAST AI" '
        'in bold, modern geometric font. The letters are metallic silver '
        'with a subtle blue glow effect. Below in smaller text: '
        '"Innovation through Technology". Clean white background, '
        'minimalist design, corporate branding style, vector-like quality.'
    )

    print(f"Generating image 3: Logo design...")
    image3 = generator.generate_video(
        prompt3,
        output_path=OUTPUT_PATH,
        save_video=True,
        num_frames=1,  # Single image for T2I
        height=512,
        width=512,  # Square for logo
        num_inference_steps=50,
        guidance_scale=5.0,
    )

    print(f"\nAll images saved to {OUTPUT_PATH}/")
    print("Ovis-Image generation complete!")


if __name__ == "__main__":
    main()
