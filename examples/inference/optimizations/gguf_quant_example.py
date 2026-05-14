"""Example: running inference with a GGUF-quantized text encoder.

GGUF quantization reduces the memory footprint of the text encoder by using
lower-precision weight types (Q4_0, Q4_K, Q8_0, etc.) while keeping model
quality close to the original. This is especially useful when running large
models on GPUs with limited VRAM.

Prerequisites:
    - A GGUF-quantized text encoder stored as a safetensors file.
    - vLLM installed (provides the GGML CUDA kernels needed for dequantization).

Usage:
    python examples/inference/optimizations/gguf_quant_example.py \
        --text_encoder_path /path/to/text_encoder_gguf.safetensors

    # With a custom model:
    python examples/inference/optimizations/gguf_quant_example.py \
        --text_encoder_path /path/to/text_encoder_gguf.safetensors \
        --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import argparse

from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_gguf_quant"


def main(model_path: str, text_encoder_path: str):
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        override_text_encoder_quant="gguf",
        override_text_encoder_safetensors=text_encoder_path,
        pin_cpu_memory=True,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, "
        "its eyes wide with interest. The playful yet serene atmosphere is "
        "complemented by soft natural light filtering through the petals. "
        "Mid-shot, warm and cheerful tones."
    )
    generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True)

    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently "
        "in the breeze, enhancing the lion's commanding presence. Low angle, "
        "steady tracking shot, cinematic."
    )
    generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate video with a GGUF-quantized text encoder."
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        help="Path to the GGUF-quantized text encoder safetensors file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model ID or local path to the diffusion model.",
    )
    args = parser.parse_args()
    main(args.model_path, args.text_encoder_path)
