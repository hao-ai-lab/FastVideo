from fastvideo import VideoGenerator
import argparse

OUTPUT_PATH = "video_samples_wan2_2_5B_ti2v"


def main(text_encoder_path: str):
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
        text_encoder_cpu_offload=False,
        # AbsMaxFP8 is the quantization method used by ComfyUI; 
        # check fastvideo/layers/quantization/* for more quantization methods
        override_text_encoder_quant="AbsMaxFP8",
        # for Wan 2.2, this is the path to "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        override_text_encoder_safetensors=text_encoder_path,
        pin_cpu_memory=True,  # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
    )

    # I2V is triggered just by passing in an image_path argument
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    video = generator.generate_video(
        prompt, output_path=OUTPUT_PATH, save_video=True, image_path=image_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        help="Path to the quantized text encoder safetensors file.",
    )
    args = parser.parse_args()
    main(args.text_encoder_path)
