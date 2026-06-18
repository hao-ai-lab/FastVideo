"""v2 port of basic_wan2_2_ti2v.py — Wan2.2-TI2V-5B (T2V mode) through the v2 VideoGenerator.

Same convenience API as upstream (from_pretrained + generate_video); only delta is importing
VideoGenerator from v2. Wan2.2-TI2V-5B reuses the Wan adapter classes (WanTransformer3DModel /
AutoencoderKLWan / UMT5) with the higher-compression VAE geometry (z_dim=48, 16x spatial, 4x temporal).

NOTE: upstream also runs I2V (image_path=...). The v2 program here is T2V-only (image conditioning is
not yet ported), so this mirrors the upstream *T2V* branch (prompt2). Modest res/frames for a quick run.
"""
from v2 import VideoGenerator

OUTPUT_PATH = "v2_video_samples_wan2_2_5B_ti2v"


def main() -> None:
    model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )

    # T2V mode (the v2 program is text-to-video; upstream's image_path I2V branch is not ported yet).
    prompt = ("A majestic lion strides across the golden savanna, its powerful frame glistening under "
              "the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's "
              "commanding presence. Low angle, steady tracking shot, cinematic.")
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, output_video_name="wan22_ti2v_lion",
                                     save_video=True, num_frames=25, height=448, width=768,
                                     num_inference_steps=20, guidance_scale=5.0)
    print(f"Output: {video.video_path}")


if __name__ == "__main__":
    main()
