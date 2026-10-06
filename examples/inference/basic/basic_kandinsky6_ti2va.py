import os

from fastvideo import VideoGenerator

DEFAULT_PROMPT = (
    "cinematic shot: a giant stone samurai on a stormy cliff above a neon city opens glowing golden eyes and "
    "raises a katana. Blue lightning strikes the blade, creating a massive shockwave through the clouds. The "
    "camera rapidly pulls back from a low angle. Photorealistic, epic scale, dark blue and gold lighting, rain, "
    "sparks, volumetric lightning, blockbuster quality. Audio: heavy rain, deep thunder, metallic sword hum, "
    "rising brass and choir, electrical crackle, perfectly synchronized lightning impact, sub-bass shockwave. "
    "No dialogue, text, or logos."
)

OUTPUT_PATH = "video_samples_kandinsky6_ti2va"

# The base checkpoint uses 50 steps and guidance 5.0. Set KANDINSKY6_MODEL_PATH to run the
# distilled pi-Flow checkpoint with this same example:
#   kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers
# or a local directory named ``Kandinsky-6.0-Pro-distill-5s-Diffusers``.
# That name selects the registered distilled defaults automatically: 10 steps,
# guidance 1.0, eps 1e-6, final_step_size_scale 0.5, and
# num_policy_substeps 128.
MODEL_PATH = os.environ.get("KANDINSKY6_MODEL_PATH", "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers")

IMAGE_PATH = None  # e.g. "assets/girl.png" to condition on an image


def main():
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    _ = generator.generate_video(
        DEFAULT_PROMPT,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=512,
        width=768,
        num_frames=121,
    )


if __name__ == "__main__":
    main()
