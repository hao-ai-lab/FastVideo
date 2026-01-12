import os
from fastvideo import VideoGenerator

PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic."
)

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, "
    "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, "
    "exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted "
    "audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

# Official LTX-2 distilled defaults: num_frames = (8 x K) + 1, e.g. 121 frames at 24fps = ~5 seconds
NUM_FRAMES = 121
FPS = 24
HEIGHT = 512   # DEFAULT_1_STAGE_HEIGHT
WIDTH = 768    # DEFAULT_1_STAGE_WIDTH
SEED = 10      # DEFAULT_SEED


def main() -> None:
    diffusers_path = os.getenv("LTX2_DIFFUSERS_PATH",
                               "converted/ltx2_diffusers")

    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        ltx2_vae_tiling=True,
        ltx2_vae_spatial_tile_size_in_pixels=512,
        ltx2_vae_spatial_tile_overlap_in_pixels=64,
        ltx2_vae_temporal_tile_size_in_frames=64,
        ltx2_vae_temporal_tile_overlap_in_frames=24,
    )

    output_path = "outputs_video/ltx2_basic/backyard_drama.mp4"
    generator.generate_video(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        output_path=output_path,
        save_video=True,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        seed=SEED,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
