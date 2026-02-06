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

official_negative_prompt = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
def main() -> None:
    # Uses FastVideo default sampling settings for LTX2 base.
    generator = VideoGenerator.from_pretrained(
        "converted_weights/ltx2-base",
        num_gpus=1,
    )

    output_path = "outputs_video/ltx2_basic/output_ltx2_base_t2v_1088_1920_1.0.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        num_frames=121,
        height=1088,
        width=1920,
        fps=24,
        negative_prompt=official_negative_prompt,
        guidance_scale=3.0,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
