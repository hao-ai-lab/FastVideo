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


NUM_FRAMES = 121
FPS = 24
HEIGHT = 512 
WIDTH = 768
SEED = 10


def main() -> None:
    diffusers_path = os.getenv("LTX2_DIFFUSERS_PATH",
                               "FastVideo/LTX2-Distilled-Diffusers")

    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )

    output_path = "outputs_video/ltx2_basic/backyard_drama.mp4"
    generator.generate_video(
        prompt=PROMPT,
        # No negative_prompt for distilled model (guidance_scale=1.0 by default)
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
