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


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "FastVideo/LTX2-Distilled-Diffusers",
        num_gpus=1,
    )

    output_path = "outputs_video/ltx2_basic/output_ltx2_distilled_t2v.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
