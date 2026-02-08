# SPDX-License-Identifier: Apache-2.0
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

# HF model ID (base LTX-2, 40-step schedule).
MODEL_ID = "FastVideo/LTX2-Diffusers"

OUTPUT_PATH = "outputs_video/ltx2_basic/output_ltx2_base_t2v.mp4"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        MODEL_ID,
        num_gpus=8,
    )

    generator.generate_video(
        prompt=PROMPT,
        output_path=OUTPUT_PATH,
        num_inference_steps=40,
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
