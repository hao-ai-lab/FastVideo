# SPDX-License-Identifier: Apache-2.0

import os

from PIL import Image

from fastvideo import VideoGenerator

OUTPUT_PATH = "image_output"
CONDITION_IMAGE = "assets/images/couple.jpg"

PROMPT = "Change the background to a snowy mountain landscape at golden hour."


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "zai-org/GLM-Image",
        num_gpus=1,
        trust_remote_code=True,
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    condition = Image.open(CONDITION_IMAGE).convert("RGB")

    result = generator.generate_video(
        prompt=PROMPT,
        pil_image=condition,
        output_path=OUTPUT_PATH,
        save_video=False,
        return_frames=True,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=1.5,
    )

    frames = result.get("frames") if isinstance(result, dict) else None
    if frames:
        img = Image.fromarray(frames[0])
        img.save(os.path.join(OUTPUT_PATH, "edited.png"))
        print(f"Saved image to {OUTPUT_PATH}/edited.png")

    generator.shutdown()


if __name__ == "__main__":
    main()
