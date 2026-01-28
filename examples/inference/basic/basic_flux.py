import os

from fastvideo import VideoGenerator

PROMPT = (
    "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        num_gpus=1,
    )

    output_path = "outputs_image/flux_basic/output_flux_t2i.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
