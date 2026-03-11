import torch

torch.backends.cuda.preferred_blas_library("cublaslt")

from fastvideo import VideoGenerator


PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "its eyes wide with interest. The camera slowly zooms in, capturing the "
    "raccoon's detailed fur and the golden petals swaying gently in the breeze."
)


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "KyleShao/LTX2-fp8",
        num_gpus=1,
    )

    output_path = "outputs_video/ltx2_fp8/output_ltx2_fp8_t2v.mp4"
    generator.generate_video(
        prompt=PROMPT,
        output_path=output_path,
        save_video=True,
        num_frames=121,
        height=1088,
        width=1920,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
