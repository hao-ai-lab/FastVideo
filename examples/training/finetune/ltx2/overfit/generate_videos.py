
# Generate videos from a JSONL prompt file for LTX2 finetuning.
import argparse
import json
import os
from pathlib import Path

from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from a JSONL prompt file."
    )
    parser.add_argument("json_file", type=str, help="Path to input JSONL file.")
    parser.add_argument(
        "start_idx", type=int, help="Start index (included) in JSONL lines."
    )
    parser.add_argument(
        "end_idx", type=int, help="End index (not included) in JSONL lines."
    )
    parser.add_argument("output_dir", type=str, help="Directory to save videos.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs. num_gpus and sp_size are both set to this.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if output video already exists.",
    )
    return parser.parse_args()


def extract_id(record: dict) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Record must contain non-empty string field 'id'.")
    return value


def extract_prompt(record: dict) -> str:
    value = record.get("video_prompt")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Record must contain non-empty string field 'video_prompt'."
        )
    return value


def generate_single_video(
    generator: VideoGenerator,
    prompt: str,
    output_path: str,
) -> None:
    generator.generate_video(
        prompt=prompt,
        output_path=output_path,
        save_video=True,
        negative_prompt="",
        num_frames=121,
        fps=24,
        height=1088,
        width=1920,
        guidance_scale=1.0,
        num_inference_steps=8,
    )


def main() -> None:
    args = parse_args()
    print(f"Args: {args}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        "FastVideo/LTX2-Distilled-Diffusers",
        num_gpus=args.num_gpus,
        sp_size=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        dit_layerwise_offload=False,
    )

    with open(args.json_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.start_idx:
                continue
            if idx >= args.end_idx:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt_id = extract_id(record)
            output_file = output_dir / f"{prompt_id}.mp4"
            if args.skip_existing and output_file.exists():
                print(f"Skipping existing video: {output_file}")
                continue
            prompt = extract_prompt(record)
            output_path = str(output_file)
            generate_single_video(
                generator=generator,
                prompt=prompt,
                output_path=output_path,
            )

    generator.shutdown()


if __name__ == "__main__":
    main()
