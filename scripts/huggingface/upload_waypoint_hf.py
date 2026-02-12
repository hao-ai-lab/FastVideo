from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a Waypoint diffusers-style folder to Hugging Face.")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Local diffusers-style folder to upload (model_index.json + weights).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="FastVideo/Waypoint-1-Small-Diffusers",
        help="Target HF repo id (e.g. FastVideo/Waypoint-1-Small-Diffusers).",
    )
    args = parser.parse_args()

    api = HfApi()
    api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="model",
    )


if __name__ == "__main__":
    main()


