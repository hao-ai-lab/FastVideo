# SPDX-License-Identifier: Apache-2.0
"""Build the FastVideo Waypoint bundle from pinned upstream snapshots."""

import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

WAYPOINT_REPO = "Overworld/Waypoint-1-Small"
WAYPOINT_REVISION = "93fe14eed217bb09c0244a9d91b2b3e88c3de181"
TEXT_ENCODER_REPO = "google/umt5-xl"
TEXT_ENCODER_REVISION = "e02ba7fcf3d6286215043111ee2fc83a9d1f18e2"

MODEL_INDEX = {
    "_class_name": "WaypointPipeline",
    "_diffusers_version": "0.36.0",
    "transformer": ["diffusers", "WorldModel"],
    "vae": ["diffusers", "WorldEngineVAE"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
}


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, dirs_exist_ok=True)


def build_bundle(output_dir: Path) -> None:
    waypoint = Path(
        snapshot_download(
            repo_id=WAYPOINT_REPO,
            revision=WAYPOINT_REVISION,
            allow_patterns=["transformer/**", "vae/**"],
        )
    )
    text_encoder = Path(
        snapshot_download(
            repo_id=TEXT_ENCODER_REPO,
            revision=TEXT_ENCODER_REVISION,
            allow_patterns=[
                "config.json",
                "pytorch_model*.bin*",
                "model*.safetensors*",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
    )

    _copy_tree(waypoint / "transformer", output_dir / "transformer")
    _copy_tree(waypoint / "vae", output_dir / "vae")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "text_encoder").mkdir(exist_ok=True)
    (output_dir / "tokenizer").mkdir(exist_ok=True)

    for path in text_encoder.glob("pytorch_model*.bin*"):
        shutil.copy2(path, output_dir / "text_encoder" / path.name)
    for path in text_encoder.glob("model*.safetensors*"):
        shutil.copy2(path, output_dir / "text_encoder" / path.name)
    shutil.copy2(text_encoder / "config.json", output_dir / "text_encoder")
    for name in (
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        if (text_encoder / name).is_file():
            shutil.copy2(text_encoder / name, output_dir / "tokenizer")

    (output_dir / "model_index.json").write_text(
        json.dumps(MODEL_INDEX, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "waypoint": {"repo_id": WAYPOINT_REPO, "revision": WAYPOINT_REVISION},
        "text_encoder": {
            "repo_id": TEXT_ENCODER_REPO,
            "revision": TEXT_ENCODER_REVISION,
        },
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build_bundle(args.output_dir.resolve())


if __name__ == "__main__":
    main()
