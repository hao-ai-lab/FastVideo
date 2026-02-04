# SPDX-License-Identifier: Apache-2.0
"""Create a FastVideo-compatible diffusers stub repo for Waypoint-1-Small.

FastVideo expects a `model_index.json` at the model root with module entries in
either:

1) Standard diffusers style:
   "transformer": ["diffusers", "WorldModel"]

2) Modular style (supported by FastVideo):
   "text_encoder": [null, null, {"pretrained_model_name_or_path": "...", "type_hint": ["transformers","UMT5EncoderModel"]}]

This script creates a minimal local directory that can be used with:

    StreamingVideoGenerator.from_pretrained("converted/waypoint_diffusers_stub")

The stub repo contains empty `transformer/` and `vae/` folders to satisfy
directory checks, and points components to upstream repos on HF.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="converted/waypoint_diffusers_stub",
        help="Output directory for the stub repo.",
    )
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vae"), exist_ok=True)

    model_index = {
        "_class_name": "WaypointPipeline",
        "_diffusers_version": "0.36.0",
        # Modules are resolved via FastVideo's modular loader support.
        "transformer": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "Overworld/Waypoint-1-Small",
                "subfolder": "transformer",
                "type_hint": ["diffusers", "WorldModel"],
            },
        ],
        "vae": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "Overworld/Waypoint-1-Small",
                "subfolder": "vae",
                "type_hint": ["diffusers", "WorldEngineVAE"],
            },
        ],
        "text_encoder": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "google/umt5-xl",
                "type_hint": ["transformers", "UMT5EncoderModel"],
            },
        ],
        "tokenizer": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "google/umt5-xl",
                "type_hint": ["transformers", "AutoTokenizer"],
            },
        ],
    }

    with open(os.path.join(out_dir, "model_index.json"), "w",
              encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)

    print(f"Wrote stub repo to: {out_dir}")
    print("Next:")
    print(
        f"  python examples/inference/basic/basic_waypoint_streaming.py  # after setting MODEL_ID to {out_dir}"
    )


if __name__ == "__main__":
    main()


