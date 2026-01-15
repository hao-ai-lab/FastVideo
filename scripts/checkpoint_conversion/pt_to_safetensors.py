#!/usr/bin/env python3
"""Convert a PyTorch checkpoint (.pt) to a safetensors file."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_pt_to_safetensors(
    input_path: str,
    output_path: str,
    key: str | None = None,
    force: bool = False,
    skip_patterns: list[str] | None = None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    checkpoint = torch.load(input_path, map_location="cpu")

    state_dict: dict[str, torch.Tensor]
    if isinstance(checkpoint, dict):
        if key is not None:
            if key not in checkpoint:
                raise KeyError(f"Key {key!r} not found in checkpoint.")
            state_dict = checkpoint[key]
        else:
            for k in ("state_dict", "model_state_dict", "model", "ema"):
                if k in checkpoint:
                    state_dict = checkpoint[k]
                    break
            else:
                state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a dict state_dict, got {type(state_dict)}")

    if skip_patterns:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(pat in k for pat in skip_patterns)
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint (.pt) to safetensors."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input .pt checkpoint file"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output .safetensors file"
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional key to extract from checkpoint dict (e.g. 'model_state_dict')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "--skip-pattern",
        action="append",
        dest="skip_patterns",
        help="Parameter name patterns to skip (can be used multiple times)"
    )

    args = parser.parse_args()

    convert_pt_to_safetensors(
        args.input,
        args.output,
        args.key,
        args.force,
        args.skip_patterns
    )


if __name__ == "__main__":
    main()

