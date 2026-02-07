# SPDX-License-Identifier: Apache-2.0
"""
Convert LTX-2 latent upsampler weights to FastVideo format.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _find_shards(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]

    index_files = list(model_path.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as f:
            index = json.load(f)
        return sorted({model_path / shard for shard in index["weight_map"].values()})
    return sorted(Path(p) for p in glob.glob(str(model_path / "*.safetensors")))


def _read_metadata_config(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    if not metadata or "config" not in metadata:
        raise ValueError(f"Missing config metadata in {path}")
    return json.loads(metadata["config"])


def _load_weights(shards: list[Path]) -> dict:
    weights: dict = {}
    for shard in shards:
        weights.update(load_file(str(shard)))
    return weights


def _map_keys(weights: dict, add_model_prefix: bool) -> dict:
    remapped = {}
    for key, value in weights.items():
        new_key = key
        if not add_model_prefix and new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if add_model_prefix and not new_key.startswith("model."):
            new_key = f"model.{new_key}"
        remapped[new_key] = value
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LTX-2 spatial upsampler weights to FastVideo format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the official LTX-2 upsampler safetensors file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="LTX2LatentUpsampler",
        help="_class_name to write into config.json",
    )
    parser.add_argument(
        "--add-model-prefix",
        action="store_true",
        help="Prefix all weights with 'model.' for loading into wrapper modules",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    shards = _find_shards(source_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_path}")

    config = _read_metadata_config(shards[0])
    config["_class_name"] = args.class_name

    weights = _load_weights(shards)
    weights = _map_keys(weights, add_model_prefix=args.add_model_prefix)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_weights = output_dir / "model.safetensors"
    save_file(weights, str(output_weights))
    print(f"Saved weights to {output_weights}")

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
