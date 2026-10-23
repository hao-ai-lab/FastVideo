# SPDX-License-Identifier: Apache-2.0
"""Wrap official Bernini-R 1.3B components in a FastVideo manifest.

No tensor is renamed, cast, or dropped. The native arch config owns the
Diffusers-to-FastVideo parameter mapping. Default output links the original
files; --copy makes a portable standalone directory.
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

SOURCE_REPO = "ByteDance/Bernini-R-1.3B-Diffusers"
SOURCE_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
SKIPPED_KEYS: tuple[str, ...] = ()
COMPONENTS = {
    "transformer": ["diffusers", "BerniniRTransformer3DModel"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
}


def prepare_checkpoint(source: Path, output: Path, *, copy: bool = False) -> Path:
    source = source.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("The output must be outside the source checkpoint")
    with (source / "config.json").open() as handle:
        root_config = json.load(handle)
    if (root_config.get("model_type") != "bernini_renderer"
            or not root_config.get("use_src_id_rotary_emb")
            or root_config.get("skip_transformer_1")
            or not root_config.get("skip_transformer_2")
            or root_config.get("switch_dit_boundary") != 0
            or not root_config.get("use_unipc")):
        raise ValueError("Expected the single-expert Bernini-R 1.3B source-ID/UniPC checkpoint")
    with (source / "transformer/config.json").open() as handle:
        arch = json.load(handle)
    expected_arch = {
        "num_attention_heads": 12,
        "num_layers": 30,
        "in_channels": 16,
        "out_channels": 16,
        "attention_head_dim": 128,
        "ffn_dim": 8960,
        "patch_size": [1, 2, 2],
    }
    for key, expected in expected_arch.items():
        if arch.get(key) != expected:
            raise ValueError(f"Bernini-R 1.3B requires {key}={expected}, got {arch.get(key)}")
    for component in COMPONENTS:
        if not (source / component).is_dir():
            raise ValueError(f"Missing component {component}/ in {source}")
    if not list((source / "transformer").glob("*.safetensors")):
        raise ValueError("No transformer safetensors found")
    index = source / "transformer/diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The transformer shard index requires a nonempty weight_map")
        for shard in weight_map.values():
            if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError("Transformer shard names must be local safetensors filenames")
            if not (source / "transformer" / shard).is_file():
                raise ValueError("The transformer shard index references missing weight files")
    metadata = source / ".cache/huggingface/download/config.json.metadata"
    source_revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    if source_revision is None and source.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", source.name):
        source_revision = source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every component and the manifest have been prepared.
    with tempfile.TemporaryDirectory(prefix="bernini-layout-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "checkpoint"
        prepared.mkdir()
        for component in COMPONENTS:
            src = source / component
            dest = prepared / component
            if component == "transformer":
                dest.mkdir()
                for file in src.iterdir():
                    if file.name == "config.json" or not file.is_file():
                        continue
                    if copy:
                        shutil.copy2(file, dest / file.name)
                    else:
                        (dest / file.name).symlink_to(file.resolve())
            elif copy:
                shutil.copytree(src, dest)
            else:
                dest.symlink_to(src, target_is_directory=True)
        arch["_class_name"] = "BerniniRTransformer3DModel"
        (prepared / "transformer/config.json").write_text(json.dumps(arch, indent=2) + "\n")
        manifest = {
            "_class_name": "BerniniRPipeline",
            "_diffusers_version": arch.get("_diffusers_version", "0.35.2"),
            "_fastvideo_source": {
                "repo": SOURCE_REPO,
                "revision": source_revision,
                "layout_only": True,
            },
            **COMPONENTS,
        }
        (prepared / "model_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
        prepared.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded official checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlinking component files")
    args = parser.parse_args()
    print(prepare_checkpoint(args.source, args.output, copy=args.copy))


if __name__ == "__main__":
    main()
