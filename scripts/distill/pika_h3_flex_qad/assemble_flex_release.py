# SPDX-License-Identifier: Apache-2.0
"""Assemble a self-contained NVFP4 Pika H3 Flex model directory from packed QAD exports.

The output loads directly in FastVideo (``basic_pika_h3_flex.py --model-path <out> --nvfp4-dit h3_dit``) with no
conversion step:

* ``transformer_ref`` (Ref2VA) and/or ``transformer`` (FL2VA): packed NVFP4 trunks from
  ``convert_minimax_h3_modelopt_nvfp4_dit.py``. They already contain the Pika adapter (merged before QAD).
* ``pika_h3_flex_merged.json``: marks those trunks as merged so the pipeline does not apply the adapter again. Its
  release id and adapter sha256 come from each trunk's ``pika_flex_qad.json`` contract written by
  ``merge_pika_adapter.py``.
* ``vae``, ``audio_vae``, ``tokenizer``, ``processor`` and ``text_encoder``: from the MiniMax-H3 snapshot
  (``--text-encoder`` substitutes e.g. an NVFP4 Qwen3-VL export); ``--vae-decoder-nvfp4`` optionally adds a packed
  NVFP4 VAE decoder directory.
* ``scheduler``, ``audio_scheduler``, ``modular_model_index.json`` and the license files: from the Pika H3 Flex
  release, which defines the six-forward re-noise sampler.

Files are hard-linked when source and output share a filesystem and copied otherwise.

    python scripts/distill/pika_h3_flex_qad/assemble_flex_release.py --output release/Pika-H3-Flex-NVFP4-QAD \\
        --base <MiniMax-H3 snapshot> --pika <Pika-H3-Flex snapshot> \\
        --transformer-ref exports/ref2va/transformer_ref --ref2va-merged data/models/pika-flex-ref2va-merged \\
        --transformer exports/fl2va/transformer --fl2va-merged data/models/pika-flex-fl2va-merged
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Kept import-free so assembly runs on CPU-only hosts; must equal
# fastvideo.pipelines.basic.pika_h3_flex.adapter.PIKA_MERGED_MARKER (checked by the loader tests).
PIKA_MERGED_MARKER = "pika_h3_flex_merged.json"
CONTRACT_FILE = "pika_flex_qad.json"
# Platform metadata that must not ship (macOS AppleDouble files, Finder state).
JUNK = shutil.ignore_patterns("._*", ".DS_Store")
BASE_COMPONENTS = ("vae", "audio_vae", "tokenizer", "processor", "text_encoder")
PIKA_COMPONENTS = ("scheduler", "audio_scheduler")
PIKA_FILES = ("modular_model_index.json", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def _link_or_copy(src: str, dst: str) -> None:
    # Hugging Face snapshots hold relative symlinks into blobs/: link or copy the target, never the link.
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def place_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"missing component directory {src}")
    shutil.copytree(src.resolve(), dst, copy_function=_link_or_copy, ignore=JUNK)


def merged_entry(merged_dir: Path, expected_task: str, expected_trunk: str) -> dict[str, str]:
    contract_path = merged_dir / CONTRACT_FILE
    if not contract_path.is_file():
        raise FileNotFoundError(f"{contract_path} not found; pass the directory merge_pika_adapter.py wrote")
    contract = json.loads(contract_path.read_text())
    if contract.get("task") != expected_task or contract.get("trunk") != expected_trunk:
        raise ValueError(f"{contract_path} is for task {contract.get('task')!r} / trunk {contract.get('trunk')!r}, "
                         f"expected {expected_task!r} / {expected_trunk!r}")
    return {
        "trunk": expected_trunk,
        "release_id": contract["release_id"],
        "adapter_sha256": contract["adapter_sha256"],
    }


def write_index(out: Path, trunks: set[str], repo_id: str | None) -> list[str]:
    index_path = out / "modular_model_index.json"
    index = json.loads(index_path.read_text())
    for name in ("transformer", "transformer_ref"):
        if name not in trunks:
            index.pop(name, None)
    if repo_id:
        for entry in index.values():
            if isinstance(entry, list) and len(entry) > 2 and isinstance(entry[2], dict):
                if "pretrained_model_name_or_path" in entry[2]:
                    entry[2]["pretrained_model_name_or_path"] = repo_id
    index_path.unlink()  # may be a hard link into the Pika snapshot
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return sorted(k for k, v in index.items() if isinstance(v, list))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path, help="MiniMax-H3 snapshot at the adapter's pinned revision")
    parser.add_argument("--pika", required=True, type=Path, help="Pika H3 Flex release snapshot")
    parser.add_argument("--transformer-ref", type=Path, help="packed NVFP4 Ref2VA trunk")
    parser.add_argument("--ref2va-merged", type=Path, help="merged Ref2VA model dir the QAD run started from")
    parser.add_argument("--transformer", type=Path, help="packed NVFP4 FL2VA trunk")
    parser.add_argument("--fl2va-merged", type=Path, help="merged FL2VA model dir the QAD run started from")
    parser.add_argument("--text-encoder", type=Path, help="text_encoder directory replacing the base one")
    parser.add_argument("--vae-decoder-nvfp4", type=Path, help="packed NVFP4 VAE decoder directory to include")
    parser.add_argument("--repo-id", help="rewrite modular_model_index.json component paths to this Hub repo id")
    args = parser.parse_args()

    trunks = {"transformer_ref": (args.transformer_ref, args.ref2va_merged, "ref2va"),
              "transformer": (args.transformer, args.fl2va_merged, "fl2va")}
    trunks = {name: spec for name, spec in trunks.items() if spec[0] is not None}
    if not trunks:
        parser.error("pass --transformer-ref and/or --transformer")
    for name, (_, merged_dir, task) in trunks.items():
        if merged_dir is None:
            parser.error(f"--{name.replace('_', '-')} needs --{task}-merged for the merged-adapter marker")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")
    args.output.mkdir(parents=True, exist_ok=True)

    merged = {}
    for name, (trunk_dir, merged_dir, task) in trunks.items():
        if not (trunk_dir / "nvfp4_weights.safetensors").is_file():
            raise FileNotFoundError(f"{trunk_dir} is not a packed NVFP4 export (no nvfp4_weights.safetensors)")
        place_tree(trunk_dir, args.output / name)
        merged[task] = merged_entry(merged_dir, task, name)
    for component in BASE_COMPONENTS:
        source = args.text_encoder if component == "text_encoder" and args.text_encoder else args.base / component
        place_tree(source, args.output / component)
    if args.vae_decoder_nvfp4:
        place_tree(args.vae_decoder_nvfp4, args.output / "vae_decoder_nvfp4")
    for component in PIKA_COMPONENTS:
        place_tree(args.pika / component, args.output / component)
    for name in PIKA_FILES:
        if (args.pika / name).is_file():
            _link_or_copy(str((args.pika / name).resolve()), str(args.output / name))
    if (args.pika / "licenses").is_dir():
        place_tree(args.pika / "licenses", args.output / "licenses")

    marker = {"format_version": 1, "merged": merged}
    (args.output / PIKA_MERGED_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    components = write_index(args.output, set(trunks), args.repo_id)
    print(json.dumps({"output": str(args.output), "components": components, "merged": sorted(merged)}))


if __name__ == "__main__":
    main()
