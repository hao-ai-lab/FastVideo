#!/usr/bin/env python3
"""Inspect a Hugging Face repo or local weight directory layout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY")
RAW_WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".ckpt", ".bin")
KNOWN_COMPONENTS = {
    "audio_vae",
    "conditioner",
    "feature_extractor",
    "image_encoder",
    "scheduler",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "transformer",
    "transformer_2",
    "unet",
    "upsampler",
    "vae",
    "vocoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a HF repo or local directory as Diffusers, raw, custom, or unknown."
    )
    parser.add_argument("source", help="HF repo id or local weights directory")
    parser.add_argument("--repo-type", default="model", help="HF repo type (default: model)")
    parser.add_argument("--revision", help="HF revision to inspect")
    parser.add_argument(
        "--max-local-files",
        type=int,
        default=20000,
        help="Maximum local files to scan recursively (default: 20000)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=80,
        help="Number of file paths to print in human output (default: 80)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    return parser.parse_args()


def resolve_token() -> tuple[str | None, str | None]:
    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return key, value
    return None, None


def load_local_files(root: Path, max_files: int) -> tuple[list[str], bool]:
    files: list[str] = []
    truncated = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files.append(path.relative_to(root).as_posix())
        if len(files) >= max_files:
            truncated = True
            break
    return sorted(files), truncated


def load_local_model_index(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    index_path = root / "model_index.json"
    if not index_path.is_file():
        return None, None
    try:
        return json.loads(index_path.read_text()), None
    except Exception as exc:  # noqa: BLE001 - surface malformed JSON clearly.
        return None, f"failed to parse local model_index.json: {exc}"


def load_remote_files(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
) -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(
        list_repo_files(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    )


def load_remote_model_index(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    token: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="model_index.json",
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001 - missing/inaccessible file is data.
        return None, f"failed to download model_index.json: {exc}"

    try:
        return json.loads(Path(path).read_text()), None
    except Exception as exc:  # noqa: BLE001 - surface malformed JSON clearly.
        return None, f"failed to parse remote model_index.json: {exc}"


def root_file_names(files: list[str]) -> set[str]:
    return {name for name in files if "/" not in name}


def component_names(files: list[str], model_index: dict[str, Any] | None) -> list[str]:
    components: set[str] = set()
    for name in files:
        parts = name.split("/", 1)
        if len(parts) != 2:
            continue
        top, rest = parts
        if top in KNOWN_COMPONENTS or rest == "config.json":
            components.add(top)

    if model_index:
        for key, value in model_index.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list) and len(value) == 2:
                components.add(key)

    return sorted(components)


def classify_layout(
    files: list[str],
    model_index: dict[str, Any] | None,
    components: list[str],
) -> tuple[str, str]:
    roots = root_file_names(files)
    raw_weight_files = [name for name in roots if name.endswith(RAW_WEIGHT_SUFFIXES)]
    has_model_index = "model_index.json" in roots or model_index is not None

    if has_model_index and components:
        return "diffusers", "no"
    if has_model_index:
        return "custom", "unknown"
    if raw_weight_files:
        return "raw_official", "yes"
    if any(name.endswith(RAW_WEIGHT_SUFFIXES) for name in files):
        return "custom", "yes"
    return "unknown", "unknown"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    token_env, token = resolve_token()
    source_path = Path(args.source).expanduser()
    is_local = source_path.exists()

    if is_local:
        root = source_path.resolve()
        if not root.is_dir():
            raise ValueError(f"local source is not a directory: {root}")
        files, truncated = load_local_files(root, args.max_local_files)
        model_index, model_index_error = load_local_model_index(root)
        source_kind = "local"
        source = str(root)
    else:
        files = load_remote_files(args.source, args.repo_type, args.revision, token)
        truncated = False
        model_index, model_index_error = load_remote_model_index(
            args.source,
            args.repo_type,
            args.revision,
            token,
        )
        source_kind = "hf"
        source = args.source

    components = component_names(files, model_index)
    source_layout, needs_conversion = classify_layout(files, model_index, components)

    return {
        "source": source,
        "source_kind": source_kind,
        "repo_type": None if is_local else args.repo_type,
        "revision": args.revision,
        "token_env": token_env,
        "source_layout": source_layout,
        "needs_conversion": needs_conversion,
        "model_index_class": (model_index or {}).get("_class_name"),
        "model_index_diffusers_version": (model_index or {}).get("_diffusers_version"),
        "model_index_error": model_index_error,
        "components_seen": components,
        "file_count": len(files),
        "file_scan_truncated": truncated,
        "files_sample": files[: args.sample_limit],
    }


def print_human(result: dict[str, Any]) -> None:
    for key in (
        "source",
        "source_kind",
        "repo_type",
        "revision",
        "token_env",
        "source_layout",
        "needs_conversion",
        "model_index_class",
        "model_index_diffusers_version",
        "model_index_error",
        "file_count",
        "file_scan_truncated",
    ):
        value = result.get(key)
        if value is not None:
            print(f"{key}: {value}")

    components = result["components_seen"]
    print("components_seen: " + (", ".join(components) if components else "none"))
    print("files_sample:")
    for name in result["files_sample"]:
        print(f"  {name}")


def main() -> int:
    args = parse_args()
    try:
        result = build_result(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
