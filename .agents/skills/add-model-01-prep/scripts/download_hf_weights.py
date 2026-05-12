#!/usr/bin/env python3
"""Download HF weights using the standard FastVideo token env vars."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF model snapshot or selected files into a local directory."
    )
    parser.add_argument("repo_id", help="HF repo id, for example Org/Model")
    parser.add_argument("local_dir", help="Destination directory")
    parser.add_argument("--repo-type", default="model", help="HF repo type (default: model)")
    parser.add_argument("--revision", help="HF branch, tag, or commit")
    parser.add_argument(
        "--file-name",
        action="append",
        default=[],
        help="Download one file; may be repeated. If omitted, download full snapshot.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Snapshot allow pattern; may be repeated. Ignored when --file-name is used.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Snapshot ignore pattern; may be repeated. Ignored when --file-name is used.",
    )
    return parser.parse_args()


def resolve_token() -> tuple[str | None, str | None]:
    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return key, value
    return None, None


def main() -> int:
    args = parse_args()
    token_env, token = resolve_token()
    local_dir = Path(args.local_dir).expanduser()

    if token_env:
        print(f"token_env: {token_env}")
    else:
        print("token_env: none", file=sys.stderr)

    try:
        if local_dir.exists() and not local_dir.is_dir():
            print(
                f"error: destination exists and is not a directory: {local_dir}",
                file=sys.stderr,
            )
            return 1
        local_dir.mkdir(parents=True, exist_ok=True)
        if args.file_name:
            from huggingface_hub import hf_hub_download

            for file_name in args.file_name:
                path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=file_name,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    local_dir=str(local_dir),
                    token=token,
                )
                print(f"downloaded_file: {path}")
        else:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=args.allow_pattern or None,
                ignore_patterns=args.ignore_pattern or None,
            )
            print(f"downloaded_snapshot: {path}")
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"local_dir: {local_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
