# SPDX-License-Identifier: Apache-2.0
"""Push a converted daVinci-MagiHuman Diffusers-format directory to the Hub.

This is a thin wrapper around `huggingface_hub.create_repo` + `upload_folder`,
dedicated to the MagiHuman upload flow. It does NOT modify `create_hf_repo.py`
(which is LTX-2-oriented and rewrites component weights inside an existing
Diffusers repo).

Example (one-shot per variant):
    python scripts/checkpoint_conversion/push_magi_human_to_hf.py \\
        --local-dir converted_weights/magi_human_base \\
        --repo-id FastVideo/MagiHuman-Base-Diffusers \\
        --public

    python scripts/checkpoint_conversion/push_magi_human_to_hf.py \\
        --local-dir converted_weights/magi_human_distill \\
        --repo-id FastVideo/MagiHuman-Distilled-Diffusers \\
        --public

After upload, the local directory can be deleted — the HF repo is the
source of truth. `VideoGenerator.from_pretrained("FastVideo/...")` pulls
shards on demand.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def _validate_local_dir(local_dir: Path) -> None:
    required = ["model_index.json", "transformer"]
    missing = [r for r in required if not (local_dir / r).exists()]
    if missing:
        sys.exit(
            f"Error: {local_dir} is missing {missing}. Run "
            f"convert_magi_human_to_diffusers.py first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-dir", required=True, help="Path to the converted Diffusers directory.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. FastVideo/MagiHuman-Base-Diffusers.")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private). Mutually exclusive with --private.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private. Default when neither --public nor --private is set.",
    )
    parser.add_argument(
        "--commit-message",
        default="Initial upload of daVinci-MagiHuman Diffusers-format conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe what would happen without creating a repo or uploading.",
    )
    args = parser.parse_args()

    if args.public and args.private:
        sys.exit("Error: --public and --private are mutually exclusive.")
    private = args.private or not args.public

    local_dir = Path(args.local_dir).resolve()
    _validate_local_dir(local_dir)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")
    )
    if not token:
        sys.exit(
            "Error: no HF token in env (set HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY)."
        )

    api = HfApi()
    me = api.whoami(token=token)
    print(f"token user: {me.get('name')}")
    print(f"source:     {local_dir}")
    print(f"target:     {args.repo_id}")
    print(f"visibility: {'private' if private else 'public'}")
    if args.dry_run:
        print("(dry run — not creating or uploading)")
        return

    print(f"-> create_repo (exist_ok=True)")
    create_repo(
        repo_id=args.repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    print(f"-> upload_folder (this can take a while for 30 GB)")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        token=token,
        commit_message=args.commit_message,
        repo_type="model",
    )
    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
