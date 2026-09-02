# SPDX-License-Identifier: Apache-2.0
"""Fail-closed verifier for immutable H3 REST trajectory caches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from fastvideo.dataset.h3_rest_cache import validate_h3_rest_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip payload SHA-256 reads; sizes and manifest hashes are still checked.",
    )
    parser.add_argument(
        "--student-timesteps",
        nargs="+",
        type=float,
        default=[1000, 750, 500, 250, 0],
    )
    args = parser.parse_args()
    summary = validate_h3_rest_cache(
        args.cache_dir,
        verify_file_hashes=not args.metadata_only,
        expected_student_timesteps=args.student_timesteps,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
