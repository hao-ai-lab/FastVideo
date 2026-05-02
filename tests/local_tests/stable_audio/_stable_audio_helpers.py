# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Stable Audio local parity tests.

Centralizes the HF-token lookup, env-mirroring, and gated-repo access
probe that all parity tests in this directory need; uses
`fastvideo.utils.resolve_hf_token` as the source of truth for env-var
precedence so the tests stay in sync with the rest of the codebase.
"""
from __future__ import annotations

import os

from fastvideo.utils import resolve_hf_token


def setup_hf_env() -> None:
    """Mirror the resolved token to both `HF_TOKEN` and
    `HUGGINGFACE_HUB_TOKEN` so downstream libraries see it under
    whichever name they happen to check.
    """
    token = resolve_hf_token()
    if token is None:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def can_access_repo(repo_id: str, filename: str = "model_index.json") -> bool:
    """True iff the resolved HF token grants access to `filename` in
    `repo_id`. Cheapest possible probe (single small file fetch).
    """
    token = resolve_hf_token()
    if token is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return True
    except Exception:
        return False
