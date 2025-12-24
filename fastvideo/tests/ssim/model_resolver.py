# SPDX-License-Identifier: Apache-2.0
"""Helpers for SSIM tests to resolve model paths.

Why this exists:
- In Modal CI we mount a `modal.Volume` containing model weights into the container.
- SSIM tests should be deterministic and avoid downloading models from the network.

Conventions:
- Set `MODEL_PATH` (already used by LongCat tests) to the *base directory* that contains
  per-model folders, e.g. `/root/data/weights`.
- Each model should live under one of these layouts:
    - <MODEL_PATH>/<model_id>/
    - <MODEL_PATH>/<repo_name>/                 (repo_name = last path component of HF id)
    - <MODEL_PATH>/<org>/<repo_name>/           (mirrors HF "org/repo" structure)

Inference steps are intentionally hard-coded inside SSIM tests to keep them fast.
"""

from __future__ import annotations

import os
from pathlib import Path


def _looks_like_local_path(path: str) -> bool:
    return path.startswith(("/", "./", "../", "weights/", "data/"))


def get_ssim_model_base_dir() -> str | None:
    base = (
        os.getenv("FASTVIDEO_SSIM_MODEL_DIR")
        or os.getenv("MODEL_PATH")
        or os.getenv("FASTVIDEO_MODEL_DIR")
    )
    base = (base or "").strip()
    return base or None


def resolve_ssim_model_path(*, model_id: str, model_ref: str) -> str:
    """Resolve a model path for SSIM tests.

    - If `model_ref` already points to an existing local path, return it.
    - Else if a base dir is configured (MODEL_PATH/FASTVIDEO_SSIM_MODEL_DIR), map to a local folder.
    - Else fall back to `model_ref` (HF repo id); this is mainly for local dev.
    """
    model_ref = (model_ref or "").strip()
    if not model_ref:
        raise ValueError("Empty model_ref passed to resolve_ssim_model_path")

    if _looks_like_local_path(model_ref) and os.path.exists(model_ref):
        return model_ref

    base = get_ssim_model_base_dir()
    if not base:
        # No base dir configured → allow HF IDs (existing behavior).
        return model_ref

    base_p = Path(base)
    repo_name = model_ref.split("/")[-1]

    candidates: list[Path] = []
    if model_id:
        candidates.append(base_p / model_id)
    if repo_name and repo_name != model_id:
        candidates.append(base_p / repo_name)
    # Mirror HF structure under base dir: <base>/<org>/<repo>
    if "/" in model_ref and not _looks_like_local_path(model_ref):
        candidates.append(base_p / model_ref)

    for c in candidates:
        if c.exists():
            return str(c)

    candidates_str = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "SSIM model not found under configured model base dir.\n"
        f"- MODEL_PATH/FASTVIDEO_SSIM_MODEL_DIR: {base}\n"
        f"- model_id: {model_id}\n"
        f"- model_ref: {model_ref}\n"
        f"- tried:{candidates_str}\n"
        "Fix: upload/sync the model folder into the mounted Modal Volume and ensure it appears "
        "under the base dir with one of the expected layouts."
    )


