# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K draft bundles: a draft handed over as its native latent plus its conditioning.

A paired draft worker (for example a FastH3 server with ``FASTVIDEO_H3_EXPORT_DRAFT_DIR``) can save, per finished
clip, a ``.pt`` dict ``{"format": 1, "latent": [C, T, h, w], "prompt_embeds": ..., "text_token_tags": ...,
"prompt": ...}`` in the normalized H3 latent space. Passed as the draft path, it skips the draft VAE encode, and when it
carries prompt embeddings the text encoder too. A bare latent tensor ``.pt`` keeps working as before.
"""

from __future__ import annotations

from typing import Any

import torch

BUNDLE_FORMAT = 1
BUNDLE_EXTRA_KEY = "open_h3_regenerate_2k_draft_bundle"


def is_bundle_path(path: str | None) -> bool:
    return bool(path) and str(path).endswith(".pt")


def load_draft_bundle(path: str) -> dict[str, Any] | None:
    """The bundle dict at ``path``, or ``None`` when the file holds a bare latent tensor."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    if payload.get("format") != BUNDLE_FORMAT or not isinstance(payload.get("latent"), torch.Tensor):
        raise ValueError(f"{path} is not a format-{BUNDLE_FORMAT} Open-H3-Regenerate-2K draft bundle.")
    return payload


def bundle_prompt_conditioning(bundle: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(prompt_embeds, text_token_tags)`` carried by the bundle, or ``None`` when it has none."""
    embeds, tags = bundle.get("prompt_embeds"), bundle.get("text_token_tags")
    if isinstance(embeds, torch.Tensor) and isinstance(tags, torch.Tensor):
        return embeds, tags
    return None


__all__ = ["BUNDLE_EXTRA_KEY", "BUNDLE_FORMAT", "bundle_prompt_conditioning", "is_bundle_path", "load_draft_bundle"]
