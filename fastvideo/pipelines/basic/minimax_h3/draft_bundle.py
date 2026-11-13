# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 draft bundles: a finished draft handed to another H3 pipeline as its native latent plus conditioning.

A generator (any MiniMax-H3 pipeline) writes a bundle when a request asks for one (``return_draft_bundle`` on the
OpenAI API, ``h3_draft_bundle_path`` on the batch); a consumer such as Open-H3-Regenerate-2K reads it as its draft
instead of an MP4. That skips the decode -> re-encode round trip, and when the text encoders match, the consumer's
text encoder as well. The same file works in-process, on one machine, or across machines.

``{"format": 1, "latent": [C, T, h, w] normalized H3 latent, "prompt_embeds": ..., "text_token_tags": ...,
"prompt": str, "text_encoder_fingerprint": str | None}``; ``torch.load(weights_only=True)`` reads it safely.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

BUNDLE_FORMAT = 1
BUNDLE_SUFFIX = ".h3draft.pt"
BUNDLE_MEDIA_TYPE = "application/x-fastvideo-h3-draft"
BUNDLE_EXTRA_KEY = "open_h3_regenerate_2k_draft_bundle"
BUNDLE_PATH_EXTRA_KEY = "h3_draft_bundle_path"
PROMPT_EMBEDS_POLICIES = ("match", "always", "never")


def is_bundle_path(path: str | None) -> bool:
    return bool(path) and str(path).endswith(".pt")


def text_encoder_fingerprint(directory: str | os.PathLike[str] | None, quant: object = None) -> str | None:
    """Stable id of the text encoder files in ``directory`` (config and shard names/sizes), or ``None``.

    Two pipelines whose fingerprints match produce the same prompt embeddings, so a consumer can reuse the producer's
    embeddings instead of running its own encoder. It never loads the model.
    """
    if directory is None:
        return None
    root = Path(directory)
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    config = root / "config.json"
    if config.is_file():
        digest.update(json.dumps(json.loads(config.read_text()), sort_keys=True).encode("utf-8"))
    for shard in sorted(root.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())
    digest.update(f"quant:{quant}".encode())
    return digest.hexdigest()[:32]


def save_draft_bundle(path: str | os.PathLike[str], *, latent: torch.Tensor, prompt_embeds: torch.Tensor | None,
                      text_token_tags: torch.Tensor | None, prompt: str, text_encoder: str | None) -> None:
    """Write a bundle atomically (temp file, then rename)."""
    latent = latent.detach()
    if latent.ndim == 5:
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(f"A draft bundle latent must be [C, T, h, w], got {tuple(latent.shape)}.")
    bundle = {
        "format": BUNDLE_FORMAT,
        "latent": latent.to(device="cpu", dtype=torch.float32).contiguous(),
        "prompt_embeds": None if prompt_embeds is None else prompt_embeds.detach().cpu().contiguous(),
        "text_token_tags": None if text_token_tags is None else text_token_tags.detach().cpu().contiguous(),
        "prompt": prompt,
        "text_encoder_fingerprint": text_encoder,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    torch.save(bundle, tmp)
    os.replace(tmp, target)


def load_draft_bundle(path: str) -> dict[str, Any] | None:
    """The bundle dict at ``path``, or ``None`` when the file holds a bare latent tensor."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    if payload.get("format") != BUNDLE_FORMAT or not isinstance(payload.get("latent"), torch.Tensor):
        raise ValueError(f"{path} is not a format-{BUNDLE_FORMAT} MiniMax-H3 draft bundle.")
    return payload


def bundle_prompt_conditioning(bundle: dict[str, Any],
                               *,
                               policy: str = "always",
                               text_encoder: str | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(prompt_embeds, text_token_tags)`` to reuse from the bundle, or ``None`` to run the consumer's encoder.

    ``policy``: ``match`` reuses them only when the bundle's text-encoder fingerprint equals ``text_encoder``;
    ``always`` reuses any embeddings the bundle carries (a different encoder conditions the consumer differently);
    ``never`` always re-encodes.
    """
    if policy not in PROMPT_EMBEDS_POLICIES:
        raise ValueError(f"policy must be one of {PROMPT_EMBEDS_POLICIES}, got {policy!r}")
    embeds, tags = bundle.get("prompt_embeds"), bundle.get("text_token_tags")
    if policy == "never" or not isinstance(embeds, torch.Tensor) or not isinstance(tags, torch.Tensor):
        return None
    if policy == "match":
        source = bundle.get("text_encoder_fingerprint")
        if source is None or text_encoder is None or source != text_encoder:
            return None
    return embeds, tags


__all__ = [
    "BUNDLE_EXTRA_KEY",
    "BUNDLE_FORMAT",
    "BUNDLE_MEDIA_TYPE",
    "BUNDLE_PATH_EXTRA_KEY",
    "BUNDLE_SUFFIX",
    "PROMPT_EMBEDS_POLICIES",
    "bundle_prompt_conditioning",
    "is_bundle_path",
    "load_draft_bundle",
    "save_draft_bundle",
    "text_encoder_fingerprint",
]
