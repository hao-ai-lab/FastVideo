"""Golden probes and their IO — the data hub for numerics anchoring.

Goldens are small captured tensors from the *official* Wan2.1 implementation
(Wan-Video/Wan2.1), produced once by ``capture_official.py`` in a scratch
environment and committed under ``fastvideo2/evidence/goldens/<set>/``. Every
other implementation (fastvideo2 via diffusers, fastvideo main) is compared
*against the files*, never against live official code: the official repo
appears in this codebase only as a commit hash inside ``manifest.json``.

Probe inputs are defined here, in numpy, from fixed legacy ``RandomState``
seeds (stable across numpy versions) — and are additionally stored inside the
golden files, so the files remain the source of truth even if this module
changes. This module is numpy+stdlib only.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

GOLDENS_ROOT = os.path.join(os.path.dirname(__file__), "..", "evidence", "goldens")
DEFAULT_SET = "wan21-official"

# --------------------------------------------------------------------------- #
# Probe definitions (must match verify.PROBE where they overlap)              #
# --------------------------------------------------------------------------- #
PROMPT = "A red panda eating bamboo in a sunlit forest, cinematic."
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

DIT_LATENT_SHAPE = (1, 16, 5, 60, 104)   # the T2 probe geometry (17f @ 480x832)
DIT_TIMESTEPS = (750.0, 250.0)           # high- and low-noise probes (0..1000 scale)
VAE_LATENT_SHAPE = (1, 16, 5, 30, 52)    # half spatial, keeps the golden video small
SCHEDULES = ((8, 3.0), (50, 3.0), (50, 8.0))  # (steps, shift): probe, 480p default, official-recommended


def dit_probe_latent() -> np.ndarray:
    return np.random.RandomState(2024).standard_normal(DIT_LATENT_SHAPE).astype(np.float32)


def vae_probe_latent() -> np.ndarray:
    # ~unit-normal in the *normalized* latent space (what the denoise loop emits)
    return np.random.RandomState(2025).standard_normal(VAE_LATENT_SHAPE).astype(np.float32)


def pad_context(embeds: np.ndarray, text_len: int = 512) -> np.ndarray:
    """Zero-pad unpadded [len, 4096] text embeddings to [text_len, 4096] — the
    convention both the official model (internally) and diffusers (externally)
    use, so it is the canonical exchange form for DiT context."""
    out = np.zeros((text_len, embeds.shape[1]), dtype=np.float32)
    out[: embeds.shape[0]] = embeds
    return out


# --------------------------------------------------------------------------- #
# IO                                                                           #
# --------------------------------------------------------------------------- #
def golden_dir(name: str = DEFAULT_SET) -> str:
    return os.path.normpath(os.path.join(GOLDENS_ROOT, name))


def save_golden(dir_: str, name: str, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> str:
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    _merge_manifest(dir_, {name: meta})
    return path


def load_golden(dir_: str, name: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(dir_, f"{name}.npz")) as z:
        return {k: z[k] for k in z.files}


def _merge_manifest(dir_: str, patch: dict) -> None:
    path = os.path.join(dir_, "manifest.json")
    manifest = {}
    if os.path.exists(path):
        with open(path) as f:
            manifest = json.load(f)
    manifest.update(patch)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)


def load_manifest(dir_: str) -> dict:
    with open(os.path.join(dir_, "manifest.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Comparison metrics (numpy; shared by every impl adapter)                     #
# --------------------------------------------------------------------------- #
def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(b)) or 1.0
    return float(np.linalg.norm(a - b)) / denom


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def compare(name: str, ours: np.ndarray, golden: np.ndarray, tol_rel: float) -> dict:
    """One typed comparison record; shape mismatch is a hard fail."""
    if tuple(ours.shape) != tuple(golden.shape):
        return {"name": name, "status": "fail", "detail": f"shape {ours.shape} != golden {golden.shape}"}
    r = rel_l2(ours, golden)
    return {"name": name, "status": "pass" if r <= tol_rel else "fail",
            "rel_l2": r, "max_abs": max_abs(ours, golden), "tol_rel": tol_rel}
