# SPDX-License-Identifier: Apache-2.0
"""Reject NVIDIA FastWan-QAD checkpoints on the Apple Silicon MLX path.

FastMetal-QAD is the Apple Silicon release: DMD2 students trained on the affine
INT8 grid, shipped as packed ``mlx_dit.safetensors`` + ``mlx_dit.json``.
``FastVideo/FastWan-QAD-1.3B`` and ``FastVideo/FastWan-QAD-FP8-1.3B`` are
NVIDIA-only QAD checkpoints (NVFP4 / FP8). Loading them through the MLX
Diffusers path silently requantizes the wrong weights and produces videos that
ignore the prompt. Fail loudly instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FASTMETAL_COLLECTION_URL = "https://huggingface.co/collections/FastVideo/fastmetal"
FASTMETAL_BLOG_URL = "https://haoailab.com/blogs/fastmetal/"

FASTMETAL_MODEL_IDS = (
    "FastVideo/FastMetal-1.3B-QAD",
    "FastVideo/FastMetal-5B-QAD",
    "FastVideo/FastMetal-14B-QAD",
)

MLX_DIT_MANIFEST = "mlx_dit.json"
MLX_DIT_WEIGHTS = "mlx_dit.safetensors"
CUDA_QAD_OVERLAY_DIR = "generator_inference_transformer"

# Directory / HF-cache names for the NVIDIA FastWan-QAD family. The INT8 Apple
# snapshots used an older FastWan-QAD-INT8-* name; ``int8`` in the path is
# excluded below so those packed MLX checkpoints still load.
_NVIDIA_FASTWAN_QAD_RE = re.compile(
    r"(?:^|[/\\._-]|--)fastwan-qad-(?:fp8-)?1\.3b(?:-sa2)?(?:$|[/\\._-]|--)",
    re.IGNORECASE,
)

_NVIDIA_QUANT_MARKERS = (
    "nvfp4",
    "nv_fp4",
    "sageattention3",
    "attn_qat_infer",
)


class UnsupportedMLXCheckpointError(ValueError):
    """Raised when an NVIDIA FastWan-QAD (or similarly incompatible) tree is used on MLX."""


def is_mlx_dit_checkpoint(path: str | Path) -> bool:
    """Return True if ``path`` is a packed FastMetal / MLX DiT directory."""
    checkpoint_dir = Path(path)
    return (checkpoint_dir / MLX_DIT_MANIFEST).is_file() and (checkpoint_dir / MLX_DIT_WEIGHTS).is_file()


def discover_mlx_checkpoint(*candidates: str | Path | None) -> Path | None:
    """Return the first candidate that is a packed MLX DiT directory."""
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if is_mlx_dit_checkpoint(path):
            return path
    return None


def resolve_mlx_checkpoint(explicit: str | Path | None, *search_roots: str | Path | None) -> Path | None:
    """Prefer an explicit ``--mlx-checkpoint``, otherwise scan search roots."""
    if explicit is not None:
        return Path(explicit)
    return discover_mlx_checkpoint(*search_roots)


def mlx_checkpoint_missing_hint(checkpoint_dir: str | Path) -> str:
    """Extra FileNotFoundError text when a directory is not a packed MLX DiT."""
    nvidia_reason = nvidia_fastwan_qad_reason(checkpoint_dir)
    prefix = (f"Not an MLX DiT checkpoint directory: {checkpoint_dir} "
              f"(expected {MLX_DIT_MANIFEST} and {MLX_DIT_WEIGHTS}).")
    if nvidia_reason is not None:
        return prefix + "\n\n" + _nvidia_fastwan_qad_message(Path(checkpoint_dir), nvidia_reason)
    return prefix + "\n\n" + _fastmetal_howto()


def raise_if_unsupported_mlx_checkpoint(*paths: str | Path | None) -> None:
    """Raise if any path is an NVIDIA FastWan-QAD tree that must not run on MLX.

    Packed FastMetal / MLX DiT directories are always allowed, including the
    older FastWan-QAD-INT8 directory name, because those already contain
    ``mlx_dit.json``.
    """
    for path in paths:
        if path is None:
            continue
        checkpoint = Path(path)
        if is_mlx_dit_checkpoint(checkpoint):
            continue
        reason = nvidia_fastwan_qad_reason(checkpoint)
        if reason is None:
            continue
        raise UnsupportedMLXCheckpointError(_nvidia_fastwan_qad_message(checkpoint, reason))


def nvidia_fastwan_qad_reason(path: str | Path) -> str | None:
    """Return a short reason if ``path`` looks like NVIDIA FastWan-QAD, else None."""
    checkpoint = Path(path)
    haystack = _path_haystack(checkpoint)
    if "int8" in haystack and is_mlx_dit_checkpoint(checkpoint):
        return None
    if _NVIDIA_FASTWAN_QAD_RE.search(haystack) and "int8" not in haystack:
        return "NVIDIA FastWan-QAD checkpoint name (NVFP4/FP8, not FastMetal INT8)"
    if (checkpoint / CUDA_QAD_OVERLAY_DIR).is_dir() or (checkpoint.parent / CUDA_QAD_OVERLAY_DIR).is_dir():
        return f"CUDA QAD overlay directory ({CUDA_QAD_OVERLAY_DIR}/)"
    for config_path in _config_candidates(checkpoint):
        marker = _quant_marker_in_file(config_path)
        if marker is not None:
            return f"{config_path.name} contains NVIDIA quantization marker {marker!r}"
    return None


def _path_haystack(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return str(resolved).replace("\\", "/").lower()


def _config_candidates(path: Path) -> list[Path]:
    roots = [path.parent, path.parent.parent] if path.is_file() else [path, path.parent, path / "transformer"]
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        for candidate in (root / "config.json", root / "model_index.json", root / "transformer" / "config.json"):
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            files.append(candidate)
    return files


def _quant_marker_in_file(config_path: Path) -> str | None:
    try:
        payload = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    lowered = payload.lower()
    for marker in _NVIDIA_QUANT_MARKERS:
        if marker in lowered:
            return marker
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    quant = parsed.get("quantization_config") if isinstance(parsed, dict) else None
    if isinstance(quant, dict):
        blob = json.dumps(quant).lower()
        for marker in ("fp8", "float8", "nvfp4", "fp4"):
            if marker in blob:
                return marker
    return None


def _fastmetal_howto() -> str:
    models = ", ".join(FASTMETAL_MODEL_IDS)
    return ("The Apple Silicon MLX runtime requires FastMetal-QAD INT8 checkpoints, "
            f"not CUDA FastWan-QAD weights.\n"
            f"  Download: hf download FastVideo/FastMetal-1.3B-QAD --local-dir ./FastMetal-1.3B-QAD\n"
            "  FastMetal repos ship mlx_dit.json (not transformer/config.json).\n"
            f"  Run: python examples/inference/basic/mlx_wan_prompt_to_video.py "
            f"--model-root ./FastMetal-1.3B-QAD --mlx-checkpoint ./FastMetal-1.3B-QAD\n"
            f"  5B uses examples/inference/basic/mlx_wan22_generate.py with FastMetal-5B-QAD.\n"
            f"  Models: {models}\n"
            f"  Guide: {FASTMETAL_BLOG_URL}\n"
            f"  Collection: {FASTMETAL_COLLECTION_URL}\n"
            "FastVideo/FastWan-QAD-1.3B and FastVideo/FastWan-QAD-FP8-1.3B are "
            "NVIDIA NVFP4/FP8 QAD checkpoints.")


def _nvidia_fastwan_qad_message(path: Path, reason: str) -> str:
    return (f"Refusing to load {path} on the Apple Silicon MLX runtime ({reason}).\n\n" + _fastmetal_howto())
