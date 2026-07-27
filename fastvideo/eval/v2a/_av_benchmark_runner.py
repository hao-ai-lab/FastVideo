# SPDX-License-Identifier: Apache-2.0
"""Standalone process runner for the official av-benchmark API.

This file is executed directly by an interpreter from an isolated environment.
Keep it free of ``fastvideo`` imports so the benchmark environment does not
need FastVideo's runtime dependency stack.
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Any
from collections.abc import Callable

_BASE_CACHE_FILES = (
    "pann_features.pth",
    "vggish_features.pth",
    "passt_features_embed.pth",
    "passt_logits.pth",
)
_VIDEO_CACHE_FILES = (
    "imagebind_audio.pth",
    "synchformer_audio.pth",
)
_CLAP_CACHE_FILES = (
    "clap_laion_audio.pth",
    "clap_ms_audio.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--audio-length", type=float, default=8.0)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--skip-video-related", action="store_true")
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--align-prediction-keys", action="store_true")
    return parser.parse_args()


def _load_backend() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from av_bench.evaluate import evaluate
        from av_bench.extract import extract
    except ImportError as error:
        raise ImportError("Official av_bench is not installed in this Python environment. "
                          "Install https://github.com/hkchengrex/av-benchmark and pass its "
                          "Python through --python-executable.") from error
    return extract, evaluate


def _required_prediction_files(*, skip_video_related: bool, skip_clap: bool) -> tuple[str, ...]:
    required = list(_BASE_CACHE_FILES)
    if not skip_video_related:
        required.extend(_VIDEO_CACHE_FILES)
    if not skip_clap:
        required.extend(_CLAP_CACHE_FILES)
    return tuple(required)


def _validate_gt_cache(gt_cache: Path) -> None:
    missing = [name for name in _BASE_CACHE_FILES if not (gt_cache / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Ground-truth cache {gt_cache} is missing required files: {missing}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _align_prediction_cache_keys(
    *,
    gt_cache: Path,
    prediction_cache: Path,
    required_files: tuple[str, ...],
) -> dict[str, int]:
    """Align filename-sanitized prediction keys to unique GT cache keys.

    Official VGGSound caches prefix one or more underscores to some stems.
    Generated audio keeps the original stem, so paired metrics otherwise drop
    those samples silently. Only unique leading-underscore matches are changed.
    """
    import torch

    gt_logits = torch.load(gt_cache / "passt_logits.pth", map_location="cpu", weights_only=True)
    pred_logits = torch.load(prediction_cache / "passt_logits.pth", map_location="cpu", weights_only=True)
    if not isinstance(gt_logits, dict) or not isinstance(pred_logits, dict):
        raise TypeError("PaSST caches must be dictionaries keyed by sample id")

    gt_keys = {str(key) for key in gt_logits}
    pred_keys = {str(key) for key in pred_logits}
    candidates_by_stem: dict[str, list[str]] = {}
    for key in gt_keys:
        candidates_by_stem.setdefault(key.lstrip("_"), []).append(key)

    occupied = pred_keys & gt_keys
    mapping: dict[str, str] = {}
    ambiguous = 0
    for key in sorted(pred_keys - gt_keys):
        candidates = [
            candidate for candidate in candidates_by_stem.get(key.lstrip("_"), []) if candidate not in occupied
        ]
        if len(candidates) == 1:
            mapping[key] = candidates[0]
            occupied.add(candidates[0])
        elif len(candidates) > 1:
            ambiguous += 1

    if mapping:
        for filename in required_files:
            path = prediction_cache / filename
            values = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(values, dict):
                raise TypeError(f"Prediction cache must be a dictionary: {path}")
            aligned = {mapping.get(str(key), str(key)): value for key, value in values.items()}
            if len(aligned) != len(values):
                raise RuntimeError(f"Prediction key alignment created a collision in {path}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(aligned, temporary)
            temporary.replace(path)

    return {
        "exact_before_alignment": len(pred_keys & gt_keys),
        "remapped": len(mapping),
        "ambiguous": ambiguous,
        "unmatched_after_alignment": len(pred_keys) - len(pred_keys & gt_keys) - len(mapping),
    }


def run_benchmark(
    *,
    audio_dir: Path,
    gt_cache: Path,
    prediction_cache: Path,
    output: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    audio_length: float,
    recompute: bool,
    skip_video_related: bool,
    skip_clap: bool,
    align_prediction_keys: bool = False,
) -> dict[str, Any]:
    audio_dir = audio_dir.expanduser().resolve()
    gt_cache = gt_cache.expanduser().resolve()
    prediction_cache = prediction_cache.expanduser().resolve()
    output = output.expanduser().resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Generated audio directory does not exist: {audio_dir}")
    if not gt_cache.is_dir():
        raise FileNotFoundError(f"Ground-truth cache directory does not exist: {gt_cache}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if audio_length <= 0:
        raise ValueError(f"audio_length must be positive, got {audio_length}")

    audio_files = sorted((*audio_dir.glob("*.wav"), *audio_dir.glob("*.flac")))
    if not audio_files:
        raise FileNotFoundError(f"No WAV or FLAC files found in {audio_dir}")
    _validate_gt_cache(gt_cache)
    extract, evaluate = _load_backend()

    required = _required_prediction_files(
        skip_video_related=skip_video_related,
        skip_clap=skip_clap,
    )
    cache_complete = all((prediction_cache / name).is_file() for name in required)
    extraction_ran = recompute or not cache_complete
    if extraction_ran:
        prediction_cache.mkdir(parents=True, exist_ok=True)
        extract(
            audio_path=audio_dir,
            output_path=prediction_cache,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            audio_length=audio_length,
            skip_video_related=skip_video_related,
            skip_clap=skip_clap,
        )

    key_alignment = None
    if align_prediction_keys:
        key_alignment = _align_prediction_cache_keys(
            gt_cache=gt_cache,
            prediction_cache=prediction_cache,
            required_files=required,
        )

    metrics = evaluate(
        gt_audio_cache=gt_cache,
        pred_audio_cache=prediction_cache,
        skip_video_related=skip_video_related,
        skip_clap=skip_clap,
    )
    try:
        backend_version = metadata.version("av_bench")
    except metadata.PackageNotFoundError:
        backend_version = "editable-or-unknown"
    payload = {
        "backend": "hkchengrex/av-benchmark",
        "backend_version": backend_version,
        "audio_dir": str(audio_dir),
        "gt_cache": str(gt_cache),
        "prediction_cache": str(prediction_cache),
        "num_generated_audio": len(audio_files),
        "audio_length": audio_length,
        "extraction_ran": extraction_ran,
        "skip_video_related": skip_video_related,
        "skip_clap": skip_clap,
        "prediction_key_alignment": key_alignment,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        audio_dir=args.audio_dir,
        gt_cache=args.gt_cache,
        prediction_cache=args.prediction_cache,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_length=args.audio_length,
        recompute=args.recompute,
        skip_video_related=args.skip_video_related,
        skip_clap=args.skip_clap,
        align_prediction_keys=args.align_prediction_keys,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))


if __name__ == "__main__":
    main()
