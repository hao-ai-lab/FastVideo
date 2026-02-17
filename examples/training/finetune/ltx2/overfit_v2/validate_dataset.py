#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    ok: bool
    summary: dict[str, Any]
    errors: list[str]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("videos2caption.json must contain a list.")
    return payload


def _validate_raw_dataset(data_root: Path) -> ValidationResult:
    errors: list[str] = []
    videos_dir = data_root / "videos"
    manifest_path = data_root / "videos2caption.json"

    if not videos_dir.exists():
        errors.append(f"Missing videos directory: {videos_dir}")
    if not manifest_path.exists():
        errors.append(f"Missing manifest: {manifest_path}")
    if errors:
        return ValidationResult(ok=False, summary={}, errors=errors)

    try:
        manifest = _load_manifest(manifest_path)
    except Exception as e:
        return ValidationResult(
            ok=False, summary={}, errors=[f"Failed to load manifest: {e}"])

    missing_video_paths: list[str] = []
    malformed_items = 0
    invalid_caption_items = 0

    for idx, item in enumerate(manifest):
        if not isinstance(item, dict):
            malformed_items += 1
            continue

        name = item.get("path")
        cap = item.get("cap")
        fps = item.get("fps")
        num_frames = item.get("num_frames")
        resolution = item.get("resolution")

        if not isinstance(name, str) or not name:
            malformed_items += 1
            continue
        if (not isinstance(cap, list) or not cap or
                not isinstance(cap[0], str) or not cap[0].strip()):
            invalid_caption_items += 1
        if not isinstance(fps, (int, float)) or fps <= 0:
            malformed_items += 1
        if not isinstance(num_frames, int) or num_frames <= 0:
            malformed_items += 1
        if not isinstance(resolution, dict):
            malformed_items += 1
        else:
            w = resolution.get("width")
            h = resolution.get("height")
            if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
                malformed_items += 1

        video_path = videos_dir / name
        if not video_path.exists():
            missing_video_paths.append(name)

    summary = {
        "data_root": str(data_root),
        "videos_dir": str(videos_dir),
        "manifest_path": str(manifest_path),
        "manifest_entries": len(manifest),
        "missing_video_count": len(missing_video_paths),
        "malformed_item_count": malformed_items,
        "invalid_caption_count": invalid_caption_items,
    }
    if missing_video_paths:
        summary["missing_videos_preview"] = missing_video_paths[:20]

    if len(manifest) == 0:
        errors.append("Manifest is empty.")
    if missing_video_paths:
        errors.append(
            f"Found {len(missing_video_paths)} manifest entries with missing video files.")
    if malformed_items > 0:
        errors.append(f"Found {malformed_items} malformed manifest entries.")
    if invalid_caption_items > 0:
        errors.append(
            f"Found {invalid_caption_items} entries with invalid caption payload.")

    return ValidationResult(ok=len(errors) == 0, summary=summary, errors=errors)


def _find_precomputed_root(data_root: Path) -> Path:
    precomputed = data_root / ".precomputed"
    if precomputed.exists():
        return precomputed
    return data_root


def _pt_rel_paths(dir_path: Path) -> set[str]:
    return {str(p.relative_to(dir_path)) for p in dir_path.rglob("*.pt")}


def _validate_precomputed_dataset(data_root: Path) -> ValidationResult:
    errors: list[str] = []
    precomputed_root = _find_precomputed_root(data_root)

    latents_dir = precomputed_root / "latents"
    conditions_dir = precomputed_root / "conditions"
    audio_dir = precomputed_root / "audio_latents"

    if not latents_dir.exists():
        errors.append(f"Missing latents directory: {latents_dir}")
    if not conditions_dir.exists():
        errors.append(f"Missing conditions directory: {conditions_dir}")
    if errors:
        return ValidationResult(ok=False, summary={}, errors=errors)

    latents = _pt_rel_paths(latents_dir)
    conditions = _pt_rel_paths(conditions_dir)
    has_audio_dir = audio_dir.exists()
    audio = _pt_rel_paths(audio_dir) if has_audio_dir else set()

    summary = {
        "data_root": str(data_root),
        "precomputed_root": str(precomputed_root),
        "latents_count": len(latents),
        "conditions_count": len(conditions),
        "audio_latents_count": len(audio),
        "audio_enabled": has_audio_dir,
    }

    if len(latents) == 0:
        errors.append("No .pt files found in latents.")
    if len(conditions) == 0:
        errors.append("No .pt files found in conditions.")

    missing_conditions = sorted(latents - conditions)
    missing_latents = sorted(conditions - latents)
    if missing_conditions:
        errors.append(
            f"Found {len(missing_conditions)} latents without matching conditions.")
        summary["missing_conditions_preview"] = missing_conditions[:20]
    if missing_latents:
        errors.append(
            f"Found {len(missing_latents)} conditions without matching latents.")
        summary["missing_latents_preview"] = missing_latents[:20]

    if has_audio_dir:
        missing_audio = sorted(latents - audio)
        if missing_audio:
            errors.append(
                f"Found {len(missing_audio)} latents without matching audio_latents.")
            summary["missing_audio_preview"] = missing_audio[:20]

    return ValidationResult(ok=len(errors) == 0, summary=summary, errors=errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw or precomputed LTX2 overfit dataset.")
    parser.add_argument("--mode",
                        choices=["raw", "precomputed"],
                        required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()

    if args.mode == "raw":
        result = _validate_raw_dataset(data_root)
    else:
        result = _validate_precomputed_dataset(data_root)

    payload = {
        "mode": args.mode,
        "ok": result.ok,
        "summary": result.summary,
        "errors": result.errors,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
