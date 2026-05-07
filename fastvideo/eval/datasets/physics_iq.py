"""Physics-IQ benchmark prompt corpus.

Walks the on-disk Physics-IQ release (descriptions CSV + per-take MP4
files + per-take real motion masks) and yields one sample dict per
take-1 scenario, paired with its take-2 reference and (when available)
both takes' real motion-mask MP4s.

Each sample dict is shaped to drop straight into
:meth:`fastvideo.eval.Evaluator.evaluate` for the ``physics_iq`` metric:

    {
        "prompt": <description>,
        "reference":            "<take-1 mp4>",
        "reference_take2":      "<take-2 mp4>",
        "reference_mask":       "<take-1 mask mp4>"  (if real masks ship),
        "reference_take2_mask": "<take-2 mask mp4>",
        "scenario": <scenario_id>,
        "view": <camera view>,
        "auxiliary_info": { ... metadata ... },
    }

The metric's :func:`fastvideo.eval.metrics.physics_iq.utils.unpack_batch_value`
already accepts ``(str, Path)`` source types, so passing these paths in
is enough — no preloading is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import cv2
import numpy as np

from fastvideo.eval.datasets.base import PromptDataset
from fastvideo.eval.datasets.registry import register_dataset

VIEWS = ("perspective-left", "perspective-center", "perspective-right")
TAKE1_TOKEN = "take-1"
TAKE2_TOKEN = "take-2"

# FPS the dataset rows should resolve to. Source release is recorded at
# 30 FPS; if a different value is requested the loader transcodes once
# into a per-repo-root cache directory.
_DEFAULT_FPS = 30
_DEFAULT_DURATION_SECONDS = 5


@dataclass(frozen=True)
class PhysicsIQScenario:
    """One row of the Physics-IQ manifest, fully resolved on disk."""
    scenario_id: str
    view: str
    scenario_name: str
    take1_video_path: str
    take2_video_path: str
    switch_frame_path: str
    caption: str
    expected_gen_filename: str
    generated_video_path: str | None = None
    take1_mask_path: str | None = None
    take2_mask_path: str | None = None


@register_dataset("physics_iq")
class PhysicsIQPromptDataset(PromptDataset):
    """Physics-IQ benchmark prompt corpus.

    Args:
        dataset_root: path to a checkout of the Physics-IQ release
            (the directory containing ``descriptions/descriptions.csv``
            and ``split-videos/``). Defaults to ``/root/physics-IQ-benchmark``,
            matching the release's docker layout.
        fps: target frame rate. The release ships at 30 FPS; other rates
            transcode once on first access into ``<root>/.physics_iq_cache/``.
        limit: optional truncation for quick smoke runs.
        generated_dir: optional directory of pre-generated videos —
            attaches each manifest row's expected output path to the
            sample dict under ``auxiliary_info["generated_video_path"]``.
    """

    description = ("Physics-IQ benchmark, 396 take-1 scenarios across 66 unique physics "
                   "setups × 3 perspective views, each paired with a take-2 reference.")
    requires_reference_video = True

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        fps: int = _DEFAULT_FPS,
        limit: int | None = None,
        generated_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        repo_root = Path(dataset_root or "/root/physics-IQ-benchmark").expanduser().resolve()
        self.repo_root = repo_root
        self.dataset_dir = _resolve_dataset_dir(repo_root)
        self.descriptions_path = _resolve_descriptions_path(repo_root, self.dataset_dir)
        self.cache_dir = repo_root / ".physics_iq_cache"
        self.fps = fps

        scenarios = self._iter_scenarios(
            fps=fps,
            generated_dir=generated_dir,
            limit=limit,
        )
        self._rows = [_scenario_to_row(s) for s in scenarios]

    def _iter_scenarios(
        self,
        *,
        fps: int,
        generated_dir: str | Path | None,
        limit: int | None,
    ) -> list[PhysicsIQScenario]:
        with self.descriptions_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))

        take2_by_suffix = {_scenario_suffix(row["scenario"]): row for row in rows if TAKE2_TOKEN in row["scenario"]}
        take1_rows = [row for row in rows if TAKE1_TOKEN in row["scenario"]]
        if limit is not None:
            take1_rows = take1_rows[:limit]

        generated_dir_path = (Path(generated_dir).expanduser().resolve() if generated_dir else None)
        real_mask_dir = self._resolve_mask_dir(fps, is_real=True)
        scenarios: list[PhysicsIQScenario] = []

        for row in take1_rows:
            scenario_filename = row["scenario"]
            scenario_id, view, _, scenario_name = _parse_scenario_filename(scenario_filename)
            take2_row = take2_by_suffix.get(_scenario_suffix(scenario_filename))
            if take2_row is None:
                raise FileNotFoundError(f"Could not find take-2 row matching {scenario_filename}")
            take2_id, _, _, _ = _parse_scenario_filename(take2_row["scenario"])

            take1_video_path = self._resolve_testing_video_path(
                scenario_id=scenario_id,
                view=view,
                take=TAKE1_TOKEN,
                scenario_name=scenario_name,
                fps=fps,
            )
            take2_video_path = self._resolve_testing_video_path(
                scenario_id=take2_id,
                view=view,
                take=TAKE2_TOKEN,
                scenario_name=scenario_name,
                fps=fps,
            )
            generated_video_path = (str(generated_dir_path /
                                        row["generated_video_name"]) if generated_dir_path is not None else None)

            scenarios.append(
                PhysicsIQScenario(
                    scenario_id=scenario_id,
                    view=view,
                    scenario_name=scenario_name,
                    take1_video_path=str(take1_video_path),
                    take2_video_path=str(take2_video_path),
                    switch_frame_path=str(self.dataset_dir / "switch-frames" /
                                          f"{scenario_id}_switch-frames_anyFPS_{view}_{scenario_name}.jpg"),
                    caption=row["description"],
                    expected_gen_filename=row["generated_video_name"],
                    generated_video_path=generated_video_path,
                    take1_mask_path=str(real_mask_dir /
                                        f"{scenario_id}_video-masks_{fps}FPS_{view}_{TAKE1_TOKEN}_{scenario_name}.mp4"),
                    take2_mask_path=str(real_mask_dir /
                                        f"{take2_id}_video-masks_{fps}FPS_{view}_{TAKE2_TOKEN}_{scenario_name}.mp4"),
                ))
        return scenarios

    def _resolve_testing_video_path(
        self,
        *,
        scenario_id: str,
        view: str,
        take: str,
        scenario_name: str,
        fps: int,
    ) -> Path:
        target_dir = self.dataset_dir / "split-videos" / "testing" / f"{fps}FPS"
        target_name = (f"{scenario_id}_testing-videos_{fps}FPS_{view}_{take}_{scenario_name}.mp4")
        target_path = target_dir / target_name
        if target_path.exists():
            return target_path

        # FPS-convert from the 30-FPS source release on miss; cache under
        # ``.physics_iq_cache`` so repeat runs are free.
        source_name = (f"{scenario_id}_testing-videos_30FPS_{view}_{take}_{scenario_name}.mp4")
        source_path = self.dataset_dir / "split-videos" / "testing" / "30FPS" / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Could not locate Physics-IQ testing video: {target_path}")

        cache_dir = self.cache_dir / "split-videos" / "testing" / f"{fps}FPS"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / target_name
        if not cached_path.exists():
            _convert_video_fps(source_path, cached_path, fps_new=fps)
        return cached_path

    def _resolve_mask_dir(self, fps: int, *, is_real: bool) -> Path:
        kind = "real" if is_real else "generated"
        mask_dir = self.dataset_dir / "video-masks" / kind
        if is_real:
            mask_dir = mask_dir / f"{fps}FPS"
        return mask_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_dataset_dir(repo_root: Path) -> Path:
    nested = repo_root / "physics-IQ-benchmark"
    if nested.exists():
        return nested
    return repo_root


def _resolve_descriptions_path(repo_root: Path, dataset_dir: Path) -> Path:
    candidates = (
        repo_root / "descriptions" / "descriptions.csv",
        dataset_dir / "descriptions" / "descriptions.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate Physics-IQ descriptions/descriptions.csv")


def _parse_scenario_filename(filename: str) -> tuple[str, str, str, str]:
    stem = Path(filename).name
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected Physics-IQ filename format: {filename}")
    return parts[0], parts[1], parts[2], "_".join(parts[3:])


def _scenario_suffix(filename: str) -> str:
    _, view, _, scenario_name = _parse_scenario_filename(filename)
    return f"{view}_{scenario_name}"


def _scenario_to_row(scenario: PhysicsIQScenario) -> dict:
    """Flatten a :class:`PhysicsIQScenario` into the public sample-dict shape."""
    aux: dict = {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.scenario_name,
        "switch_frame_path": scenario.switch_frame_path,
        "expected_gen_filename": scenario.expected_gen_filename,
    }
    if scenario.generated_video_path is not None:
        aux["generated_video_path"] = scenario.generated_video_path

    row: dict = {
        "prompt": scenario.caption,
        "reference": scenario.take1_video_path,
        "reference_take2": scenario.take2_video_path,
        "scenario": scenario.scenario_id,
        "view": scenario.view,
        "auxiliary_info": aux,
    }
    if scenario.take1_mask_path is not None:
        row["reference_mask"] = scenario.take1_mask_path
    if scenario.take2_mask_path is not None:
        row["reference_take2_mask"] = scenario.take2_mask_path
    return row


def _convert_video_fps(input_path: str | Path, output_path: str | Path, *, fps_new: int) -> None:
    """Trim *input_path* to ``_DEFAULT_DURATION_SECONDS`` and re-encode at
    *fps_new*, writing the result to *output_path*. Used to materialize
    Physics-IQ's 30-FPS source release at user-requested rates.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for FPS conversion: {input_path}")

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps_original if fps_original else 0.0
    width, height = width - width % 2, height - height % 2
    subclip_duration = min(_DEFAULT_DURATION_SECONDS, duration)

    frames: list[np.ndarray] = []
    for _ in range(int(subclip_duration * fps_original)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {input_path}")

    frame_count_new = int(subclip_duration * fps_new)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps_new,
        (width, height),
    )
    if frame_count_new <= 1:
        writer.write(frames[0])
        writer.release()
        return

    frame_count_original = len(frames)
    for j in range(frame_count_new):
        alpha = j * (frame_count_original - 1) / (frame_count_new - 1)
        idx = int(alpha)
        alpha -= idx
        f1 = frames[idx].astype(np.float32)
        f2 = frames[min(idx + 1, frame_count_original - 1)].astype(np.float32)
        writer.write(((1.0 - alpha) * f1 + alpha * f2).astype(np.uint8))
    writer.release()
