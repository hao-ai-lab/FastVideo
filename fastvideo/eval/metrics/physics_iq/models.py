from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import cv2
import numpy as np


VIEWS = ("perspective-left", "perspective-center", "perspective-right")
DEFAULT_TARGET_FPS = 30
DEFAULT_DURATION_SECONDS = 5
TAKE1_TOKEN = "take-1"
TAKE2_TOKEN = "take-2"


@dataclass(frozen=True)
class PhysicsIQScenario:
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


class PhysicsIQDataLoader:
    """Resolve Physics-IQ dataset metadata and file paths."""

    def __init__(self, dataset_root: str | Path | None = None) -> None:
        root = dataset_root or Path("/root/physics-IQ-benchmark")
        self.repo_root = Path(root).expanduser().resolve()
        self.dataset_dir = self._resolve_dataset_dir(self.repo_root)
        self.descriptions_path = self._resolve_descriptions_path(self.repo_root, self.dataset_dir)
        self.cache_dir = self.repo_root / ".physics_iq_cache"

    @staticmethod
    def _resolve_dataset_dir(repo_root: Path) -> Path:
        nested = repo_root / "physics-IQ-benchmark"
        if nested.exists():
            return nested
        return repo_root

    @staticmethod
    def _resolve_descriptions_path(repo_root: Path, dataset_dir: Path) -> Path:
        candidates = (
            repo_root / "descriptions" / "descriptions.csv",
            dataset_dir / "descriptions" / "descriptions.csv",
        )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError("Could not locate Physics-IQ descriptions/descriptions.csv")

    def read_rows(self) -> list[dict[str, str]]:
        with self.descriptions_path.open("r", newline="") as handle:
            return list(csv.DictReader(handle))

    def take1_rows(self) -> list[dict[str, str]]:
        return [row for row in self.read_rows() if TAKE1_TOKEN in row["scenario"]]

    def take2_rows(self) -> list[dict[str, str]]:
        return [row for row in self.read_rows() if TAKE2_TOKEN in row["scenario"]]

    @staticmethod
    def parse_scenario_filename(filename: str) -> tuple[str, str, str, str]:
        stem = Path(filename).name
        if stem.endswith(".mp4"):
            stem = stem[:-4]
        parts = stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected Physics-IQ filename format: {filename}")
        return parts[0], parts[1], parts[2], "_".join(parts[3:])

    @classmethod
    def scenario_suffix(cls, filename: str) -> str:
        _, view, _, scenario_name = cls.parse_scenario_filename(filename)
        return f"{view}_{scenario_name}"

    def iter_scenarios(
        self,
        *,
        fps: int = DEFAULT_TARGET_FPS,
        generated_dir: str | Path | None = None,
        limit: int | None = None,
    ) -> list[PhysicsIQScenario]:
        rows = self.read_rows()
        take2_by_suffix = {
            self.scenario_suffix(row["scenario"]): row
            for row in rows
            if TAKE2_TOKEN in row["scenario"]
        }
        take1_rows = [row for row in rows if TAKE1_TOKEN in row["scenario"]]
        if limit is not None:
            take1_rows = take1_rows[:limit]

        generated_dir_path = Path(generated_dir).expanduser().resolve() if generated_dir else None
        real_mask_dir = self.resolve_mask_dir(fps, is_real=True)
        scenarios: list[PhysicsIQScenario] = []

        for row in take1_rows:
            scenario_filename = row["scenario"]
            scenario_id, view, _, scenario_name = self.parse_scenario_filename(scenario_filename)
            take2_row = take2_by_suffix.get(self.scenario_suffix(scenario_filename))
            if take2_row is None:
                raise FileNotFoundError(f"Could not find take-2 row matching {scenario_filename}")

            take2_id, _, _, _ = self.parse_scenario_filename(take2_row["scenario"])
            take1_video_path = self.resolve_testing_video_path(
                scenario_id=scenario_id,
                view=view,
                take=TAKE1_TOKEN,
                scenario_name=scenario_name,
                fps=fps,
            )
            take2_video_path = self.resolve_testing_video_path(
                scenario_id=take2_id,
                view=view,
                take=TAKE2_TOKEN,
                scenario_name=scenario_name,
                fps=fps,
            )
            generated_video_path = None
            if generated_dir_path is not None:
                generated_video_path = str(generated_dir_path / row["generated_video_name"])

            scenarios.append(
                PhysicsIQScenario(
                    scenario_id=scenario_id,
                    view=view,
                    scenario_name=scenario_name,
                    take1_video_path=str(take1_video_path),
                    take2_video_path=str(take2_video_path),
                    switch_frame_path=str(
                        self.dataset_dir
                        / "switch-frames"
                        / f"{scenario_id}_switch-frames_anyFPS_{view}_{scenario_name}.jpg"
                    ),
                    caption=row["description"],
                    expected_gen_filename=row["generated_video_name"],
                    generated_video_path=generated_video_path,
                    take1_mask_path=str(
                        real_mask_dir
                        / f"{scenario_id}_video-masks_{fps}FPS_{view}_{TAKE1_TOKEN}_{scenario_name}.mp4"
                    ),
                    take2_mask_path=str(
                        real_mask_dir
                        / f"{take2_id}_video-masks_{fps}FPS_{view}_{TAKE2_TOKEN}_{scenario_name}.mp4"
                    ),
                )
            )
        return scenarios

    def resolve_testing_video_path(
        self,
        *,
        scenario_id: str,
        view: str,
        take: str,
        scenario_name: str,
        fps: int,
    ) -> Path:
        target_dir = self.dataset_dir / "split-videos" / "testing" / f"{fps}FPS"
        target_name = f"{scenario_id}_testing-videos_{fps}FPS_{view}_{take}_{scenario_name}.mp4"
        target_path = target_dir / target_name
        if target_path.exists():
            return target_path

        source_name = f"{scenario_id}_testing-videos_30FPS_{view}_{take}_{scenario_name}.mp4"
        source_path = self.dataset_dir / "split-videos" / "testing" / "30FPS" / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Could not locate Physics-IQ testing video: {target_path}")

        cache_dir = self.cache_dir / "split-videos" / "testing" / f"{fps}FPS"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / target_name
        if not cached_path.exists():
            convert_video_fps(source_path, cached_path, fps_new=fps)
        return cached_path

    def resolve_mask_dir(self, fps: int, *, is_real: bool) -> Path:
        kind = "real" if is_real else "generated"
        mask_dir = self.dataset_dir / "video-masks" / kind
        if is_real:
            mask_dir = mask_dir / f"{fps}FPS"
        return mask_dir


def convert_video_fps(input_path: str | Path, output_path: str | Path, *, fps_new: int) -> None:
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
    subclip_duration = min(DEFAULT_DURATION_SECONDS, duration)

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
    if frame_count_new <= 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps_new,
            (width, height),
        )
        writer.write(frames[0])
        writer.release()
        return

    frames_new: list[np.ndarray] = []
    frame_count_original = len(frames)
    for j in range(frame_count_new):
        alpha = j * (frame_count_original - 1) / (frame_count_new - 1)
        idx = int(alpha)
        alpha -= idx
        frame1 = frames[idx].astype(np.float32)
        frame2 = frames[min(idx + 1, frame_count_original - 1)].astype(np.float32)
        frame_interp = ((1.0 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
        frames_new.append(frame_interp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps_new,
        (width, height),
    )
    for frame in frames_new:
        writer.write(frame)
    writer.release()

