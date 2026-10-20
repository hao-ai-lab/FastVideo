# SPDX-License-Identifier: Apache-2.0
"""Raw VGGSound metadata adapter for V2A preprocessing."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from torch.utils.data import Dataset

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class VGGSoundDataset(Dataset):
    """Map VGGSound metadata rows to local MP4 paths and captions.

    ``Loie/VGGSound`` stores clips as ``<youtube-id>_<start:06d>.mp4``.
    The downloaded tar archives must be extracted before random-access GPU
    preprocessing; repeatedly seeking inside gzip archives is prohibitively
    expensive for a shuffled training dataset.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        metadata_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        metadata = (Path(metadata_path).expanduser().resolve() if metadata_path is not None else self.root /
                    "vggsound.csv")
        if not metadata.is_file():
            raise FileNotFoundError(f"VGGSound metadata file does not exist: {metadata}")

        candidates = (self.root / "videos", self.root / "video", self.root)
        self.video_root = next((path for path in candidates if path.is_dir()), self.root)
        self.samples: list[tuple[str, str, Path]] = []
        if metadata.suffix.lower() == ".tsv":
            self._read_caption_manifest(metadata, split=split)
        else:
            self._read_vggsound_csv(metadata, split=split)

        if not self.samples:
            raise ValueError(f"No VGGSound samples found for split {split!r} in {metadata}")

    def _read_caption_manifest(self, metadata: Path, *, split: str) -> None:
        seen_ids: set[str] = set()
        missing_videos = 0
        available_videos = {
            Path(name).stem
            for name in os.listdir(self.video_root)
            if Path(name).suffix.lower() == ".mp4"
        }
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required_columns = {"id", "label"}
            columns = set(reader.fieldnames or ())
            if not required_columns.issubset(columns):
                raise ValueError(
                    f"VGGSound caption manifest {metadata} must contain columns {sorted(required_columns)}, "
                    f"got {sorted(columns)}")
            for row_number, row in enumerate(reader, start=2):
                sample_id = str(row.get("id") or "").strip()
                caption = row.get("label")
                if not sample_id:
                    raise ValueError(f"VGGSound caption manifest has an empty id at row {row_number}")
                if caption is None or not caption.strip():
                    raise ValueError(f"VGGSound caption manifest has an empty label at row {row_number}")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate VGGSound id {sample_id!r} at row {row_number}")
                seen_ids.add(sample_id)
                video_path = self.video_root / f"{sample_id}.mp4"
                if sample_id not in available_videos:
                    missing_videos += 1
                    continue
                self.samples.append((sample_id, caption, video_path))
        logger.info(
            "Loaded %d VGGSound %s captions from %s (%d videos missing)",
            len(self.samples),
            split,
            metadata,
            missing_videos,
        )

    def _read_vggsound_csv(self, metadata: Path, *, split: str) -> None:
        with metadata.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 4:
                    raise ValueError(f"VGGSound CSV row {row_number} must have four columns, got {row!r}")
                youtube_id, start_seconds, caption, row_split = row
                if row_split.strip().lower() != split.lower():
                    continue
                try:
                    start = int(start_seconds)
                except ValueError as exc:
                    raise ValueError(f"Invalid VGGSound start time at row {row_number}: {start_seconds!r}") from exc
                sample_id = f"{youtube_id}_{start:06d}"
                self.samples.append((sample_id, caption, self.video_root / f"{sample_id}.mp4"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        sample_id, caption, path = self.samples[index]
        return {
            "id": sample_id,
            "caption": caption,
            "video_path": str(path),
        }


__all__ = ["VGGSoundDataset"]
