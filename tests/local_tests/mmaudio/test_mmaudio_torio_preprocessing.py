# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity for FastVideo's isolated torio preprocessing backend."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO = Path(
    os.environ.get("MMAUDIO_OFFICIAL_REPO", REPO_ROOT.parent / "MMAudio")
)


@pytest.mark.skipif(
    not os.environ.get("MMAUDIO_TORIO_TEST_VIDEO"),
    reason="Set MMAUDIO_TORIO_TEST_VIDEO to run real media parity.",
)
def test_fastvideo_torio_reader_matches_official_vggsound_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torio.io")
    if not OFFICIAL_REPO.is_dir():
        pytest.skip(f"Official MMAudio checkout is missing: {OFFICIAL_REPO}")

    video_path = Path(os.environ["MMAUDIO_TORIO_TEST_VIDEO"])
    if not video_path.is_file():
        pytest.skip(f"Test video is missing: {video_path}")

    monkeypatch.syspath_prepend(str(OFFICIAL_REPO))
    dist_utils = types.ModuleType("mmaudio.utils.dist_utils")
    dist_utils.local_rank = 0
    monkeypatch.setitem(sys.modules, "mmaudio.utils.dist_utils", dist_utils)

    from mmaudio.data.extraction.vgg_sound import VGGSound

    video_id = video_path.stem
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / f"{video_id}.mp4").symlink_to(video_path.resolve())
    manifest = tmp_path / "captions.tsv"
    manifest.write_text(
        f"id\tlabel\n{video_id}\tA reference preprocessing sample.\n",
        encoding="utf-8",
    )

    official = VGGSound(
        videos,
        tsv_path=manifest,
        sample_rate=44_100,
        duration_sec=8.0,
        audio_samples=353_280,
        normalize_audio=True,
    ).sample(0)

    from fastvideo.pipelines.preprocess.mmaudio.torio_media_reader import (
        preprocess_mmaudio_media_with_torio,
    )

    audio, clip, sync, duration = preprocess_mmaudio_media_with_torio(
        video_path,
        duration_s=8.0,
        target_sample_rate=44_100,
        target_samples=353_280,
        normalize_audio=True,
    )

    torch.testing.assert_close(audio, official["audio"], atol=0, rtol=0)
    torch.testing.assert_close(clip, official["clip_video"], atol=0, rtol=0)
    torch.testing.assert_close(sync, official["sync_video"], atol=0, rtol=0)
    assert duration == 8.0
