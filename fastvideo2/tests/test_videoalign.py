"""VideoAlign vendor + wrapper tests (CPU, no checkpoint).

Parity argument for the wan2.1 RL reward port: the runtime under
``fastvideo2/rl_rewards/VideoAlign`` is byte-identical to fastvideo
PR #1476 (``maint/pr1476-runtime-compat`` @518aeab0b — the authority), and
the wrapper differs from the PR's ``rewards/videoalign.py`` only by the
vendored-root path and the inlined ``media_to_uint8_array`` (itself verbatim
from the PR's ``rewards/media.py``). Code identity ⇒ behavior identity; the
cluster gate (``gates/videoalign_anchor.py``) then proves the runtime works
end-to-end on the real KwaiVGI/VideoReward checkpoint.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

try:
    import torch
    from fastvideo2.train import videoalign
except ImportError:  # laptop runs keep the byte-parity test only
    torch = None
    videoalign = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch not installed")

_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rl_rewards", "VideoAlign")

# sha256 of each vendored file at PR #1476 @518aeab0b (verified against the
# git blobs at vendor time — see evidence/train_numerics_report.md)
PR_BLOB_SHA256 = {
    "LICENSE": "6169d4ae8a09a5cfcb470f46b44a1e669fe41126875de948d16e3c4cd6830c09",
    "README.md": "053ebd0ed9d06ab2cea59fbc0bef4685c2dbb5dddde38f6e6d7ed4f68c062d52",
    "data.py": "e688c53e9bfb843afc1444d13819ce17f92c0ccf1da1d4eae7503a661454211d",
    "inference.py": "cc37af9e57d96cb84774ebee4da9cc0ab605dc137a052711081b97dd4992b777",
    "prompt_template.py": "655933fa624f53d4a94512a6a512b6c29d31f5aef4d92a28d561f3e710c5c20d",
    "reward_model.py": "16b5d39cc89aa661ba7f4480ea554156911269721f40c1ca219878a636627b07",
    "runtime.py": "0d4c370e407a47632966cc6add4d40da8e07044987810be1f621cc760a12b6f6",
    "vision_process.py": "9b8b183622a27eaa0a0a0b5a9dd8905a6a207bd11fc9e393341a97dfcbf4b6b6",
}


def test_vendored_runtime_matches_pr1476():
    root = _VENDOR_ROOT
    for name, want in PR_BLOB_SHA256.items():
        with open(os.path.join(root, name), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        assert got == want, f"{name} drifted from PR #1476 @518aeab0b"


class _FakeInferencer:
    def __init__(self, score_key: str) -> None:
        self.score_key = score_key
        self.calls: list[tuple[list[str], list[str], bool]] = []

    def reward(self, paths, prompts, *, use_norm):
        self.calls.append((list(paths), list(prompts), use_norm))
        return [{self.score_key: 0.5}]


@pytest.fixture()
def capture_frames(monkeypatch, tmp_path):
    captured: list[np.ndarray] = []

    def save_video(frames: np.ndarray, fps: int = 8) -> str:
        captured.append(frames.copy())
        p = tmp_path / "sample.mp4"
        p.touch()
        return str(p)

    monkeypatch.setattr(videoalign, "_save_video_to_temp", save_video)
    return captured


@requires_torch
@pytest.mark.parametrize(("scorer_name", "score_key"), [
    ("VideoAlignMotionQualityScorer", "MQ"),
    ("VideoAlignVisualQualityScorer", "VQ"),
    ("VideoAlignTextAlignmentScorer", "TA"),
])
def test_scorers_preserve_prompts_and_norm(scorer_name, score_key,
                                           monkeypatch, capture_frames):
    inf = _FakeInferencer(score_key)
    monkeypatch.setattr(videoalign, "_get_inferencer", lambda *_a: inf)
    scores = getattr(videoalign, scorer_name)(device="cpu")(
        torch.zeros(1, 3, 2, 4, 4), ["a red fox runs"])
    assert scores.tolist() == [0.5]
    assert scores.dtype == torch.float32
    assert inf.calls[0][1] == ["a red fox runs"]
    assert inf.calls[0][2] is True


@requires_torch
def test_mq_grayscales_frames(monkeypatch, capture_frames):
    inf = _FakeInferencer("MQ")
    monkeypatch.setattr(videoalign, "_get_inferencer", lambda *_a: inf)
    media = torch.rand(1, 3, 2, 4, 4)
    videoalign.VideoAlignMotionQualityScorer(device="cpu")(media, ["p"])
    frames = capture_frames[0]
    assert frames.dtype == np.uint8
    assert (frames[..., 0] == frames[..., 1]).all()
    assert (frames[..., 1] == frames[..., 2]).all()


@requires_torch
def test_media_to_uint8_layouts():
    for shape in [(1, 3, 4, 4), (1, 4, 4, 3), (1, 3, 2, 4, 4),
                  (1, 2, 3, 4, 4), (1, 2, 4, 4, 3)]:
        out = videoalign.media_to_uint8_array(torch.rand(*shape))
        assert out.dtype == np.uint8
        assert out.shape[-1] == 3
    with pytest.raises(ValueError):
        videoalign.media_to_uint8_array(np.zeros((2, 2)))
