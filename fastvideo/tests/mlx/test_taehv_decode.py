# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the vendored TAEHV decoder helper."""

from fastvideo.mlx_runtime.taehv_decode import TAEW2_1_CHECKPOINT_SHA256, ensure_taew2_1_checkpoint


def test_taehv_checkpoint_pin_is_a_sha256() -> None:
    assert len(TAEW2_1_CHECKPOINT_SHA256) == 64
    assert int(TAEW2_1_CHECKPOINT_SHA256, 16) >= 0


def test_explicit_taehv_checkpoint_path_is_not_downloaded(tmp_path) -> None:
    checkpoint = tmp_path / "custom-taehv.pth"
    checkpoint.write_bytes(b"local test checkpoint")
    assert ensure_taew2_1_checkpoint(checkpoint) == checkpoint
