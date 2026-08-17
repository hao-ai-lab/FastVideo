# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the vendored TAEHV decoder helper."""

import re

import pytest

from fastvideo.mlx_runtime import wan_vae
from fastvideo.mlx_runtime.taehv_decode import TAEW2_1_CHECKPOINT_SHA256, ensure_taew2_1_checkpoint


def test_taehv_checkpoint_pin_is_a_sha256() -> None:
    assert len(TAEW2_1_CHECKPOINT_SHA256) == 64
    assert int(TAEW2_1_CHECKPOINT_SHA256, 16) >= 0


def test_explicit_taehv_checkpoint_path_is_not_downloaded(tmp_path) -> None:
    checkpoint = tmp_path / "custom-taehv.pth"
    checkpoint.write_bytes(b"local test checkpoint")
    assert ensure_taew2_1_checkpoint(checkpoint) == checkpoint


def test_taew2_2_checkpoint_pin_is_immutable_and_sha256() -> None:
    assert "/main/" not in wan_vae.TAEW2_2_URL
    assert re.search(r"/[0-9a-f]{40}/taew2_2\.pth$", wan_vae.TAEW2_2_URL)
    assert re.fullmatch(r"[0-9a-f]{64}", wan_vae.TAEW2_2_SHA256)


def test_cached_taew2_2_checkpoint_is_verified(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(wan_vae, "_cache_dir", lambda: tmp_path)
    (tmp_path / "taew2_2.pth").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="failed sha256 verification"):
        wan_vae.ensure_taehv_checkpoint(z_dim=48)
