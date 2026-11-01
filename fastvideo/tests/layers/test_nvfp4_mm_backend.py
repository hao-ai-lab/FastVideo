# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.layers.quantization import nvfp4_config


@pytest.mark.parametrize(("profile", "sm100", "expected"), [
    ("h3_dit", True, "cudnn"),
    ("h3_dit_ffn", True, "cudnn"),
    ("h3_dit_vsa", True, "cudnn"),
    ("refine", True, "auto"),
    (None, True, "auto"),
    ("h3_dit", False, "auto"),
])
def test_auto_backend_resolution(monkeypatch, profile, sm100, expected) -> None:
    monkeypatch.delenv("FASTVIDEO_NVFP4_MM_BACKEND", raising=False)
    monkeypatch.setattr(nvfp4_config, "_is_sm100", lambda: sm100)
    assert nvfp4_config._mm_fp4_backend(profile) == expected


def test_explicit_backend_wins(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_NVFP4_MM_BACKEND", "cutlass")
    monkeypatch.setattr(nvfp4_config, "_is_sm100", lambda: True)
    assert nvfp4_config._mm_fp4_backend("h3_dit") == "cutlass"
