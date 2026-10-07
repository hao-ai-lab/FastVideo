# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Apple-Silicon MLX RIFE interpolation backend."""

from __future__ import annotations

import numpy as np
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

mx = pytest.importorskip("mlx.core", reason="MLX required for RIFE tests")


@pytest.mark.skipif(
    not bool(getattr(mx, "metal", None) and mx.metal.is_available()),
    reason="RIFE MLX regression requires Apple Silicon Metal",
)
def test_rife_interpolation_preserves_keyframes_shape_and_count() -> None:
    from fastvideo.mlx_runtime.rife_interp import RIFEWeightsUnavailableError, interpolate, load_model

    frame0 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1 = np.zeros((64, 96, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255
    try:
        # The backend is vendored, but its weights are fetched from Hugging Face
        # on first use, so an offline machine should skip rather than fail.
        model = load_model()
    except RIFEWeightsUnavailableError as exc:
        pytest.skip(f"RIFE weights unavailable: {exc}")

    frames = interpolate([frame0, frame1], factor=2, model=model)

    assert len(frames) == 3
    assert frames[1].shape == frame0.shape
    assert frames[1].dtype == np.uint8
    np.testing.assert_array_equal(frames[0], frame0)
    np.testing.assert_array_equal(frames[-1], frame1)


def test_rife_download_unavailable_has_specific_error(monkeypatch) -> None:
    from fastvideo.mlx_runtime.rife_interp import RIFEWeightsUnavailableError, load_model
    from fastvideo.third_party.rife_mlx.utils import weights

    def unavailable(*args, **kwargs):
        raise LocalEntryNotFoundError("offline")

    monkeypatch.setattr(weights, "build_model", unavailable)
    load_model.cache_clear()
    with pytest.raises(RIFEWeightsUnavailableError):
        load_model()


def test_rife_backend_regression_is_not_skip_eligible(monkeypatch) -> None:
    from fastvideo.mlx_runtime.rife_interp import RIFEBackendError, RIFEWeightsUnavailableError, load_model
    from fastvideo.third_party.rife_mlx.utils import weights

    def broken(*args, **kwargs):
        raise AssertionError("backend regression")

    monkeypatch.setattr(weights, "build_model", broken)
    load_model.cache_clear()
    with pytest.raises(RIFEBackendError) as caught:
        load_model()
    assert not isinstance(caught.value, RIFEWeightsUnavailableError)
