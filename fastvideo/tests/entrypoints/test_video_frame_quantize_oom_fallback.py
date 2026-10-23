# SPDX-License-Identifier: Apache-2.0
"""``_quantize_video_frames_to_uint8`` and its CUDA-OOM fallback.

Found on a real GPU run of the K6 T2IVA-distilled example (30B params, ``dit_cpu_offload=False``): the worker process
keeps the transformer resident on the device after denoising and decode, so the client process's on-device quantize
(``fastvideo/entrypoints/video_generator.py``) can be the allocation that finally OOMs, even though the data it needs
(``src``, IPC-shared from the worker) is already there and only needs a device->host copy. These tests force that path
on CPU (no GPU here) by making the on-device arithmetic raise ``torch.cuda.OutOfMemoryError`` once, and check the
fallback produces the identical result instead of propagating the error.
"""
from __future__ import annotations

import logging

import pytest
import torch

from fastvideo.entrypoints.video_generator import _quantize_video_frames_to_uint8


def _reference(src: torch.Tensor) -> torch.Tensor:
    """Plain (non-fused) equivalent of the function under test, for comparison."""
    return ((src.float() * 255).clamp(0, 255).to(torch.uint8)).permute(2, 0, 1, 3, 4)


@pytest.fixture()
def sample_video() -> torch.Tensor:
    # [b, c, t, h, w]; includes values outside [0, 1] to exercise clamp_().
    torch.manual_seed(0)
    return torch.rand(2, 3, 4, 5, 6) * 1.3 - 0.15


def test_happy_path_matches_the_plain_computation(sample_video):
    out = _quantize_video_frames_to_uint8(sample_video.clone())
    assert out.shape == (sample_video.shape[2], sample_video.shape[0], sample_video.shape[1], sample_video.shape[3],
                         sample_video.shape[4])
    assert out.dtype == torch.uint8
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, _reference(sample_video), rtol=0, atol=0)


def test_out_of_range_values_are_clamped_not_wrapped(sample_video):
    src = sample_video.clone()
    src[0, 0, 0, 0, 0] = 5.0  # would wrap past 255 without clamp_()
    src[0, 0, 0, 0, 1] = -5.0
    out = _quantize_video_frames_to_uint8(src)
    assert out[0, 0, 0, 0, 0].item() == 255
    assert out[0, 0, 0, 0, 1].item() == 0


def test_cuda_oom_on_the_fast_path_falls_back_to_a_cpu_quantize(sample_video, monkeypatch, caplog):
    calls = {"n": 0}
    real_clamp_ = torch.Tensor.clamp_

    def flaky_clamp_(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated: device holds the resident DiT, nothing left to allocate")
        return real_clamp_(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clamp_", flaky_clamp_)
    with caplog.at_level(logging.WARNING):
        out = _quantize_video_frames_to_uint8(sample_video.clone())

    assert calls["n"] == 2  # the failed fast-path attempt, then the fallback
    assert out.dtype == torch.uint8 and out.device.type == "cpu"
    torch.testing.assert_close(out, _reference(sample_video), rtol=0, atol=0)
    assert any("out of memory" in message.lower() for message in caplog.messages)


def test_a_non_oom_runtime_error_is_not_swallowed(sample_video, monkeypatch):
    def always_fails(self, *args, **kwargs):
        raise RuntimeError("some unrelated failure")

    monkeypatch.setattr(torch.Tensor, "clamp_", always_fails)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        _quantize_video_frames_to_uint8(sample_video.clone())
