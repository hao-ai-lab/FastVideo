# SPDX-License-Identifier: Apache-2.0
"""Real-weights smoke / reference-comparison scaffold for the Kandinsky6 SR pipeline (CUDA + an official SR repo).

STATUS: scaffold. It needs a CUDA device and a local copy of an official Diffusers SR repo, and skips without them.
Run it once per repo: the flow-matching ``kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers`` and the distilled
``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers`` (the reference comparison needs the reference output of
the same model).  Environment variables:

    KANDINSKY6_SR_BUNDLE           local directory of the official Diffusers SR repo under test
    KANDINSKY6_SR_TEST_VIDEO       a short low-resolution clip (e.g. 16-24 frames at 24 fps, a few hundred pixels wide)
    KANDINSKY6_SR_REFERENCE_VIDEO  optional: the ``kandy-sr from-video`` output for the same clip, scale and seed
    KANDINSKY6_SR_REFERENCE_SCALE  the ``--resolution-scale`` used for that reference run (default 2)
    KANDINSKY6_SR_MIN_PSNR         optional dB floor for the comparison (default 30; NOT calibrated, see README)

Run on a CUDA machine: ``pytest tests/local_tests/kandinsky6_sr/test_kandinsky6_sr_real_weights.py -v -s``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

BUNDLE = os.environ.get("KANDINSKY6_SR_BUNDLE")
VIDEO = os.environ.get("KANDINSKY6_SR_TEST_VIDEO")
REFERENCE_VIDEO = os.environ.get("KANDINSKY6_SR_REFERENCE_VIDEO")
MIN_PSNR = float(os.environ.get("KANDINSKY6_SR_MIN_PSNR", "30"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="real-weights SR validation needs a CUDA device"),
    pytest.mark.skipif(not BUNDLE or not Path(BUNDLE or "").is_dir(),
                       reason="set KANDINSKY6_SR_BUNDLE to a local copy of the official Diffusers SR repo"),
    pytest.mark.skipif(not VIDEO or not Path(VIDEO or "").is_file(),
                       reason="set KANDINSKY6_SR_TEST_VIDEO to a short low-resolution clip"),
]


def decode_rgb(path) -> np.ndarray:
    """All frames of a video as a ``[T, H, W, 3]`` uint8 array."""
    av = pytest.importorskip("av")
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def per_frame_psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    mse = ((a.astype(np.float64) - b.astype(np.float64))**2).mean(axis=(1, 2, 3))
    return 10.0 * np.log10(255.0**2 / np.maximum(mse, 1e-10))


@pytest.fixture(scope="module")
def generator():
    from fastvideo import VideoGenerator

    # Strict loading of the DiT / KVAE / latent-upscaler weights happens here: a missing or unexpected key fails
    # from_pretrained instead of producing garbage later.
    gen = VideoGenerator.from_pretrained(
        BUNDLE,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
    )
    yield gen
    gen.shutdown()


def _sr(generator, out_dir: Path, **extensions):
    return generator.generate({
        "inputs": {
            "video_path": VIDEO
        },
        "output": {
            "output_path": str(out_dir),
            "save_video": True,
            "return_frames": False,
        },
        "extensions": extensions,
    })


@pytest.mark.parametrize("scale", [2, 2.25])
def test_sr_generates_a_decodable_video_of_the_expected_geometry(generator, tmp_path, scale):
    from fastvideo.pipelines.basic.kandinsky6_sr import sr_io

    result = _sr(generator, tmp_path, sr_resolution_scale=scale)
    assert result.video_path and Path(result.video_path).is_file()

    source = decode_rgb(VIDEO)
    out = decode_rgb(result.video_path)
    expected_frames = sr_io.read_video(VIDEO)[0].shape[0]  # fps rule + 1+8k alignment of the request contract
    assert out.shape[0] == expected_frames
    assert out.shape[1] >= int(source.shape[1] * scale) - 16 and out.shape[2] >= int(source.shape[2] * scale) - 16
    assert out.dtype == np.uint8 and out.std() > 1.0, "output is constant / empty"


def test_sr_is_deterministic_for_a_fixed_seed(generator, tmp_path):
    first = decode_rgb(_sr(generator, tmp_path / "a", sr_resolution_scale=2).video_path)
    second = decode_rgb(_sr(generator, tmp_path / "b", sr_resolution_scale=2).video_path)
    # Same seed, same clip: drift here is sampling nondeterminism (the 40 dB floor is a starting value, uncalibrated).
    assert per_frame_psnr(first, second).min() > 40.0


@pytest.mark.skipif(not REFERENCE_VIDEO or not Path(REFERENCE_VIDEO or "").is_file(),
                    reason="set KANDINSKY6_SR_REFERENCE_VIDEO to the kandy-sr output for the same clip")
def test_sr_is_close_to_the_reference_output(generator, tmp_path):
    scale = float(os.environ.get("KANDINSKY6_SR_REFERENCE_SCALE", "2"))
    ours = decode_rgb(_sr(generator, tmp_path, sr_resolution_scale=scale).video_path)
    reference = decode_rgb(REFERENCE_VIDEO)
    psnr = per_frame_psnr(ours, reference)
    print(f"PSNR vs reference: mean {psnr.mean():.2f} dB, min {psnr.min():.2f} dB")
    assert psnr.mean() >= MIN_PSNR
