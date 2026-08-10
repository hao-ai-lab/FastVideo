# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the MLX prompt-to-video decode dispatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_prompt_to_video_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "examples/inference/basic/mlx_wan_prompt_to_video.py"
    spec = importlib.util.spec_from_file_location(
        "mlx_wan_prompt_to_video_for_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_taehv_decode_uses_mlx_backend(monkeypatch, tmp_path):
    module = _load_prompt_to_video_module()
    import fastvideo.mlx_runtime.wan_vae as wan_vae

    calls = []

    def fake_decode_latents_to_video(latents_np, output_path, **kwargs):
        calls.append((latents_np, output_path, kwargs))
        output_path.write_bytes(b"fake mp4")
        return {"backend": kwargs["backend"]}

    monkeypatch.setattr(
        wan_vae, "decode_latents_to_video", fake_decode_latents_to_video
    )
    latents = np.zeros((1, 16, 1, 2, 2), dtype=np.float32)
    output_path = tmp_path / "out.mp4"

    module.decode_latents_to_video(
        model_root=tmp_path,
        latents_np=latents,
        output_path=output_path,
        fps=16,
        device_arg="cpu",
        dtype_arg="fp32",
        backend="taehv",
        taehv_source_path=None,
        taehv_checkpoint_path=None,
        taehv_parallel=False,
    )

    assert output_path.read_bytes() == b"fake mp4"
    assert len(calls) == 1
    _, called_output_path, kwargs = calls[0]
    assert called_output_path == output_path
    assert kwargs["backend"] == "taehv"
    assert kwargs["z_dim"] == 16


def test_vendored_taehv_import_available():
    from fastvideo.third_party.taehv import TAEHV

    assert TAEHV.__name__ == "TAEHV"
