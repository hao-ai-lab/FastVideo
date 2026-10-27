# SPDX-License-Identifier: Apache-2.0
"""Regression test for the Wan2.2 MLX/MPS memory handoff."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import weakref
from pathlib import Path

import numpy as np


def _load_wan22_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "examples/inference/basic/mlx_wan22_generate.py"
    spec = importlib.util.spec_from_file_location("mlx_wan22_generate_memory_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wan22_releases_compiled_dit_before_torch_decode_and_mlx_rife(monkeypatch, tmp_path: Path) -> None:
    import mlx.core as mx
    import torch
    from diffusers import utils as diffusers_utils

    from examples.inference.basic import mlx_wan_prompt_to_video as shared
    from fastvideo.mlx_runtime import wan22, wan22_sample, wan_vae

    module = _load_wan22_module()
    checkpoint = tmp_path / "mlx"
    checkpoint.mkdir()
    (checkpoint / "mlx_dit.json").write_text(
        json.dumps({"config": {"patch_size": [1, 2, 2], "in_channels": 1}})
    )

    events: list[str] = []
    dit_ref = None

    class CyclicCompiledDiT:
        def __init__(self) -> None:
            self.compiled = self.forward

        def forward(self) -> None:
            pass

    def fake_load(*args, **kwargs):
        nonlocal dit_ref
        assert kwargs["compile"] is True
        dit = CyclicCompiledDiT()
        dit_ref = weakref.ref(dit)
        return dit

    def fake_decode(*args, **kwargs):
        events.append("decode")
        assert dit_ref is not None and dit_ref() is None
        assert events[-3:] == ["gc", "mlx-clear", "decode"]
        return np.zeros((1, 1, 2, 2, 3), dtype=np.float32)

    def fake_postprocess(**kwargs) -> None:
        events.append("rife")
        assert events[-4:] == ["decode", "gc", "mps-clear", "rife"]

    real_collect = gc.collect

    def tracked_collect() -> int:
        events.append("gc")
        return real_collect()

    monkeypatch.setattr(shared, "encode_prompt", lambda **kwargs: torch.zeros((1, 1, 1)))
    monkeypatch.setattr(shared, "make_rotary_embeddings", lambda *args, **kwargs: None)
    monkeypatch.setattr(shared, "_postprocess_video", fake_postprocess)
    monkeypatch.setattr(wan22, "mlx_wan22_dit_from_mlx_checkpoint", fake_load)
    monkeypatch.setattr(wan22_sample, "sample_wan22_dmd", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(wan_vae, "decode_latents_wan_vae_torch", fake_decode)
    monkeypatch.setattr(diffusers_utils, "export_to_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(gc, "collect", tracked_collect)
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("mlx-clear"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("mps-clear"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_wan22_generate.py",
            "--text-encoder-root",
            str(tmp_path / "text"),
            "--no-prompt-cache",
            "--mlx-checkpoint",
            str(checkpoint),
            "--height",
            "32",
            "--width",
            "32",
            "--num-frames",
            "1",
            "--dmd-denoising-steps",
            "1",
            "--compile",
            "--fast",
            "--decode-backend",
            "wan-vae",
            "--vae-root",
            str(tmp_path / "vae"),
            "--output-path",
            str(tmp_path / "out.mp4"),
        ],
    )

    module.main()

    assert events == ["gc", "mlx-clear", "decode", "gc", "mps-clear", "rife"]
