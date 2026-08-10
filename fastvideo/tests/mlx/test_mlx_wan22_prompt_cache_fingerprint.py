# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 prompt-embedding cache fingerprint regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_wan22_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "examples/inference/basic/mlx_wan22_generate.py"
    spec = importlib.util.spec_from_file_location(
        "mlx_wan22_generate_for_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wan22_prompt_cache_rejects_mismatched_fingerprint(tmp_path: Path) -> None:
    module = _load_wan22_module()
    cache_path = tmp_path / "prompt.npy"
    text_encoder_root = tmp_path / "encoder"
    text_encoder_root.mkdir()

    fingerprint = module._prompt_cache_fingerprint(
        prompt="a fox",
        prompt_used="a fox, cinematic",
        enhance_prompt=True,
        enhance_prompt_backend="template",
        text_encoder_root=text_encoder_root,
        max_sequence_length=512,
        dtype="fp16",
    )
    embeds = np.ones((1, 512, 4), dtype=np.float32)

    class _Embeds:
        def cpu(self):
            return self

        def numpy(self):
            return embeds

    module._save_prompt_cache(cache_path, _Embeds(), fingerprint)

    assert module._load_prompt_cache_if_valid(cache_path, fingerprint) is not None

    changed = dict(fingerprint)
    changed["prompt"] = "a cat"
    assert module._load_prompt_cache_if_valid(cache_path, changed) is None


def test_wan22_prompt_cache_rejects_missing_metadata(tmp_path: Path) -> None:
    module = _load_wan22_module()
    cache_path = tmp_path / "prompt.npy"
    np.save(cache_path, np.zeros((1, 512, 4), dtype=np.float32))

    fingerprint = module._prompt_cache_fingerprint(
        prompt="a fox",
        prompt_used="a fox",
        enhance_prompt=False,
        enhance_prompt_backend="template",
        text_encoder_root=tmp_path,
        max_sequence_length=512,
        dtype="fp16",
    )

    assert module._load_prompt_cache_if_valid(cache_path, fingerprint) is None
