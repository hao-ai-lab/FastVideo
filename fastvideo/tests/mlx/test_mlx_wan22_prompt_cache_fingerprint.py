# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 prompt-embedding cache fingerprint regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


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


def test_wan22_missing_paths_download_only_selected_assets(monkeypatch, tmp_path: Path) -> None:
    module = _load_wan22_module()
    import huggingface_hub

    calls = []

    def fake_snapshot_download(model_id, allow_patterns):
        calls.append((model_id, allow_patterns))
        return str(tmp_path / ("wan21" if "2.1" in model_id else "wan22"))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    text_root, dit_checkpoint, dit_config, vae_root = module._resolve_model_paths(
        text_encoder_root=None,
        dit_checkpoint=None,
        dit_config=None,
        vae_root=None,
        mlx_checkpoint=None,
        decode_backend="taehv",
    )

    assert text_root == tmp_path / "wan21"
    assert dit_checkpoint == tmp_path / "wan22/transformer/diffusion_pytorch_model.safetensors"
    assert dit_config == tmp_path / "wan22/transformer/config.json"
    assert vae_root is None
    assert calls == [
        (module.FASTWAN21_MODEL_ID, ["tokenizer/*", "text_encoder/*"]),
        (
            module.FASTWAN22_MODEL_ID,
            ["transformer/diffusion_pytorch_model.safetensors", "transformer/config.json"],
        ),
    ]


def test_wan22_explicit_mlx_paths_do_not_download(monkeypatch, tmp_path: Path) -> None:
    module = _load_wan22_module()
    import huggingface_hub

    def unexpected_download(*args, **kwargs):
        pytest.fail("explicit MLX and text paths must not download model assets")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    text_root, dit_checkpoint, dit_config, vae_root = module._resolve_model_paths(
        text_encoder_root=tmp_path / "text",
        dit_checkpoint=None,
        dit_config=None,
        vae_root=None,
        mlx_checkpoint=tmp_path / "mlx",
        decode_backend="taehv",
    )

    assert text_root == tmp_path / "text"
    assert dit_checkpoint is None
    assert dit_config is None
    assert vae_root is None
