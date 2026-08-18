# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 prompt-embedding cache fingerprint regression tests."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from fastvideo.mlx_runtime.prompt_cache import load_prompt_cache, save_prompt_cache


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


def _load_wan21_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "examples/inference/basic/mlx_wan_prompt_to_video.py"
    spec = importlib.util.spec_from_file_location(
        "mlx_wan_prompt_to_video_for_cache_test", script
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

    save_prompt_cache(cache_path, embeds, fingerprint)

    assert load_prompt_cache(cache_path, fingerprint) is not None

    changed = dict(fingerprint)
    changed["prompt"] = "a cat"
    assert load_prompt_cache(cache_path, changed) is None


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

    assert load_prompt_cache(cache_path, fingerprint) is None


def test_prompt_cache_is_best_effort_and_rejects_torn_data(tmp_path: Path) -> None:
    cache_path = tmp_path / "prompt.npy"
    fingerprint = {"prompt": "a fox"}
    save_prompt_cache(cache_path, np.ones((1, 1, 1)), fingerprint)

    np.save(cache_path, np.full((1, 1, 1), 9.0))
    assert load_prompt_cache(cache_path, fingerprint) is None

    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("occupied")
    assert not save_prompt_cache(
        invalid_parent / "prompt.npy",
        np.ones((1, 1, 1)),
        fingerprint,
    )

    incomplete_path = tmp_path / "incomplete.npy"
    incomplete = {"text_encoder": {"complete": False}}
    assert not save_prompt_cache(incomplete_path, np.ones((1, 1, 1)), incomplete)
    assert not incomplete_path.exists()


def test_wan21_explicit_cache_reencodes_for_a_changed_prompt(monkeypatch, tmp_path: Path) -> None:
    import torch

    module = _load_wan21_module()
    calls = []

    def fake_encode_prompt(**kwargs):
        calls.append(kwargs["prompt"])
        return torch.tensor([[float(len(calls))]])

    monkeypatch.setattr(module, "encode_prompt", fake_encode_prompt)
    cache_path = tmp_path / "prompt.npy"
    kwargs = {
        "model_root": tmp_path / "model",
        "max_sequence_length": 512,
        "device_arg": "cpu",
        "dtype_arg": "fp16",
        "encode_mode": "inline",
        "cache_path": cache_path,
    }
    first = module.get_prompt_embeds(prompt="first", **kwargs)
    second = module.get_prompt_embeds(prompt="second", **kwargs)
    second_cached = module.get_prompt_embeds(prompt="second", **kwargs)

    assert calls == ["first", "second"]
    assert not torch.equal(first, second)
    assert torch.equal(second, second_cached)


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


def test_wan22_prompt_cache_has_a_default_path(tmp_path: Path) -> None:
    """The cache must work without an explicit --prompt-embeds-cache.

    It previously only engaged when handed a path, so every 5B run paid a full
    UMT5 encode (~45s on an M4 Max) even for a repeat prompt, while the Wan2.1
    entrypoint cached by default.
    """
    module = _load_wan22_module()
    fingerprint = module._prompt_cache_fingerprint(
        prompt="a fox",
        prompt_used="a fox",
        enhance_prompt=False,
        enhance_prompt_backend="none",
        text_encoder_root=tmp_path,
        max_sequence_length=512,
        dtype="fp16",
    )
    path = module._default_prompt_cache_path(fingerprint)
    assert path.suffix == ".npy"
    assert path.parent.name == "prompt_embeds"
    assert path.name.startswith("wan22_")


def test_wan22_default_cache_path_tracks_the_fingerprint(tmp_path: Path) -> None:
    """Two prompts must not collide, and the same prompt must be stable."""
    module = _load_wan22_module()

    def fp(prompt: str):
        return module._prompt_cache_fingerprint(
            prompt=prompt,
            prompt_used=prompt,
            enhance_prompt=False,
            enhance_prompt_backend="none",
            text_encoder_root=tmp_path,
            max_sequence_length=512,
            dtype="fp16",
        )

    assert module._default_prompt_cache_path(fp("a fox")) == module._default_prompt_cache_path(fp("a fox"))
    assert module._default_prompt_cache_path(fp("a fox")) != module._default_prompt_cache_path(fp("a cat"))


def test_wan22_default_cache_path_tracks_encoder_files(tmp_path: Path) -> None:
    module = _load_wan22_module()
    weights = tmp_path / "text_encoder/weights.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"one")

    def path():
        fingerprint = module._prompt_cache_fingerprint(
            prompt="a fox",
            prompt_used="a fox",
            enhance_prompt=False,
            enhance_prompt_backend="none",
            text_encoder_root=tmp_path,
            max_sequence_length=512,
            dtype="fp16",
        )
        return module._default_prompt_cache_path(fingerprint)

    before = path()
    weights.write_bytes(b"two")
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert path() != before


@pytest.mark.parametrize(
    ("cache_args", "expected"),
    [
        ([], "default"),
        (["--no-prompt-cache"], None),
        (["--prompt-embeds-cache", "explicit.npy"], "explicit"),
        (["--no-prompt-cache", "--prompt-embeds-cache", "explicit.npy"], "explicit"),
    ],
)
def test_wan22_main_resolves_prompt_cache(
    monkeypatch,
    tmp_path: Path,
    cache_args: list[str],
    expected: str | None,
) -> None:
    module = _load_wan22_module()
    checkpoint = tmp_path / "mlx"
    checkpoint.mkdir()
    (checkpoint / "mlx_dit.json").write_text(
        json.dumps({"config": {"patch_size": [1, 2, 2], "in_channels": 1}})
    )
    explicit = tmp_path / "explicit.npy"

    class CacheProbe(Exception):
        pass

    def probe(cache_path, fingerprint):
        if expected == "default":
            assert cache_path == module._default_prompt_cache_path(fingerprint)
        elif expected == "explicit":
            assert cache_path == explicit
        else:
            assert cache_path is None
        raise CacheProbe

    resolved_args = [str(explicit) if arg == "explicit.npy" else arg for arg in cache_args]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(module, "load_prompt_cache", probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx_wan22_generate.py",
            "--text-encoder-root",
            str(tmp_path / "text"),
            "--mlx-checkpoint",
            str(checkpoint),
            "--height",
            "32",
            "--width",
            "32",
            "--num-frames",
            "1",
            *resolved_args,
        ],
    )

    with pytest.raises(CacheProbe):
        module.main()
