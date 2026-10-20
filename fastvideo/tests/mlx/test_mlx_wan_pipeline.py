# SPDX-License-Identifier: Apache-2.0
"""Wan constructor guards and real generate() control flow with lightweight components.

The orchestration tests use NumPy arrays in place of Metal arrays and keep the
native schedulers and samplers. Real-weight video checks live in the opt-in
Apple Silicon smoke test.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

import pytest

from fastvideo.mlx_runtime.wan_pipeline import (
    MLXWan22Pipeline,
    MLXWanPipeline,
)
from fastvideo.mlx_runtime.wan_helpers import resolve_wan_torch_dtype as _resolve_wan_torch_dtype


def _make_model_root(tmp_path: Path) -> Path:
    """A model root with just enough structure to pass the constructor's checks."""
    model_root = tmp_path / "model_root"
    (model_root / "tokenizer").mkdir(parents=True)
    (model_root / "text_encoder").mkdir(parents=True)
    return model_root


def _make_packed_checkpoint(tmp_path: Path, name: str = "FastMetal-1.3B-QAD-mlx", in_channels: int | None = None) -> Path:
    """A directory shaped like a real packed MLX DiT checkpoint."""
    checkpoint = tmp_path / name
    checkpoint.mkdir(parents=True)
    manifest = {"config": {"in_channels": in_channels}} if in_channels is not None else {}
    (checkpoint / "mlx_dit.json").write_text(json.dumps(manifest))
    (checkpoint / "mlx_dit.safetensors").write_bytes(b"")
    return checkpoint


def test_init_accepts_a_valid_model_root_and_checkpoint(tmp_path) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path)
    pipeline = MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)
    assert pipeline.model_root == model_root
    assert pipeline.mlx_checkpoint == checkpoint


def test_init_rejects_missing_tokenizer_dir(tmp_path) -> None:
    model_root = tmp_path / "model_root"
    (model_root / "text_encoder").mkdir(parents=True)
    checkpoint = _make_packed_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_rejects_missing_text_encoder_dir(tmp_path) -> None:
    model_root = tmp_path / "model_root"
    (model_root / "tokenizer").mkdir(parents=True)
    checkpoint = _make_packed_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="text_encoder"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_rejects_nvidia_fastwan_qad_checkpoint_name(tmp_path) -> None:
    """FastWan-QAD is the NVIDIA NVFP4/FP8 release; loading it through MLX
    silently requantizes the wrong weights (see checkpoint_compat.py)."""
    model_root = _make_model_root(tmp_path)
    nvidia_checkpoint = tmp_path / "FastWan-QAD-1.3B"
    nvidia_checkpoint.mkdir()
    with pytest.raises(ValueError, match="FastWan-QAD"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=nvidia_checkpoint)


def test_init_allows_the_legacy_int8_nvidia_named_directory(tmp_path) -> None:
    """FastWan-QAD-INT8-* predates the mlx_dit.json packing convention but is
    a real Apple checkpoint; the name-based NVIDIA check must not reject it."""
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path, name="FastWan-QAD-INT8-1.3B")
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_does_not_validate_checkpoint_contents(tmp_path) -> None:
    """A directory that is neither a packed MLX checkpoint nor NVIDIA-flagged
    passes __init__ unexamined -- real content validation happens inside
    generate() when the weights are actually loaded (needs Metal)."""
    model_root = _make_model_root(tmp_path)
    empty_checkpoint = tmp_path / "not_a_real_checkpoint"
    empty_checkpoint.mkdir()
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=empty_checkpoint)


def test_init_rejects_a_wan22_checkpoint(tmp_path) -> None:
    """Pointing the Wan2.1 pipeline at a 48-channel FastMetal-5B-QAD checkpoint
    would silently produce garbled output (wrong VAE compression assumed) --
    this must be caught here, not discovered downstream."""
    model_root = _make_model_root(tmp_path)
    wan22_checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
    with pytest.raises(ValueError, match="Wan2.2-TI2V"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=wan22_checkpoint)


def test_init_accepts_a_checkpoint_with_declared_16_channels(tmp_path) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path, in_channels=16)
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def _write_manifest(tmp_path: Path, body: str) -> Path:
    """A packed-checkpoint directory whose mlx_dit.json holds arbitrary JSON."""
    checkpoint = tmp_path / "FastMetal-1.3B-QAD-mlx"
    checkpoint.mkdir(parents=True)
    (checkpoint / "mlx_dit.json").write_text(body)
    (checkpoint / "mlx_dit.safetensors").write_bytes(b"")
    return checkpoint


@pytest.mark.parametrize("body", ["[]", '"nope"', "42", "null"])
def test_init_tolerates_a_manifest_that_is_not_an_object(tmp_path, body) -> None:
    """The channel probe is documented as best-effort: a readable but wrongly shaped
    manifest must fall through to the weight load, not raise out of __init__."""
    model_root = _make_model_root(tmp_path)
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=_write_manifest(tmp_path, body))


def test_init_tolerates_a_manifest_whose_config_is_not_an_object(tmp_path) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _write_manifest(tmp_path, json.dumps({"config": ["in_channels"]}))
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


@pytest.mark.parametrize("channels", ["sixteen", [16], {"value": 16}])
def test_init_tolerates_a_non_numeric_in_channels(tmp_path, channels) -> None:
    model_root = _make_model_root(tmp_path)
    checkpoint = _write_manifest(tmp_path, json.dumps({"config": {"in_channels": channels}}))
    MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_init_still_reads_a_numeric_string_in_channels(tmp_path) -> None:
    """Tolerating junk must not stop the probe recognising a real Wan2.2 checkpoint."""
    model_root = _make_model_root(tmp_path)
    checkpoint = _write_manifest(tmp_path, json.dumps({"config": {"in_channels": "48"}}))
    with pytest.raises(ValueError, match="Wan2.2-TI2V"):
        MLXWanPipeline(model_root=model_root, mlx_checkpoint=checkpoint)


def test_resolve_torch_dtype_maps_the_recipe_names() -> None:
    """Wan2.1 and Wan2.2-TI2V pass different names; both must resolve exactly."""
    torch = pytest.importorskip("torch")
    assert _resolve_wan_torch_dtype("bf16") is torch.bfloat16
    assert _resolve_wan_torch_dtype("fp16") is torch.float16
    assert _resolve_wan_torch_dtype("fp32") is torch.float32


def test_resolve_torch_dtype_rejects_an_unknown_name() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Unsupported text-encoder dtype"):
        _resolve_wan_torch_dtype("float8")


class TestMLXWan22Pipeline:
    """Mirrors the MLXWanPipeline coverage above for the Wan2.2-TI2V (5B) pipeline."""

    def test_init_accepts_a_valid_model_root_and_checkpoint(self, tmp_path) -> None:
        model_root = _make_model_root(tmp_path)
        checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
        pipeline = MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=checkpoint)
        assert pipeline.model_root == model_root

    def test_init_rejects_missing_tokenizer_dir(self, tmp_path) -> None:
        model_root = tmp_path / "model_root"
        (model_root / "text_encoder").mkdir(parents=True)
        checkpoint = _make_packed_checkpoint(tmp_path, name="FastMetal-5B-QAD", in_channels=48)
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=checkpoint)

    def test_init_rejects_a_wan21_checkpoint(self, tmp_path) -> None:
        """The reverse mistake: pointing the 5B pipeline at a 1.3B/14B checkpoint."""
        model_root = _make_model_root(tmp_path)
        wan21_checkpoint = _make_packed_checkpoint(tmp_path, in_channels=16)
        with pytest.raises(ValueError, match="MLXWanPipeline for 1.3B/14B"):
            MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=wan21_checkpoint)

    def test_init_does_not_validate_checkpoint_contents(self, tmp_path) -> None:
        model_root = _make_model_root(tmp_path)
        empty_checkpoint = tmp_path / "not_a_real_checkpoint"
        empty_checkpoint.mkdir()
        MLXWan22Pipeline(model_root=model_root, mlx_checkpoint=empty_checkpoint)


@pytest.mark.parametrize("pipeline_cls,channels,spatial,encoder_dtype,shift", [
    (MLXWanPipeline, 16, 8, "bf16", 8.0),
    (MLXWan22Pipeline, 48, 16, "fp16", 5.0),
])
def test_generate_runs_the_family_recipe_and_releases_dit_before_decode(
    tmp_path, monkeypatch, pipeline_cls, channels, spatial, encoder_dtype, shift,
):
    import torch
    from fastvideo.mlx_runtime import wan_pipeline, wan22_sample
    from fastvideo.models.schedulers import scheduling_flow_match_euler_discrete as scheduler_module

    events = []
    forwards = []
    captured = {}
    mx = ModuleType("mlx.core")
    mx.array, mx.float16, mx.float32 = np.array, np.float16, np.float32
    mx.full = np.full
    mx.eval = lambda *args: None
    mx.clear_cache = lambda: None
    mx.reset_peak_memory = lambda: None
    mx.get_peak_memory = lambda: 2**30
    rng = np.random.RandomState()
    mx.random = SimpleNamespace(seed=rng.seed, normal=lambda shape: rng.normal(size=shape))
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class TinyDiT:
        config = {"in_channels": channels, "num_attention_heads": 1,
                  "attention_head_dim": 12, "patch_size": [1, 2, 2]}
        patch_size = (1, 2, 2)

        def __call__(self, latents, embeddings, timestep, rope):
            forwards.append((latents.copy(), timestep.copy()))
            assert embeddings.dtype == np.float16
            return np.zeros_like(latents)

    def load(checkpoint, *, compile):
        assert compile is True
        captured["checkpoint"] = checkpoint
        return TinyDiT()

    def encode(**kwargs):
        captured["encode"] = kwargs
        events.append("encode")
        return torch.ones((1, 4, 12), dtype=torch.float16 if encoder_dtype == "fp16" else torch.float32)

    original_scheduler = scheduler_module.FlowMatchEulerDiscreteScheduler

    def make_scheduler(**kwargs):
        captured["shift"] = kwargs["shift"]
        return original_scheduler(**kwargs)

    original_sample = wan22_sample.sample_wan22_dmd

    def sample(*args, **kwargs):
        captured["sample"] = kwargs
        return original_sample(*args, **kwargs)

    def decode(latents, output_path, **kwargs):
        events.append("decode")
        assert events[-2] == "release_dit"
        captured["decoded"] = latents.copy()
        captured["decode"] = kwargs
        output_path.write_bytes(b"test video")

    monkeypatch.setattr("fastvideo.mlx_runtime.checkpoint.load_mlx_dit_checkpoint", load)
    monkeypatch.setattr("fastvideo.mlx_runtime.wan22.mlx_wan22_dit_from_mlx_checkpoint", load)
    monkeypatch.setattr(wan_pipeline, "_encode_wan_prompt", encode)
    monkeypatch.setattr(wan_pipeline, "_make_wan_rotary_embeddings", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(wan_pipeline, "cleanup_mlx", lambda: events.append("release_dit"))
    monkeypatch.setattr(wan_pipeline, "cleanup_torch_mps", lambda: None)
    monkeypatch.setattr(scheduler_module, "FlowMatchEulerDiscreteScheduler", make_scheduler)
    monkeypatch.setattr(wan22_sample, "sample_wan22_dmd", sample)
    monkeypatch.setattr("fastvideo.mlx_runtime.wan_vae.decode_latents_to_video", decode)

    root = _make_model_root(tmp_path)
    checkpoint = _make_packed_checkpoint(tmp_path, in_channels=channels)
    pipeline = pipeline_cls(model_root=root, mlx_checkpoint=checkpoint, prompt_cache_dir=tmp_path / "cache")
    result = pipeline.generate("a fox", output_path=tmp_path / "out.mp4", width=64, height=32,
                               num_frames=5, seed=1234, fps=27, max_sequence_length=4)
    assert captured["encode"].get("dtype_arg", "bf16") == encoder_dtype
    assert captured["encode"].get("device_arg", "auto") == ("cpu" if channels == 48 else "auto")
    assert captured["encode"]["cache_dir"] == tmp_path / "cache"
    assert captured["checkpoint"] == checkpoint
    assert captured["shift"] == shift
    shape = (1, channels, 2, 32 // spatial, 64 // spatial)
    expected_noise = torch.randn(shape, generator=torch.Generator().manual_seed(1234),
                                dtype=torch.float32).numpy().astype(np.float16)
    np.testing.assert_array_equal(forwards[0][0], expected_noise)
    assert len(forwards) == 3
    if channels == 48:
        assert captured["sample"] == {"dmd_denoising_steps": [1000, 757, 522], "flow_shift": 5.0,
                                      "warp_denoising_step": True, "seed": 0}
        _, expected_steps = wan22_sample.build_wan22_dmd_schedule()
    else:
        expected_steps = [1000, 757, 522]
    assert [float(np.asarray(t).flat[0]) for _, t in forwards] == expected_steps
    assert captured["decoded"].shape == shape
    assert captured["decoded"].dtype == np.float32
    assert captured["decode"] == {"fps": 27, "backend": "taehv", "z_dim": channels,
                                  "taehv_checkpoint": None, "torch_device": "auto"}
    assert result.video_path == str(tmp_path / "out.mp4")
    assert result.video_decode_backend == "taehv"
    assert result.peak_memory_gib["load_peak_gib"] == 1.0


@pytest.mark.parametrize("pipeline_cls,channels,bad_fields", [
    (MLXWanPipeline, 16, {"num_frames": 80}),
    (MLXWanPipeline, 16, {"width": 840}),
    (MLXWan22Pipeline, 48, {"num_frames": 80}),
    (MLXWan22Pipeline, 48, {"width": 1296}),
])
def test_generate_rejects_invalid_shapes_before_prompt_encoding(tmp_path, monkeypatch, pipeline_cls, channels, bad_fields):
    from fastvideo.mlx_runtime import wan_pipeline

    def must_not_encode(**kwargs):
        raise AssertionError("Invalid geometry reached the encoder")

    monkeypatch.setattr(wan_pipeline, "_encode_wan_prompt", must_not_encode)
    pipeline = pipeline_cls(model_root=_make_model_root(tmp_path),
                            mlx_checkpoint=_make_packed_checkpoint(tmp_path, in_channels=channels))
    with pytest.raises(ValueError):
        pipeline.generate("a fox", output_path=tmp_path / "out.mp4", **bad_fields)
