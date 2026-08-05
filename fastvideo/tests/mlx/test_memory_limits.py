# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from fastvideo.mlx_runtime.memory import add_memory_limit_args, apply_memory_limits, gib_to_bytes


class _FakeMLX:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def set_memory_limit(self, value: int) -> int:
        self.calls.append(("memory", value))
        return 111

    def set_cache_limit(self, value: int) -> int:
        self.calls.append(("cache", value))
        return 222

    def set_wired_limit(self, value: int) -> int:
        self.calls.append(("wired", value))
        return 333


def test_gib_to_bytes_rejects_non_positive_values() -> None:
    assert gib_to_bytes(None) is None
    assert gib_to_bytes(1.5) == int(1.5 * 1024**3)
    with pytest.raises(ValueError, match="positive"):
        gib_to_bytes(0)


def test_apply_memory_limits_sets_mlx_limits_and_metrics(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    fake_mlx = _FakeMLX()

    applied = apply_memory_limits(
        mlx_memory_limit_gib=16,
        mlx_disable_cache=True,
        mlx_wired_limit_gib=12,
        torch_mps_high_watermark_ratio=0.57,
        mx_module=fake_mlx,
    )

    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.57"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.0"
    assert fake_mlx.calls == [
        ("memory", 16 * 1024**3),
        ("cache", 0),
        ("wired", 12 * 1024**3),
    ]
    metrics = applied.as_metrics()
    assert metrics["mlx_memory_limit_bytes"] == 16 * 1024**3
    assert metrics["mlx_cache_limit_bytes"] == 0
    assert metrics["previous_mlx_memory_limit_bytes"] == 111
    assert metrics["previous_mlx_cache_limit_bytes"] == 222
    assert metrics["previous_mlx_wired_limit_bytes"] == 333


def test_apply_memory_limits_validates_watermarks() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_memory_limits(torch_mps_high_watermark_ratio=-1)


def test_add_memory_limit_args_uses_defaults() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    add_memory_limit_args(
        parser,
        mlx_memory_limit_gib=16,
        mlx_disable_cache=True,
        torch_mps_high_watermark_ratio=0.57,
        torch_mps_low_watermark_ratio=0.0,
    )

    args = parser.parse_args([])
    assert args.mlx_memory_limit_gib == 16
    assert args.mlx_disable_cache is True
    assert args.torch_mps_high_watermark_ratio == 0.57
    assert args.torch_mps_low_watermark_ratio == 0.0
