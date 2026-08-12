# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for modular training profiler lifecycle ownership."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from fastvideo.train.entrypoint.train import run_training_from_config


class _RecordingProfiler:

    def __init__(self) -> None:
        self.events: list[tuple[str, str] | tuple[str]] = []

    @contextmanager
    def region(self, name: str):
        self.events.append(("enter", name))
        try:
            yield
        finally:
            self.events.append(("exit", name))

    def stop(self) -> None:
        self.events.append(("stop", ))


def test_dry_run_profiles_model_build_and_flushes(monkeypatch) -> None:
    profiler = _RecordingProfiler()
    training = SimpleNamespace(
        distributed=SimpleNamespace(tp_size=1, sp_size=1),
        vsa_sparsity=0.0,
        model_path="model",
    )
    cfg = SimpleNamespace(training=training)

    monkeypatch.setattr(
        "fastvideo.train.utils.config.load_run_config",
        lambda *args, **kwargs: cfg,
    )
    monkeypatch.setattr(
        "fastvideo.distributed.maybe_init_distributed_environment_and_model_parallel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "fastvideo.train.utils.builder.build_from_config",
        lambda loaded_cfg: (loaded_cfg.training, object(), object(), 0),
    )
    monkeypatch.setattr(
        "fastvideo.train.entrypoint.train.get_or_create_profiler",
        lambda trace_dir: profiler,
    )

    run_training_from_config("unused.yaml", dry_run=True)

    assert profiler.events == [
        ("enter", "profiler_region_model_loading"),
        ("exit", "profiler_region_model_loading"),
        ("stop", ),
    ]
