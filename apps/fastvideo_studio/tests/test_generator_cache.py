# SPDX-License-Identifier: Apache-2.0
"""Tests for the resident-generator API: preload, list, unload.

The generator itself is faked at the JobRunner boundary — these tests pin the
cache/lifecycle contract (idempotent preload, state transitions, unload
refusal while a job runs), not model loading.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from fastvideo_studio.job_runner import JobRunner, JobStatus


class _FakeGenerator:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    r = JobRunner.__new__(JobRunner)  # skip __init__: no DB/Manager needed
    r._jobs = {}
    r._jobs_lock = threading.Lock()
    r._generators = {}
    r._generators_lock = threading.Lock()
    r._preloads = {}
    return r


def _wait_state(runner, model_id, state, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        entries = [g for g in runner.list_generators() if g["model_id"] == model_id]
        if entries and entries[0]["state"] == state:
            return entries[0]
        time.sleep(0.02)
    raise AssertionError(f"{model_id} never reached state {state}: {runner.list_generators()}")


def test_preload_loads_once_and_lists_ready(runner, monkeypatch):
    created = []

    def fake_create(self, **params):
        gen = _FakeGenerator()
        created.append(params["model_id"])
        key = self._generator_cache_key(**params)
        with self._generators_lock:
            self._generators[key] = gen
        return gen

    monkeypatch.setattr(JobRunner, "_get_or_create_generator", fake_create)

    first = runner.preload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8)
    # the loader thread may finish before we return; either state is legal here
    assert first["state"] in ("loading", "ready")
    ready = _wait_state(runner, "org/model-a", "ready")
    assert ready["num_gpus"] == 8

    # idempotent: preloading a resident config is an immediate ready, no reload
    again = runner.preload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8)
    assert again["state"] == "ready"
    assert created == ["org/model-a"]


def test_preload_failure_is_reported_not_raised(runner, monkeypatch):
    def fake_create(self, **params):
        raise RuntimeError("no CUDA on this box")

    monkeypatch.setattr(JobRunner, "_get_or_create_generator", fake_create)
    entry = runner.preload_generator(model_id="org/broken", workload_type="t2v", num_gpus=1)
    assert entry["state"] in ("loading", "failed")
    failed = _wait_state(runner, "org/broken", "failed")
    assert "no CUDA" in failed["error"]
    # a failed preload must not leave a resident entry
    assert all(g["state"] != "ready" for g in runner.list_generators())


def test_unload_shuts_down_and_removes(runner):
    gen = _FakeGenerator()
    key = runner._generator_cache_key(model_id="org/model-a", workload_type="t2v", num_gpus=8)
    runner._generators[key] = gen

    assert runner.unload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8) is True
    assert gen.shutdown_calls == 1
    assert runner.list_generators() == []
    # second unload: nothing resident
    assert runner.unload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8) is False


def test_unload_refuses_while_inference_job_runs(runner):
    key = runner._generator_cache_key(model_id="org/model-a", workload_type="t2v", num_gpus=8)
    gen = _FakeGenerator()
    runner._generators[key] = gen
    runner._jobs["j1"] = SimpleNamespace(id="j1", status=JobStatus.RUNNING, job_type="inference")

    with pytest.raises(RuntimeError, match="j1"):
        runner.unload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8)
    assert gen.shutdown_calls == 0  # still resident

    # training jobs don't block unload
    runner._jobs["j1"].job_type = "finetune"
    assert runner.unload_generator(model_id="org/model-a", workload_type="t2v", num_gpus=8) is True


def test_cache_key_matches_job_lookup_defaults(runner):
    """A preload with GeneratorRequest defaults must hit the same cache slot a
    job with CreateJobRequest defaults resolves to — else preloading is a lie."""
    from fastvideo_studio.models import CreateJobRequest, GeneratorRequest

    job = CreateJobRequest(model_id="m", prompt="p").model_dump()
    pre = GeneratorRequest(model_id="m").model_dump()
    job_key = runner._generator_cache_key(
        **{k: job[k] for k in pre})
    pre_key = runner._generator_cache_key(**pre)
    assert job_key == pre_key
