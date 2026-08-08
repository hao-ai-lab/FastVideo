# SPDX-License-Identifier: Apache-2.0
"""Tests for the single-slot resident generator: preload, list, unload.

Exactly one VideoGenerator lives in memory. One load at a time; loading a new
config always releases the old instance; unload deletes it. The generator is
faked at the ``_create_generator_into_slot``/VideoGenerator boundary.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from fastvideo_studio.job_runner import JobRunner, JobStatus


class _FakeGenerator:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture()
def runner():
    r = JobRunner.__new__(JobRunner)  # skip __init__: no DB/Manager needed
    r._jobs = {}
    r._jobs_lock = threading.Lock()
    r._generator = None
    r._generator_config = None
    r._generator_state = "empty"
    r._generator_error = None
    r._generator_lock = threading.Lock()
    r._load_lock = threading.Lock()
    r._worker_log_queue = None  # no Manager in unit tests
    import queue as _queue
    r._loader_queue = _queue.Queue()
    threading.Thread(target=r._loader_loop, daemon=True).start()
    return r


def _install_fake_loader(runner, monkeypatch, made: list | None = None,
                         gate: threading.Event | None = None,
                         fail: str | None = None):
    """Replace the VideoGenerator load inside _create_generator_into_slot."""
    import fastvideo_studio.job_runner as jr

    class _FakeVG:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            if gate is not None:
                assert gate.wait(5), "test gate never opened"
            if fail is not None:
                raise RuntimeError(fail)
            gen = _FakeGenerator()
            if made is not None:
                made.append((model_path, gen))
            return gen

    monkeypatch.setitem(__import__("sys").modules, "fastvideo",
                        SimpleNamespace(VideoGenerator=_FakeVG))
    return jr


def _wait_state(runner, state, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with runner._generator_lock:
            if runner._generator_state == state:
                return runner._slot_entry()
        time.sleep(0.02)
    raise AssertionError(f"never reached state {state}: {runner._generator_state}")


def test_preload_then_ready_then_idempotent(runner, monkeypatch):
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)

    entry = runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    assert entry["state"] in ("loading", "ready")
    _wait_state(runner, "ready")

    again = runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    assert again["state"] == "ready"
    assert len(made) == 1  # same config never reloads


def test_only_one_load_at_a_time(runner, monkeypatch):
    gate = threading.Event()
    _install_fake_loader(runner, monkeypatch, gate=gate)

    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    with pytest.raises(RuntimeError, match="already in progress"):
        runner.preload_generator(model_id="org/b", workload_type="t2v", num_gpus=8)
    gate.set()
    _wait_state(runner, "ready")


def test_new_config_always_releases_old_instance(runner, monkeypatch):
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)

    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")
    first = made[0][1]

    runner.preload_generator(model_id="org/b", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")
    assert first.shutdown_calls == 1  # old instance released, not stacked
    assert [m[0] for m in made] == ["org/a", "org/b"]
    assert len(runner.list_generators()) == 1
    assert runner.list_generators()[0]["model_id"] == "org/b"


def test_failed_load_reports_error_and_allows_retry(runner, monkeypatch):
    _install_fake_loader(runner, monkeypatch, fail="no CUDA on this box")
    runner.preload_generator(model_id="org/broken", workload_type="t2v", num_gpus=1)
    failed = _wait_state(runner, "failed")
    assert "no CUDA" in failed["error"]

    # retry after failure must be accepted (this was the reported bug)
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)
    runner.preload_generator(model_id="org/broken", workload_type="t2v", num_gpus=1)
    _wait_state(runner, "ready")
    assert len(made) == 1


def test_unload_deletes_instance(runner, monkeypatch):
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)
    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")

    assert runner.unload_generator() is True
    assert made[0][1].shutdown_calls == 1
    assert runner.list_generators() == []
    assert runner._generator is None
    assert runner.unload_generator() is False  # nothing resident


def test_unload_refuses_while_inference_job_runs(runner, monkeypatch):
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)
    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")
    runner._jobs["j1"] = SimpleNamespace(id="j1", status=JobStatus.RUNNING, job_type="inference")

    with pytest.raises(RuntimeError, match="j1"):
        runner.unload_generator()
    assert made[0][1].shutdown_calls == 0

    runner._jobs["j1"].job_type = "finetune"  # training doesn't block
    assert runner.unload_generator() is True


def test_job_waits_for_matching_preload(runner, monkeypatch):
    gate = threading.Event()
    made = []
    _install_fake_loader(runner, monkeypatch, made=made, gate=gate)

    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    got = []
    t = threading.Thread(
        target=lambda: got.append(
            runner._get_or_create_generator("org/a", "t2v", 8)),
        daemon=True)
    t.start()
    time.sleep(0.2)
    assert not got  # job blocked on the in-flight load
    gate.set()
    t.join(5)
    assert got and got[0] is made[0][1]
    assert len(made) == 1  # the job reused the preloaded instance


def test_job_with_different_config_replaces_slot(runner, monkeypatch):
    made = []
    _install_fake_loader(runner, monkeypatch, made=made)
    runner.preload_generator(model_id="org/a", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")

    gen = runner._get_or_create_generator("org/b", "t2v", 4)
    assert gen is made[1][1]
    assert made[0][1].shutdown_calls == 1  # old released before new load
    assert runner.list_generators()[0]["model_id"] == "org/b"


def test_job_during_replace_never_sees_half_replaced_slot(runner, monkeypatch):
    """Regression: user restarted a stale 1-gpu job while an 8-gpu generator
    was resident; the replace nulled the generator with state still 'ready',
    and the next (correct) job grabbed None -> AttributeError on
    generate_video. All transitions now serialize under the load lock."""
    made = []
    gate = threading.Event()
    _install_fake_loader(runner, monkeypatch, made=made)
    runner.preload_generator(model_id="org/h3", workload_type="t2v", num_gpus=8)
    _wait_state(runner, "ready")

    _install_fake_loader(runner, monkeypatch, made=made, gate=gate)
    results: dict[str, Any] = {}

    def job_a():  # stale job: mismatching config triggers a slow replace
        results["a"] = runner._get_or_create_generator("org/h3", "t2v", 1)

    def job_b():  # correct job arriving mid-replace
        time.sleep(0.3)
        results["b"] = runner._get_or_create_generator("org/h3", "t2v", 8)

    ta = threading.Thread(target=job_a, daemon=True)
    tb = threading.Thread(target=job_b, daemon=True)
    ta.start(); tb.start()
    time.sleep(0.6)
    gate.set()
    ta.join(10); tb.join(10)

    assert results["a"] is not None and hasattr(results["a"], "shutdown")
    assert results["b"] is not None and hasattr(results["b"], "shutdown")
    # b arrived second, so the slot ends at b's 8-gpu config
    assert runner.list_generators()[0]["num_gpus"] == 8


def test_config_dict_matches_request_defaults(runner):
    """GeneratorRequest defaults and CreateJobRequest defaults must resolve to
    the same slot config — else preloading never matches the job."""
    from fastvideo_studio.models import CreateJobRequest, GeneratorRequest

    job = CreateJobRequest(model_id="m", prompt="p").model_dump()
    pre = GeneratorRequest(model_id="m").model_dump()
    assert runner._generator_config_dict(**pre) == runner._generator_config_dict(
        **{k: job[k] for k in pre})


def test_engine_log_buffer_incremental_tail():
    from fastvideo_studio.server import _EngineLogBuffer

    buf = _EngineLogBuffer(maxlen=3)
    buf.write("a\nb\n")
    buf.write("c")          # partial line: not visible yet
    lines, total = buf.get_lines(0)
    assert lines == ["a", "b"] and total == 2
    buf.write("!\nd\ne\n")  # completes "c!", then overflows the ring
    lines, total = buf.get_lines(total)
    assert lines == ["c!", "d", "e"] and total == 5
    # reader far behind: dropped lines are skipped, no crash
    lines, _ = buf.get_lines(0)
    assert lines == ["c!", "d", "e"]


def test_engine_feed_drives_job_progress(runner):
    from fastvideo_studio.job_runner import Job

    job = Job(id="j-prog", model_id="m", prompt="p")
    runner._active_inference_job = job
    # ray wraps relayed lines in ANSI color codes — the bridge must strip them
    runner.feed_engine_line("\x1b[36m(RayWorkerWrapper pid=123, ip=10.0.0.2)\x1b[0m denoising:  40%|████      | 20/50 [00:30<00:45,  1.5s/it]")
    assert job._log_buf.progress == 40.0
    assert job._log_buf.progress_msg == "20/50 steps"

    # driver-side lines (no ray prefix) are NOT double-fed
    before = job._log_buf.get_lines()[1]
    runner.feed_engine_line("INFO 08-07 [video_generator.py] driver line")
    assert job._log_buf.get_lines()[1] == before

    # no active job: no-op
    runner._active_inference_job = None
    runner.feed_engine_line("(RayWorkerWrapper pid=123)  90%|████| 45/50")
    assert job._log_buf.progress == 40.0
