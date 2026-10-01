# SPDX-License-Identifier: Apache-2.0
"""Suite lifecycle coverage with real local admission/locks and fake workers."""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_ci import dispatcher
from scripts.gpu_ci.policy import LANES


class FakeBackend:
    """Check reservation ordering at the external worker creation boundary."""

    def __init__(self, store, *, outcomes=None, stop_result=True, hold=None):
        self.store = store
        self.outcomes = outcomes or {}
        self.stop_result = stop_result
        self.hold = hold
        self.started = threading.Event()
        self.events = []
        self.persisted_handles = []
        self.lock = threading.Lock()

    def handle(self, build_id, lane_id):
        return {"build": build_id, "lane": lane_id, "job": f"{build_id}-{lane_id}"}

    def run(self, handle, request, lane, cancel):
        active = [allocation for allocation in self.store.snapshot()["allocations"]
                  if allocation["build_id"] == request["build_id"]
                  and allocation["lane_id"] == lane["key"] and allocation["state"] == "active"]
        assert len(active) == 1 and active[0]["handle"] == handle
        with self.lock:
            self.events.append(("start", lane["key"]))
            self.persisted_handles.append(handle)
        self.started.set()
        if self.hold is not None:
            deadline = time.monotonic() + 5
            while not self.hold.wait(0.005):
                if cancel.is_set():
                    return 143
                if time.monotonic() > deadline:
                    raise RuntimeError("Test worker was not released")
        outcome = self.outcomes.get(lane["key"], 0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def stop(self, handle):
        with self.lock:
            self.events.append(("stop", handle["lane"]))
        if isinstance(self.stop_result, Exception):
            raise self.stop_result
        return self.stop_result


@pytest.fixture
def config(tmp_path):
    return {"state_path": str(tmp_path / "admission.db"), "artifacts_dir": str(tmp_path / "artifacts"),
            "queue_timeout_seconds": 10, "max_active_prs": 2, "max_gpus_per_pr": 4, "max_gpus": 8}


def _lane(key="unit", gpus=1, fastcheck=True):
    return {"key": key, "gpus": gpus, "fastcheck": fastcheck}


def _request(build="build", pr="repo#1", lanes=None, commit="a" * 40):
    return {"build_id": build, "pr_key": pr, "commit": commit, "backend": "vllm",
            "scope": "full", "lanes": [_lane()] if lanes is None else lanes}


def _summary(config, build="build"):
    return json.loads((dispatcher.run_directory(config, build) / "summary.json").read_text())


def _build(store, build="build"):
    return next(item for item in store.snapshot()["builds"] if item["build_id"] == build)


def _active(store, build="build"):
    return [allocation for allocation in store.snapshot()["allocations"]
            if allocation["build_id"] == build and allocation["state"] == "active"]


def _wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for dispatcher state")
        time.sleep(0.005)


def test_full_suite_success_persists_handles_and_finishes_every_lane(config):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store)
    assert dispatcher.execute(_request(lanes=[dict(lane) for lane in LANES]), config,
                              backend=backend, poll_seconds=0.001) == 0
    summary = _summary(config)
    assert set(summary["lanes"]) == {lane["key"] for lane in LANES}
    assert all(result["state"] == "passed" and result["stopped"] for result in summary["lanes"].values())
    assert len(backend.persisted_handles) == 20
    assert _build(store)["state"] == "finished"
    assert not _active(store)
    assert not summary["recovery_required"]
    assert backend.events.index(("stop", "golden-gate")) < backend.events.index(("start", "ssim"))


@pytest.mark.parametrize("outcome, expected", [
    (42, 42), (RuntimeError("worker transport failed"), 97), (False, 97), (-1, 97), (256, 97),
])
def test_lane_failure_returns_exit_status_after_confirmed_cleanup(config, outcome, expected):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store, outcomes={"unit": outcome})
    assert dispatcher.execute(_request(), config, backend=backend, poll_seconds=0.001) == expected
    assert _summary(config)["lanes"]["unit"]["state"] == "failed"
    assert _summary(config)["lanes"]["unit"]["exit_code"] == expected
    assert _build(store)["state"] == "finished"
    assert not _active(store)
    assert ("stop", "unit") in backend.events


def test_failed_golden_gate_skips_integration_but_runs_fastcheck(config):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store, outcomes={"golden-gate": 7})
    lanes = [_lane("ssim", 4, False), _lane("encoder"), _lane("golden-gate", 1, False),
             _lane("training", 4, False)]
    assert dispatcher.execute(_request(lanes=lanes), config, backend=backend, poll_seconds=0.001) == 7
    started = {lane for action, lane in backend.events if action == "start"}
    assert started == {"encoder", "golden-gate"}
    summary = _summary(config)
    assert summary["lanes"]["ssim"]["state"] == "skipped"
    assert summary["lanes"]["training"]["reason"] == "golden gate failed"
    assert _build(store)["state"] == "finished"


def test_direct_integration_lane_without_selected_golden_runs(config):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store)
    request = _request(lanes=[_lane("ssim", 4, False)])
    request["scope"] = "direct"
    assert dispatcher.execute(request, config, backend=backend, poll_seconds=0.001) == 0
    assert ("start", "ssim") in backend.events


def test_empty_merge_plan_does_not_wait_for_gpu_admission(config):
    config["max_active_prs"] = 1
    config["queue_timeout_seconds"] = 0
    store = dispatcher.make_store(config)
    store.register("busy", "repo#other", "vllm", "b" * 40)
    assert store.try_admit("busy")
    backend = FakeBackend(store)
    request = _request(lanes=[])
    request["scope"] = "merge"
    assert dispatcher.execute(request, config, backend=backend, poll_seconds=0.001) == 0
    assert not backend.events
    assert not _summary(config)["recovery_required"]
    assert _summary(config)["lanes"] == {}
    assert _build(store)["state"] == "finished"


def test_persisted_cancellation_of_queued_build_never_starts_backend(config):
    config["max_active_prs"] = 1
    store = dispatcher.make_store(config)
    store.register("busy", "repo#other", "vllm", "b" * 40)
    assert store.try_admit("busy")
    backend = FakeBackend(store)
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(dispatcher.execute, _request(), config, backend=backend,
                             cancel=cancel, poll_seconds=0.001)
        try:
            _wait_for(lambda: any(build["build_id"] == "build" for build in store.snapshot()["builds"]))
            store.request_cancel("build")
            assert future.result(timeout=5) == dispatcher.INFRASTRUCTURE_FAILURE
        finally:
            cancel.set()
    assert not backend.events
    assert _build(store)["state"] == "finished"
    assert _build(store)["cancelled"]
    assert _build(store, "busy")["state"] == "admitted"


def test_queue_deadline_finishes_without_allocating_gpus(config):
    config["queue_timeout_seconds"] = 0
    store = dispatcher.make_store(config)
    backend = FakeBackend(store)
    assert dispatcher.execute(_request(), config, backend=backend, poll_seconds=0.001) == 97
    assert not backend.events
    assert _build(store)["state"] == "finished"
    assert _build(store)["cancelled"]


def test_cancellation_stops_active_worker_and_cancels_pending_lanes(config):
    store = dispatcher.make_store(config)
    hold = threading.Event()
    cancel = threading.Event()
    backend = FakeBackend(store, hold=hold)
    request = _request(lanes=[_lane("running", 4), _lane("pending", 1)])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(dispatcher.execute, request, config, backend=backend,
                             cancel=cancel, poll_seconds=0.001)
        try:
            assert backend.started.wait(5)
            assert len(_active(store)) == 1
            store.request_cancel("build")
            assert future.result(timeout=5) != 0
        finally:
            cancel.set()
            hold.set()
    assert {lane for action, lane in backend.events if action == "start"} == {"running"}
    assert _summary(config)["lanes"]["pending"]["state"] == "cancelled"
    assert not _active(store)
    assert _build(store)["state"] == "finished"


@pytest.mark.parametrize("stop_result", [False, RuntimeError("termination probe unavailable")])
def test_uncertain_cleanup_keeps_reservation_until_explicit_recovery(config, stop_result):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store, stop_result=stop_result)
    assert dispatcher.execute(_request(), config, backend=backend, poll_seconds=0.001) == 97
    assert len(_active(store)) == 1
    assert _build(store)["state"] == "admitted"
    assert _summary(config)["recovery_required"]
    with pytest.raises(RuntimeError, match="termination"):
        dispatcher.recover(config, "build", backend=backend)
    assert len(_active(store)) == 1
    assert store.is_cancelled("build")
    backend.stop_result = True
    dispatcher.recover(config, "build", backend=backend)
    assert not _active(store)
    assert _build(store)["state"] == "finished"
    dispatcher.recover(config, "build", backend=backend)


def test_recovery_refuses_a_live_dispatcher_and_keeps_its_reservation(config):
    store = dispatcher.make_store(config)
    store.register("build", "repo#1", "vllm", "a" * 40)
    assert store.try_admit("build")
    backend = FakeBackend(store)
    handle = backend.handle("build", "unit")
    assert store.try_acquire("build", "unit", 1, handle)
    with dispatcher.build_lock(dispatcher.run_directory(config, "build")):
        with pytest.raises(RuntimeError, match="dispatcher is still running"):
            dispatcher.recover(config, "build", backend=backend)
    assert not backend.events
    assert len(_active(store)) == 1
    dispatcher.recover(config, "build", backend=backend)
    assert not _active(store)


def test_finished_attempt_cannot_be_replayed(config):
    store = dispatcher.make_store(config)
    backend = FakeBackend(store)
    assert dispatcher.execute(_request(), config, backend=backend, poll_seconds=0.001) == 0
    before = list(backend.events)
    with pytest.raises(RuntimeError, match="already exists"):
        dispatcher.execute(_request(), config, backend=backend, poll_seconds=0.001)
    assert backend.events == before


def test_overlapping_same_pr_builds_share_slot_and_gpu_budget(config):
    config["max_active_prs"] = 1
    store = dispatcher.make_store(config)
    hold = threading.Event()
    cancel = threading.Event()
    first_backend = FakeBackend(store, hold=hold)
    second_backend = FakeBackend(store, hold=hold)
    first = _request("fastcheck", lanes=[_lane("encoder", 2)])
    second = _request("merge", lanes=[_lane("training", 2, False)])
    second["backend"] = "modal"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(dispatcher.execute, first, config, backend=first_backend,
                                   cancel=cancel, poll_seconds=0.001)
        second_future = None
        try:
            assert first_backend.started.wait(5)
            second_future = pool.submit(dispatcher.execute, second, config, backend=second_backend,
                                        cancel=cancel, poll_seconds=0.001)
            assert second_backend.started.wait(5)
            store.register("other-pr", "repo#other", "vllm", "b" * 40)
            assert not store.try_admit("other-pr")
            assert sum(allocation["gpus"] for allocation in _active(store, "fastcheck")
                       + _active(store, "merge")) == 4
            assert _build(store, "fastcheck")["state"] == "admitted"
            assert _build(store, "merge")["state"] == "admitted"
            hold.set()
            assert first_future.result(timeout=5) == 0
            assert second_future.result(timeout=5) == 0
        finally:
            hold.set()
            cancel.set()
    assert store.try_admit("other-pr")


def test_unverified_different_sha_does_not_cancel_existing_pr_build(config):
    # Arrival order does not establish which commit is the current PR head:
    # a delayed webhook or explicit old-SHA diagnostic may arrive last.
    store = dispatcher.make_store(config)
    store.register("current-head", "repo#1", "vllm", "b" * 40)
    assert store.try_admit("current-head")
    backend = FakeBackend(store)
    delayed = _request("delayed-older-sha", commit="a" * 40)
    assert dispatcher.execute(delayed, config, backend=backend, poll_seconds=0.001) == 0
    assert not store.is_cancelled("current-head")
    assert _build(store, "current-head")["state"] == "admitted"


def test_recovery_of_unknown_attempt_fails_without_backend_actions(config):
    backend = FakeBackend(dispatcher.make_store(config))
    with pytest.raises(ValueError, match="Unknown build"):
        dispatcher.recover(config, "not-registered", backend=backend)
    assert not backend.events
