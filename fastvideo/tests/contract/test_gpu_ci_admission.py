# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission tests; no FastVideo, GPU, or backend service imports."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ADMISSION_PATH = Path(__file__).resolve().parents[3] / "scripts/gpu_ci/admission.py"
SPEC = importlib.util.spec_from_file_location("gpu_ci_admission", ADMISSION_PATH)
assert SPEC is not None and SPEC.loader is not None
ADMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMISSION)
AdmissionStore = ADMISSION.AdmissionStore


def _active(store):
    return [allocation for allocation in store.snapshot()["allocations"] if allocation["state"] == "active"]


def _register(store, build, pr=None, backend="kubernetes", commit="a" * 40):
    store.register(build, pr, backend, commit)


def test_same_pr_builds_share_gpu_budget_and_hold_slot_across_lane_gaps(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "fastcheck", "repo#1")
    _register(store, "other-pr", "repo#2")
    assert store.try_admit("fastcheck")
    assert store.try_admit("other-pr")
    _register(store, "merge", "repo#1", backend="modal")
    _register(store, "direct", "repo#1", commit="b" * 40)
    _register(store, "waiting", "repo#3")
    assert store.try_admit("merge")
    assert store.try_admit("direct")
    assert not store.try_admit("waiting")
    assert store.try_acquire("fastcheck", "encoder", 2, {"job": "encoder"})
    assert store.try_acquire("merge", "training", 2, {"call": "training"})
    assert not store.try_acquire("direct", "vae", 1, {"job": "vae"})
    store.release("fastcheck", "encoder")
    store.finish("fastcheck")
    store.release("merge", "training")
    store.finish("merge")
    assert not store.try_admit("waiting")
    assert store.try_acquire("direct", "vae", 1, {"job": "vae"})
    store.release("direct", "vae")
    assert not store.try_admit("waiting")
    store.finish("direct")
    assert store.try_admit("waiting")


def test_new_pr_admission_is_fifo_even_when_later_pr_polls_first(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    for index in range(4):
        _register(store, f"build-{index}", f"repo#{index}")
    assert not store.try_admit("build-2")
    assert store.try_admit("build-1")
    assert not store.try_admit("build-2")
    assert store.try_admit("build-0")
    store.finish("build-1")
    assert not store.try_admit("build-3")
    assert store.try_admit("build-2")


def test_non_pr_builds_get_distinct_slots(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "schedule-1")
    _register(store, "schedule-2", "")
    _register(store, "schedule-3")
    assert store.try_admit("schedule-1")
    assert store.try_admit("schedule-2")
    assert not store.try_admit("schedule-3")
    store.finish("schedule-1")
    assert store.try_admit("schedule-3")


def test_gpu_queue_does_not_starve_large_lanes_or_block_other_pr_budget(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "a", "repo#1")
    _register(store, "b", "repo#2")
    assert store.try_admit("a") and store.try_admit("b")
    assert store.try_acquire("a", "running", 3, {"job": "running"})
    assert not store.try_acquire("a", "older", 2, {"job": "older"})
    assert not store.try_acquire("a", "younger", 1, {"job": "younger"})
    assert store.try_acquire("b", "independent", 4, {"job": "independent"})
    store.release("a", "running")
    assert not store.try_acquire("a", "younger", 1, {"job": "younger"})
    assert store.try_acquire("a", "older", 2, {"job": "older"})
    assert store.try_acquire("a", "younger", 1, {"job": "younger"})


def test_gpu_queue_preserves_global_capacity_for_oldest_eligible_request(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db", max_gpus=4)
    for build in ("a", "b"):
        _register(store, build, f"repo#{build}")
        assert store.try_admit(build)
    assert store.try_acquire("a", "running", 3, {"job": "running"})
    assert not store.try_acquire("b", "older", 2, {"job": "older"})
    assert not store.try_acquire("a", "younger", 1, {"job": "younger"})
    store.release("a", "running")
    assert not store.try_acquire("a", "younger", 1, {"job": "younger"})
    assert store.try_acquire("b", "older", 2, {"job": "older"})
    assert store.try_acquire("a", "younger", 1, {"job": "younger"})


def test_restart_retains_handles_stale_allocations_and_pr_slots(tmp_path, monkeypatch):
    path = tmp_path / "admission.db"
    store = AdmissionStore(path, max_prs=1)
    with monkeypatch.context() as clock:
        clock.setattr(ADMISSION.time, "time", lambda: 1.0)
        _register(store, "running", "repo#1")
        assert store.try_admit("running")
        assert store.try_acquire("running", "lane", 4, {"namespace": "vllm", "job": "known-before-create"})
    restarted = AdmissionStore(path, max_prs=1)
    _register(restarted, "waiting", "repo#2")
    assert not restarted.try_admit("waiting")
    assert _active(restarted)[0]["handle"] == {"namespace": "vllm", "job": "known-before-create"}
    with pytest.raises(RuntimeError, match="active GPU allocations"):
        restarted.finish("running")
    restarted.heartbeat("running")
    assert len(_active(restarted)) == 1
    restarted.release("running", "lane")
    assert not restarted.try_admit("waiting")
    restarted.finish("running")
    assert restarted.try_admit("waiting")


def test_idempotent_reservations_do_not_resurrect_released_work(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "build", "repo#1")
    assert store.try_admit("build")
    _register(store, "build", "repo#1")
    assert store.try_acquire("build", "lane", 2, {"job": "one", "namespace": "vllm"})
    assert store.try_acquire("build", "lane", 2, {"namespace": "vllm", "job": "one"})
    assert sum(row["gpus"] for row in _active(store)) == 2
    with pytest.raises(ValueError, match="identity changed"):
        store.try_acquire("build", "lane", 2, {"job": "two"})
    with pytest.raises(ValueError, match="identity changed"):
        store.try_acquire("build", "lane", 3, {"job": "one", "namespace": "vllm"})
    store.release("build", "lane")
    store.release("build", "lane")
    assert not store.try_acquire("build", "lane", 2, {"job": "one", "namespace": "vllm"})
    store.finish("build")
    _register(store, "build", "repo#1")
    store.heartbeat("build")
    assert not store.try_admit("build")
    assert not store.try_acquire("build", "new-lane", 1, {"job": "late"})
    assert store.snapshot()["builds"][0]["state"] == "finished"


@pytest.mark.parametrize("changed", [{"pr": "repo#2"}, {"backend": "modal"}, {"commit": "b" * 40}])
def test_existing_build_identity_cannot_be_replaced(tmp_path, changed):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "build", "repo#1")
    values = {"pr": "repo#1", "backend": "kubernetes", "commit": "a" * 40}
    values.update(changed)
    with pytest.raises(ValueError, match="Build identity changed"):
        _register(store, "build", **values)


def test_cancel_blocks_launches_but_keeps_active_reservation_until_confirmed_stop(tmp_path):
    path = tmp_path / "admission.db"
    store = AdmissionStore(path, max_prs=1)
    _register(store, "running", "repo#1")
    _register(store, "waiting", "repo#2")
    assert store.try_admit("running")
    assert store.try_acquire("running", "active", 4, {"job": "active"})
    assert not store.try_acquire("running", "queued", 1, {"job": "queued"})
    store.request_cancel("running")
    restarted = AdmissionStore(path, max_prs=1)
    assert restarted.is_cancelled("running")
    assert not restarted.try_admit("running")
    assert not restarted.try_acquire("running", "active", 4, {"job": "active"})
    assert not restarted.try_acquire("running", "new", 1, {"job": "new"})
    assert not restarted.try_admit("waiting")
    with pytest.raises(RuntimeError, match="active GPU allocations"):
        restarted.finish("running")
    snapshot = restarted.snapshot()
    assert {row["lane_id"]: row["state"] for row in snapshot["allocations"]} == {
        "active": "active", "queued": "cancelled"
    }
    restarted.release("running", "active")
    restarted.finish("running")
    assert restarted.try_admit("waiting")


def test_cancelled_queued_pr_does_not_block_fifo_and_can_finish(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db", max_prs=1)
    _register(store, "cancelled", "repo#1")
    _register(store, "next", "repo#2")
    store.request_cancel("cancelled")
    store.request_cancel("cancelled")
    assert not store.try_admit("cancelled")
    assert store.try_admit("next")
    store.finish("cancelled")
    store.finish("cancelled")
    assert store.snapshot()["builds"][0]["cancelled"]


def test_conflicting_limits_are_rejected_without_changing_ledger(tmp_path):
    path = tmp_path / "admission.db"
    store = AdmissionStore(path)
    _register(store, "build", "repo#1")
    for changed in ({"max_prs": 3}, {"max_gpus_per_pr": 5}, {"max_gpus": 9}):
        with pytest.raises(ValueError, match="limits conflict"):
            AdmissionStore(path, **changed)
    assert AdmissionStore(path).snapshot() == store.snapshot()


@pytest.mark.parametrize("gpus", [0, -1, 5, True, 1.5])
def test_impossible_gpu_requests_are_rejected(tmp_path, gpus):
    store = AdmissionStore(tmp_path / "admission.db")
    _register(store, "build", "repo#1")
    assert store.try_admit("build")
    with pytest.raises(ValueError, match="GPU"):
        store.try_acquire("build", "bad", gpus, {"job": "bad"})
    assert not store.snapshot()["allocations"]


def test_unknown_build_operations_fail_closed(tmp_path):
    store = AdmissionStore(tmp_path / "admission.db")
    for operation in (store.try_admit, store.finish, store.heartbeat, store.request_cancel, store.is_cancelled):
        with pytest.raises(KeyError, match="Unknown build"):
            operation("unknown")
    with pytest.raises(KeyError, match="Unknown build"):
        store.try_acquire("unknown", "lane", 1, {"job": "unknown"})


_CONTENDER = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location('admission', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = json.loads(sys.argv[3])
store = module.AdmissionStore(sys.argv[2], **request['limits'])
print('ready', flush=True)
assert sys.stdin.readline().strip() == 'go'
admitted = store.try_admit(request['build'])
acquired = admitted and store.try_acquire(
    request['build'], request['lane'], request['gpus'], {'job': request['handle']})
print(json.dumps({'admitted': admitted, 'acquired': acquired}), flush=True)
"""


def _run_contenders(path, requests, limits):
    processes = []
    try:
        for request in requests:
            process = subprocess.Popen(
                [sys.executable, "-c", _CONTENDER, str(ADMISSION_PATH), str(path),
                 json.dumps({**request, "limits": limits})],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            processes.append(process)
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        results = []
        for process in processes:
            output, error = process.communicate(timeout=30)
            assert process.returncode == 0, error
            results.append(json.loads(output))
        return results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


@pytest.mark.parametrize("pr_keys,gpus,max_gpus,expected_prs,expected_gpus", [
    (["repo#1", "repo#2", "repo#3", "repo#4"], 4, 8, 2, 8),
    (["repo#1"] * 8, 1, 8, 1, 4),
    (["repo#1"] * 4 + ["repo#2"] * 4, 2, 6, 2, 6),
])
def test_multiple_processes_cannot_exceed_pr_or_gpu_capacity(
    tmp_path, pr_keys, gpus, max_gpus, expected_prs, expected_gpus,
):
    path = tmp_path / "admission.db"
    limits = {"max_prs": 2, "max_gpus_per_pr": 4, "max_gpus": max_gpus}
    store = AdmissionStore(path, **limits)
    requests = []
    for index, pr_key in enumerate(pr_keys):
        build = f"build-{index}"
        _register(store, build, pr_key, backend="modal" if index % 2 else "kubernetes")
        requests.append({"build": build, "lane": "lane", "gpus": gpus, "handle": f"job-{index}"})
    _run_contenders(path, requests, limits)
    snapshot = store.snapshot()
    builds = {build["build_id"]: build for build in snapshot["builds"]}
    assert len({build["pr_key"] for build in builds.values() if build["state"] == "admitted"}) == expected_prs
    usage = {}
    for allocation in _active(store):
        pr_key = builds[allocation["build_id"]]["pr_key"]
        usage[pr_key] = usage.get(pr_key, 0) + allocation["gpus"]
    assert sum(usage.values()) == expected_gpus
    assert all(count <= 4 for count in usage.values())


def test_simultaneous_duplicate_retries_share_one_reservation(tmp_path):
    path = tmp_path / "admission.db"
    store = AdmissionStore(path)
    _register(store, "build", "repo#1")
    request = {"build": "build", "lane": "lane", "gpus": 4, "handle": "one-worker"}
    results = _run_contenders(path, [request] * 6, {})
    assert all(result["acquired"] for result in results)
    assert len(_active(store)) == 1
    assert _active(store)[0]["gpus"] == 4
