# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.models.loader.startup import collect_startup_timings, startup_phase


def test_nested_intervals_are_inclusive_and_serializable():
    import json
    syncs = []
    with collect_startup_timings(synchronize=lambda: syncs.append(True)) as timings:
        with startup_phase("load"):
            with startup_phase("read"):
                pass
    result = timings.to_dict()
    outer, inner = result["phases"]
    assert outer["parent_id"] is None
    assert inner["parent_id"] == outer["id"]
    assert outer["duration_seconds"] >= inner["duration_seconds"] >= 0
    assert result["recording_seconds"] >= outer["duration_seconds"]
    assert result["timing_mode"] == "synchronized_boundaries"
    assert len(syncs) == 4
    json.dumps(result)


def test_error_is_recorded_and_collector_is_restored():
    with collect_startup_timings() as outer:
        with pytest.raises(ValueError, match="checkpoint"):
            with collect_startup_timings() as failed:
                with startup_phase("read"):
                    raise ValueError("checkpoint")
        with startup_phase("recovery"):
            pass
    assert [p["name"] for p in outer.phases] == ["recovery"]
    assert failed.phases[0]["status"] == "error"
    assert outer.phases[0]["parent_id"] is None
    with startup_phase("disabled"):
        pass
    assert len(outer.phases) == 1


def test_disabled_collection_does_not_read_the_clock(monkeypatch):
    from fastvideo.models.loader import startup
    def unexpected():
        raise AssertionError("disabled collection measured work")
    monkeypatch.setattr(startup.time, "perf_counter", unexpected)
    with startup_phase("disabled"):
        pass


def test_sync_failure_does_not_leave_a_stale_parent():
    calls = 0
    def synchronize():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("CUDA failure")
    with collect_startup_timings(synchronize=synchronize) as timings:
        with pytest.raises(RuntimeError, match="CUDA failure"):
            with startup_phase("failed"):
                pass
        with startup_phase("next"):
            pass
    assert timings.phases[0]["status"] == "error"
    assert timings.phases[1]["parent_id"] is None
