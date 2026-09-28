# SPDX-License-Identifier: Apache-2.0
"""Queued jobs run in order, one at a time, and a clip waits for the clip whose last frame it uses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fastvideo_studio.job_queue import (
    circular_dependency,
    deferred_last_clip_source,
    deferred_last_frame_source,
    dependencies,
    dependency_of,
    plan_dispatch,
    resolve_references,
    UnresolvedReference,
)


@dataclass
class J:
    id: str
    status: str = "pending"
    queued_at: float | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    name: str = ""


def after(job_id: str) -> list[dict[str, str]]:
    return [{"source": deferred_last_frame_source(job_id), "media_type": "image"}]


def queue(*jobs: J) -> dict[str, J]:
    for i, job in enumerate(jobs):
        if job.status == "queued" and job.queued_at is None:
            job.queued_at = float(i)
    return {j.id: j for j in jobs}


def test_deferred_source_round_trips():
    assert dependency_of({"source": deferred_last_frame_source("abc")}) == "abc"


@pytest.mark.parametrize("ref", [None, {}, {"source": "/tmp/a.png"}, {"source": "job-last-frame:"}, {"source": 3}])
def test_ordinary_references_have_no_dependency(ref):
    assert dependency_of(ref) is None


def test_dependencies_keep_order_and_drop_repeats():
    refs = [*after("b"), {"source": "/x.png"}, *after("a"), *after("b")]
    assert dependencies(refs) == ["b", "a"]


class TestDispatchOrder:
    def test_starts_the_oldest_queued_job_first(self):
        jobs = queue(J("late", "queued", 5.0), J("early", "queued", 1.0))
        assert plan_dispatch(jobs, 1).start == ["early"]

    def test_runs_only_max_concurrent_including_running_jobs(self):
        jobs = queue(J("r", "running"), J("a", "queued"), J("b", "queued"))
        assert plan_dispatch(jobs, 1).start == []
        assert plan_dispatch(jobs, 2).start == ["a"]
        assert plan_dispatch(jobs, 3).start == ["a", "b"]

    def test_ignores_jobs_that_are_not_queued(self):
        jobs = queue(J("p"), J("c", "completed"), J("f", "failed"), J("s", "stopped"))
        assert plan_dispatch(jobs, 4).start == []

    def test_a_limit_below_one_still_runs_one_job(self):
        assert plan_dispatch(queue(J("a", "queued")), 0).start == ["a"]

    def test_ties_are_broken_by_id_for_a_stable_order(self):
        jobs = {"b": J("b", "queued", 1.0), "a": J("a", "queued", 1.0)}
        assert plan_dispatch(jobs, 2).start == ["a", "b"]


class TestWaitingOnAFrame:
    def test_waits_while_the_source_clip_is_running_or_queued(self):
        for source_status in ("running", "queued"):
            jobs = queue(J("one", source_status), J("two", "queued", references=after("one")))
            plan = plan_dispatch(jobs, 2)
            assert "two" not in plan.start and not plan.fail

    def test_starts_once_the_source_clip_completed(self):
        jobs = queue(J("one", "completed"), J("two", "queued", references=after("one")))
        assert plan_dispatch(jobs, 1).start == ["two"]

    def test_waiting_does_not_hold_up_later_jobs(self):
        jobs = queue(
            J("one", "running"),
            J("two", "queued", references=after("one")),
            J("free", "queued"),
        )
        assert plan_dispatch(jobs, 2).start == ["free"]

    def test_a_source_that_is_only_pending_keeps_waiting(self):
        # The user may still queue it.
        jobs = queue(J("one", "pending"), J("two", "queued", references=after("one")))
        plan = plan_dispatch(jobs, 1)
        assert plan.start == [] and plan.fail == {}

    def test_needs_every_source(self):
        refs = [*after("a"), *after("b")]
        jobs = queue(J("a", "completed"), J("b", "running"), J("c", "queued", references=refs))
        assert plan_dispatch(jobs, 2).start == []
        jobs["b"].status = "completed"
        assert plan_dispatch(jobs, 2).start == ["c"]

    def test_a_chain_advances_one_clip_at_a_time(self):
        jobs = queue(
            J("c1", "queued"),
            J("c2", "queued", references=after("c1")),
            J("c3", "queued", references=after("c2")),
        )
        assert plan_dispatch(jobs, 5).start == ["c1"]
        jobs["c1"].status = "completed"
        assert plan_dispatch(jobs, 5).start == ["c2"]
        jobs["c2"].status = "completed"
        assert plan_dispatch(jobs, 5).start == ["c3"]


class TestImpossibleJobs:
    @pytest.mark.parametrize(("status", "word"), [("failed", "failed"), ("stopped", "stopped")])
    def test_fails_when_the_source_failed_or_was_stopped(self, status, word):
        jobs = queue(J("one", status, name="Clip 1"), J("two", "queued", references=after("one")))
        plan = plan_dispatch(jobs, 1)
        assert plan.start == []
        assert "Clip 1" in plan.fail["two"] and word in plan.fail["two"]

    def test_fails_when_the_source_was_deleted(self):
        jobs = queue(J("two", "queued", references=after("gone")))
        assert "deleted" in plan_dispatch(jobs, 1).fail["two"]

    def test_failure_cascades_down_the_chain(self):
        jobs = queue(
            J("c1", "failed"),
            J("c2", "queued", references=after("c1")),
            J("c3", "queued", references=after("c2")),
            J("c4", "queued", references=after("c3")),
            J("other", "queued"),
        )
        plan = plan_dispatch(jobs, 3)
        assert set(plan.fail) == {"c2", "c3", "c4"}
        assert plan.start == ["other"]

    def test_a_failed_job_frees_its_slot_for_the_next(self):
        jobs = queue(
            J("c1", "failed"),
            J("c2", "queued", references=after("c1")),
            J("next", "queued"),
        )
        assert plan_dispatch(jobs, 1).start == ["next"]

    def test_planning_does_not_change_the_jobs(self):
        jobs = queue(J("c1", "failed"), J("c2", "queued", references=after("c1")))
        plan_dispatch(jobs, 1)
        assert jobs["c2"].status == "queued"


class TestCircularDependency:
    def test_none_for_a_plain_chain(self):
        jobs = queue(J("a"), J("b", references=after("a")), J("c", references=after("b")))
        assert circular_dependency(jobs, "c") is None

    def test_finds_a_loop_through_the_job(self):
        jobs = queue(J("a", references=after("c")), J("b", references=after("a")), J("c", references=after("b")))
        assert circular_dependency(jobs, "a") is not None

    def test_a_job_that_waits_on_itself(self):
        assert circular_dependency(queue(J("a", references=after("a"))), "a") is not None

    def test_a_loop_elsewhere_is_not_this_jobs_problem(self):
        jobs = queue(
            J("a", references=after("b")),
            J("b", references=after("a")),
            J("c", references=after("a")),
        )
        assert circular_dependency(jobs, "c") is None


class TestResolveReferences:
    def _jobs(self, **statuses: str) -> dict[str, J]:
        return {k: J(k, v, name=f"Clip {k}") for k, v in statuses.items()}

    def test_replaces_a_deferred_reference_with_the_extracted_frame(self):
        jobs = self._jobs(one="completed")
        stored = [{"source": "/ref.png", "media_type": "image"}, *after("one")]
        resolved = resolve_references(stored, jobs.get, lambda job: f"/frames/{job.id}.png")
        assert resolved == [
            {"source": "/ref.png", "media_type": "image"},
            {"source": "/frames/one.png", "media_type": "image"},
        ]

    def test_leaves_the_stored_references_untouched(self):
        jobs = self._jobs(one="completed")
        stored = after("one")
        resolve_references(stored, jobs.get, lambda job: "/f.png")
        assert stored == after("one")

    def test_no_references(self):
        assert resolve_references(None, {}.get, lambda job: "") == []

    def test_refuses_an_unfinished_source(self):
        jobs = self._jobs(one="running")
        with pytest.raises(UnresolvedReference, match="hasn't finished.*running"):
            resolve_references(after("one"), jobs.get, lambda job: "/f.png")

    def test_refuses_a_deleted_source(self):
        with pytest.raises(UnresolvedReference, match="no longer exists"):
            resolve_references(after("gone"), {}.get, lambda job: "/f.png")

    def test_passes_on_the_extraction_error(self):
        class Boom(Exception):
            detail = "The video has no frames."

        def fail(job):
            raise Boom()

        with pytest.raises(UnresolvedReference, match="no frames"):
            resolve_references(after("one"), self._jobs(one="completed").get, fail)


def after_clip(job_id: str) -> list[dict[str, str]]:
    return [{"source": deferred_last_clip_source(job_id), "media_type": "video"}]


class TestResolveClipReferences:
    """The same resolution, but for a trailing-clip (job-last-clip:) reference."""

    def _jobs(self, **statuses: str) -> dict[str, J]:
        return {k: J(k, v, name=f"Clip {k}") for k, v in statuses.items()}

    def test_uses_the_clip_resolver_not_the_frame_one(self):
        jobs = self._jobs(one="completed")
        stored = after_clip("one")
        resolved = resolve_references(
            stored, jobs.get,
            last_frame=lambda job: f"/frames/{job.id}.png",
            last_clip=lambda job: f"/clips/{job.id}.mp4",
        )
        assert resolved == [{"source": "/clips/one.mp4", "media_type": "video"}]

    def test_a_mix_of_frame_and_clip_references_each_use_their_own_resolver(self):
        jobs = self._jobs(a="completed", b="completed")
        stored = [*after("a"), *after_clip("b"), {"source": "/x.png", "media_type": "image"}]
        resolved = resolve_references(
            stored, jobs.get,
            last_frame=lambda job: f"/frames/{job.id}.png",
            last_clip=lambda job: f"/clips/{job.id}.mp4",
        )
        assert resolved == [
            {"source": "/frames/a.png", "media_type": "image"},
            {"source": "/clips/b.mp4", "media_type": "video"},
            {"source": "/x.png", "media_type": "image"},
        ]

    def test_refuses_an_unfinished_source(self):
        jobs = self._jobs(one="running")
        with pytest.raises(UnresolvedReference, match="hasn't finished.*running"):
            resolve_references(after_clip("one"), jobs.get, last_frame=lambda j: "", last_clip=lambda j: "/c.mp4")

    def test_refuses_a_deleted_source(self):
        with pytest.raises(UnresolvedReference, match="no longer exists"):
            resolve_references(after_clip("gone"), {}.get, last_frame=lambda j: "", last_clip=lambda j: "/c.mp4")

    def test_without_a_clip_resolver_a_clip_reference_is_refused_not_ignored(self):
        jobs = self._jobs(one="completed")
        with pytest.raises(UnresolvedReference, match="can't resolve"):
            resolve_references(after_clip("one"), jobs.get, last_frame=lambda j: "/f.png")

    def test_dependencies_and_circular_checks_see_clip_references_too(self):
        jobs = self._jobs(a="pending")
        jobs["a"].references = after_clip("a")
        assert dependency_of(after_clip("x")[0]) == "x"
        assert dependencies(after_clip("a") + after("a")) == ["a"]
        assert circular_dependency(jobs, "a") is not None
