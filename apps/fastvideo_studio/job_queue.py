# SPDX-License-Identifier: Apache-2.0
"""Scheduling rules for queued jobs, and last-frame references between them.

Kept apart from ``job_runner.py`` because it is pure -- it works on plain
snapshots and needs neither FastVideo nor a GPU -- so it can be tested anywhere.

A job can start from the last frame of another job's video. That frame doesn't
exist until the other job finishes, so the reference is *deferred*: its source
is ``job-last-frame:<job id>`` and it is resolved to a real image only when the
job runs. That link is also the queue's dependency graph: a queued job waits
until every job it takes a frame from has completed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFERRED_LAST_FRAME_PREFIX = "job-last-frame:"

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
PENDING = "pending"
FAILED = "failed"
STOPPED = "stopped"


def deferred_last_frame_source(job_id: str) -> str:
    return f"{DEFERRED_LAST_FRAME_PREFIX}{job_id}"


def dependency_of(reference: Mapping[str, Any] | None) -> str | None:
    """The job a deferred reference takes its frame from, or None for any other."""
    source = (reference or {}).get("source")
    if isinstance(source, str) and source.startswith(DEFERRED_LAST_FRAME_PREFIX):
        return source[len(DEFERRED_LAST_FRAME_PREFIX):] or None
    return None


def dependencies(references: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """Job ids a job needs finished first, in reference order, without repeats."""
    found: list[str] = []
    for ref in references or []:
        dep = dependency_of(ref)
        if dep and dep not in found:
            found.append(dep)
    return found


class JobView(Protocol):
    """The parts of a job the scheduler looks at."""

    id: str
    references: Any
    queued_at: float | None

    @property
    def status(self) -> Any:  # a JobStatus (str enum) or plain str
        ...


def _status(job: Any) -> str:
    return str(getattr(job.status, "value", job.status))


def circular_dependency(jobs: Mapping[str, JobView], job_id: str) -> list[str] | None:
    """The chain of job ids if ``job_id`` (transitively) waits on itself, else None."""

    def walk(current: str, path: list[str]) -> list[str] | None:
        job = jobs.get(current)
        if job is None:
            return None
        for dep in dependencies(job.references):
            if dep == job_id:
                return [*path, current, dep]
            if dep in path or dep == current:
                continue
            found = walk(dep, [*path, current])
            if found:
                return found
        return None

    return walk(job_id, [])


def _label(job: Any) -> str:
    name = getattr(job, "name", "") or ""
    return f"'{name}'" if name else job.id


@dataclass
class Plan:
    """What to do now: jobs to launch, and jobs that can never run."""

    start: list[str] = field(default_factory=list)
    fail: dict[str, str] = field(default_factory=dict)  # job id -> reason


def plan_dispatch(jobs: Mapping[str, JobView], max_concurrent: int) -> Plan:
    """Decide which queued jobs launch now and which have become impossible.

    * Queued jobs run oldest-queued first, at most ``max_concurrent`` at a time
      (counting jobs already running).
    * A job whose frame source hasn't completed *waits*, and doesn't hold up
      jobs behind it that don't need it.
    * A job whose frame source failed, was stopped, or was deleted can never
      run, so it fails -- and so does anything waiting on *it*.
    * A job whose frame source is idle (neither queued nor running) keeps
      waiting: the user may still queue it.
    """
    plan = Plan()
    status = {jid: _status(job) for jid, job in jobs.items()}
    queued = sorted(
        (job for job in jobs.values() if status[job.id] == QUEUED),
        key=lambda job: (job.queued_at or 0.0, job.id),
    )

    # Failures cascade down a chain, so repeat until nothing new fails.
    changed = True
    while changed:
        changed = False
        for job in queued:
            if job.id in plan.fail:
                continue
            for dep_id in dependencies(job.references):
                dep = jobs.get(dep_id)
                if dep is None:
                    plan.fail[job.id] = "Waiting on the last frame of a job that was deleted."
                elif dep_id in plan.fail or status[dep_id] in (FAILED, STOPPED):
                    reason = "failed" if dep_id in plan.fail or status[dep_id] == FAILED else "was stopped"
                    plan.fail[job.id] = f"Needs the last frame of {_label(dep)}, which {reason}."
                else:
                    continue
                status[job.id] = FAILED
                changed = True
                break

    slots = max(max_concurrent, 1) - sum(1 for s in status.values() if s == RUNNING)
    for job in queued:
        if slots <= 0:
            break
        if job.id in plan.fail:
            continue
        if all(status.get(dep) == COMPLETED for dep in dependencies(job.references)):
            plan.start.append(job.id)
            slots -= 1
    return plan


class UnresolvedReference(ValueError):
    """A deferred reference can't be turned into an image."""


def resolve_references(
    references: list[dict[str, Any]] | None,
    get_job: Callable[[str], Any],
    last_frame: Callable[[Any], str],
) -> list[dict[str, Any]]:
    """Copy of ``references`` with each deferred last-frame reference made real.

    ``last_frame(job)`` returns the path of an extracted frame and may raise
    (see ``frames.FrameError``); its message is passed on. Other references are
    returned unchanged, and the stored list is never modified, so a re-run
    resolves again against the source job's current output.
    """
    resolved: list[dict[str, Any]] = []
    for ref in references or []:
        dep_id = dependency_of(ref)
        if dep_id is None:
            resolved.append(ref)
            continue
        dep = get_job(dep_id)
        if dep is None:
            raise UnresolvedReference("A reference needs the last frame of a job that no longer exists.")
        if _status(dep) != COMPLETED:
            raise UnresolvedReference(f"A reference needs the last frame of {_label(dep)}, which hasn't finished "
                                  f"(it is {_status(dep)}).")
        try:
            path = last_frame(dep)
        except Exception as exc:  # FrameError carries a readable .detail
            raise UnresolvedReference(str(getattr(exc, "detail", None) or exc)) from exc
        resolved.append({**ref, "source": path})
    return resolved
