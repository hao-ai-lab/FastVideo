# SPDX-License-Identifier: Apache-2.0
"""JobRunner's queue: order, concurrency, last-frame dependencies, and surviving a restart.

Runs without a GPU: `fastvideo` is stubbed (importing it needs a working triton
driver) and jobs "run" by waiting on an event instead of generating video.
"""
from __future__ import annotations

import collections
import importlib.util
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

from fastvideo_studio.database import Database
from fastvideo_studio.frames import FrameError
from fastvideo_studio.job_queue import deferred_last_frame_source

RUNNER_PATH = Path(__file__).resolve().parents[1] / "job_runner.py"


@pytest.fixture
def job_runner_module(monkeypatch):
    class Manager:
        def shutdown(self):
            pass

    stub_utils = types.ModuleType("fastvideo.utils")
    stub_utils.get_mp_context = lambda: types.SimpleNamespace(Manager=Manager)
    stub_pkg = types.ModuleType("fastvideo")
    stub_pkg.utils = stub_utils
    monkeypatch.setitem(sys.modules, "fastvideo", stub_pkg)
    monkeypatch.setitem(sys.modules, "fastvideo.utils", stub_utils)

    spec = importlib.util.spec_from_file_location("job_runner_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # @dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


class Harness:
    """Builds runners whose jobs finish only when the test says so."""

    def __init__(self, module, tmp_path, monkeypatch):
        self.module = module
        self.tmp_path = tmp_path
        self.gates: dict[str, threading.Event] = collections.defaultdict(threading.Event)
        self.outcomes: dict[str, object] = {}
        self.ran: list[str] = []
        self.runners = []
        self.thread_of: dict[str, threading.Thread] = {}
        self.closed = False

        harness = self

        def fake_run(runner_self, job):
            harness.thread_of[job.id] = threading.current_thread()
            harness.ran.append(job.id)
            if harness.closed:
                harness.gates[job.id].set()
            harness.gates[job.id].wait(10)
            job.status = harness.outcomes.get(job.id, module.JobStatus.COMPLETED)
            if job.status == module.JobStatus.FAILED:
                job.error = "boom"
            job.finished_at = time.time()
            runner_self._save_job(job)

        monkeypatch.setattr(module.JobRunner, "_run_job", fake_run)

    def runner(self, **kwargs):
        runner = self.module.JobRunner(
            output_dir=str(self.tmp_path / "out"),
            log_dir=str(self.tmp_path / "logs"),
            database=Database(self.tmp_path / "t.db"),
            upload_dir=str(self.tmp_path / "uploads"),
            **kwargs,
        )
        self.runners.append(runner)
        return runner

    def make(self, runner, name, **over):
        kwargs = dict(job_id=str(uuid.uuid4()), model_id="m", name=name, prompt="p")
        kwargs.update(over)
        return runner.create_job(**kwargs)

    def finish(self, job, status=None):
        if status is not None:
            self.outcomes[job.id] = status
        self.gates[job.id].set()

    def release_all(self):
        for gate in self.gates.values():
            gate.set()


def wait_for(condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def h(job_runner_module, tmp_path, monkeypatch):
    harness = Harness(job_runner_module, tmp_path, monkeypatch)
    yield harness
    # Let every job (and the queue's follow-up launches) finish while `_run_job` is
    # still patched, so no stray thread starts a real job after the test is over.
    harness.closed = True
    active = ("queued", "running")
    for runner in harness.runners:
        for _ in range(100):
            harness.release_all()
            if not any(j.status.value in active for j in runner.list_jobs()):
                break
            time.sleep(0.05)
        runner._shutdown()  # stops the worker threads


def statuses(runner, *jobs):
    return [runner.get_job(j.id).status.value for j in jobs]


def frame_of(job):
    return [{"source": deferred_last_frame_source(job.id), "media_type": "image"}]


class TestOrderAndConcurrency:
    def test_one_job_runs_at_a_time_in_the_order_queued(self, h):
        r = h.runner()
        a, b, c = (h.make(r, n) for n in "abc")
        r.enqueue_jobs([b.id, c.id, a.id])

        assert wait_for(lambda: statuses(r, a, b, c) == ["queued", "running", "queued"])
        h.finish(b)
        assert wait_for(lambda: statuses(r, a, b, c) == ["queued", "completed", "running"])
        h.finish(c)
        assert wait_for(lambda: statuses(r, a, b, c) == ["running", "completed", "completed"])
        h.finish(a)
        assert wait_for(lambda: statuses(r, a, b, c) == ["completed"] * 3)
        assert h.ran == [b.id, c.id, a.id]

    def test_runs_as_many_at_once_as_allowed(self, h):
        r = h.runner(max_concurrent_jobs=2)
        a, b, c = (h.make(r, n) for n in "abc")
        r.enqueue_jobs([a.id, b.id, c.id])
        assert wait_for(lambda: statuses(r, a, b, c) == ["running", "running", "queued"])
        h.finish(a)
        assert wait_for(lambda: statuses(r, a, b, c) == ["completed", "running", "running"])

    def test_a_failed_job_does_not_stop_the_queue(self, h):
        r = h.runner()
        a, b = h.make(r, "a"), h.make(r, "b")
        r.enqueue_jobs([a.id, b.id])
        h.finish(a, h.module.JobStatus.FAILED)
        assert wait_for(lambda: statuses(r, a, b) == ["failed", "running"])

    def test_queued_jobs_have_a_queue_time_and_running_ones_do_not(self, h):
        r = h.runner()
        a, b = h.make(r, "a"), h.make(r, "b")
        r.enqueue_jobs([a.id, b.id])
        assert wait_for(lambda: r.get_job(a.id).status.value == "running")
        assert r.get_job(a.id).queued_at is None
        assert r.get_job(b.id).queued_at is not None
        assert r.get_job(b.id).to_dict()["status"] == "queued"


class TestLongLivedWorkers:
    """Model workers die with the thread that started them, so inference jobs share long-lived threads."""

    def _run_all(self, h, r, n):
        jobs = [h.make(r, f"j{i}") for i in range(n)]
        for job in jobs:
            h.gates[job.id].set()  # each finishes as soon as it starts
        r.enqueue_jobs([j.id for j in jobs])
        assert wait_for(lambda: statuses(r, *jobs) == ["completed"] * n)
        return jobs

    def test_consecutive_jobs_run_on_the_same_thread_and_it_stays_alive(self, h):
        r = h.runner()
        jobs = self._run_all(h, r, 4)
        threads = {h.thread_of[j.id] for j in jobs}
        assert len(threads) == 1
        assert threads.pop().is_alive()

    def test_the_thread_is_not_one_made_for_the_job(self, h):
        r = h.runner()
        (job,) = self._run_all(h, r, 1)
        assert h.thread_of[job.id].name.startswith("inference-worker")

    def test_as_many_workers_as_jobs_allowed_at_once(self, h):
        r = h.runner(max_concurrent_jobs=2)
        jobs = self._run_all(h, r, 6)
        assert len({h.thread_of[j.id] for j in jobs}) <= 2

    def test_two_jobs_at_once_use_two_different_workers(self, h):
        r = h.runner(max_concurrent_jobs=2)
        a, b = h.make(r, "a"), h.make(r, "b")
        r.enqueue_jobs([a.id, b.id])
        assert wait_for(lambda: statuses(r, a, b) == ["running", "running"])
        assert wait_for(lambda: len(h.thread_of) == 2)
        assert h.thread_of[a.id] is not h.thread_of[b.id]

    def test_a_job_that_blows_up_does_not_kill_its_worker(self, h):
        r = h.runner()
        bad, good = h.make(r, "bad"), h.make(r, "good")
        original = h.module.JobRunner._run_job

        def explode_once(self, job):
            if job.id == bad.id:
                raise RuntimeError("boom")
            return original(self, job)

        h.module.JobRunner._run_job = explode_once
        h.gates[good.id].set()
        r.enqueue_jobs([bad.id, good.id])
        assert wait_for(lambda: statuses(r, bad, good) == ["failed", "completed"])
        assert "boom" in r.get_job(bad.id).error

    def test_a_direct_start_is_refused_while_the_workers_are_busy(self, h):
        r = h.runner()
        first, second = h.make(r, "first"), h.make(r, "second")
        r.start_job(first.id)
        with pytest.raises(ValueError, match="already running.*Queue this job"):
            r.start_job(second.id)
        assert r.get_job(second.id).status.value == "pending"

    def test_a_chain_runs_clean_when_every_clip_finishes_instantly(self, h):
        r = h.runner()
        clips = []
        for i in range(6):
            clips.append(h.make(r, f"clip {i}", references=frame_of(clips[-1]) if clips else []))
        for clip in clips:
            h.gates[clip.id].set()
        r.enqueue_jobs([c.id for c in clips])
        assert wait_for(lambda: statuses(r, *clips) == ["completed"] * 6)
        assert h.ran == [c.id for c in clips]


class TestGeneratorCache:
    @pytest.fixture
    def generators(self, job_runner_module, monkeypatch):
        made = []

        class FakeGenerator:
            def __init__(self, model_id):
                self.model_id = model_id
                self.shut_down = False
                made.append(self)

            def shutdown(self):
                self.shut_down = True

        stub = types.SimpleNamespace(from_pretrained=lambda model_id, *a, **k: FakeGenerator(model_id))
        monkeypatch.setattr(sys.modules["fastvideo"], "VideoGenerator", stub, raising=False)
        return made

    def _get(self, runner, model="m"):
        return runner._get_or_create_generator(model, "i2v", 1)

    def _on_thread(self, fn):
        box = []
        t = threading.Thread(target=lambda: box.append(fn()))
        t.start()
        t.join()
        return box[0]

    def test_the_thread_that_built_it_keeps_reusing_it(self, h, generators):
        r = h.runner()
        first = self._get(r)
        assert self._get(r) is first
        assert len(generators) == 1

    def test_a_generator_from_an_exited_thread_is_dropped_and_replaced(self, h, generators):
        r = h.runner()
        old = self._on_thread(lambda: self._get(r))  # that thread has exited: its workers are gone
        new = self._get(r)
        assert new is not old
        assert old.shut_down
        assert len(generators) == 2

    def test_threads_do_not_share_generators(self, h, generators):
        # Sharing would break the second the moment the first one's thread ends.
        r = h.runner()
        mine = self._get(r)
        other = self._on_thread(lambda: self._get(r))
        assert other is not mine
        assert not mine.shut_down  # still owned by a live thread

    def test_a_different_model_replaces_the_loaded_one_instead_of_sitting_beside_it(self, h, generators):
        r = h.runner()
        first = self._get(r, "model-a")
        second = self._get(r, "model-b")
        assert first.shut_down  # its GPU memory is freed before the next load
        assert not second.shut_down
        assert self._get(r, "model-b") is second

    def test_switching_back_reloads(self, h, generators):
        r = h.runner()
        a1 = self._get(r, "model-a")
        self._get(r, "model-b")
        a2 = self._get(r, "model-a")
        assert a2 is not a1

    def test_discarding_unloads_this_threads_models_only(self, h, generators):
        r = h.runner()
        mine = self._get(r)
        gate, done = threading.Event(), threading.Event()
        theirs = []

        def other_thread():
            theirs.append(self._get(r, "other"))
            done.set()
            gate.wait(5)

        t = threading.Thread(target=other_thread)
        t.start()
        done.wait(5)
        r._discard_generators()
        gate.set()
        t.join()
        assert mine.shut_down and not theirs[0].shut_down
        assert self._get(r) is not mine  # reloads fresh next time


class TestLastFrameChains:
    def _chain(self, h, r, n=3):
        clips = []
        for i in range(n):
            refs = frame_of(clips[-1]) if clips else []
            clips.append(h.make(r, f"clip {i + 1}", references=refs))
        return clips

    def test_a_scene_runs_clip_by_clip_however_many_slots_there_are(self, h):
        r = h.runner(max_concurrent_jobs=4)
        c1, c2, c3 = self._chain(h, r)
        r.enqueue_jobs([c1.id, c2.id, c3.id])

        assert wait_for(lambda: statuses(r, c1, c2, c3) == ["running", "queued", "queued"])
        time.sleep(0.05)
        assert statuses(r, c1, c2, c3) == ["running", "queued", "queued"]  # spare slots stay unused
        h.finish(c1)
        assert wait_for(lambda: statuses(r, c1, c2, c3) == ["completed", "running", "queued"])
        h.finish(c2)
        assert wait_for(lambda: statuses(r, c1, c2, c3) == ["completed", "completed", "running"])

    def test_a_clip_does_not_block_unrelated_jobs_behind_it(self, h):
        r = h.runner(max_concurrent_jobs=2)
        c1, c2 = self._chain(h, r, 2)
        other = h.make(r, "other")
        r.enqueue_jobs([c1.id, c2.id, other.id])
        assert wait_for(lambda: statuses(r, c1, c2, other) == ["running", "queued", "running"])

    def test_a_failed_clip_fails_the_clips_after_it_and_says_why(self, h):
        r = h.runner()
        c1, c2, c3 = self._chain(h, r)
        r.enqueue_jobs([c1.id, c2.id, c3.id])
        h.finish(c1, h.module.JobStatus.FAILED)

        assert wait_for(lambda: statuses(r, c1, c2, c3) == ["failed"] * 3)
        assert "clip 1" in r.get_job(c2.id).error and "failed" in r.get_job(c2.id).error
        assert "clip 2" in r.get_job(c3.id).error
        assert h.ran == [c1.id]

    def test_a_stopped_clip_fails_the_clips_after_it(self, h):
        r = h.runner()
        c1, c2 = self._chain(h, r, 2)
        r.enqueue_jobs([c1.id, c2.id])
        h.finish(c1, h.module.JobStatus.STOPPED)
        assert wait_for(lambda: statuses(r, c1, c2) == ["stopped", "failed"])
        assert "stopped" in r.get_job(c2.id).error

    def test_deleting_a_queued_clip_fails_the_clips_after_it(self, h):
        r = h.runner()
        blocker = h.make(r, "blocker")
        c1, c2 = self._chain(h, r, 2)
        r.enqueue_jobs([blocker.id, c1.id, c2.id])
        assert wait_for(lambda: statuses(r, blocker, c1, c2) == ["running", "queued", "queued"])
        assert r.delete_job(c1.id)
        assert wait_for(lambda: r.get_job(c2.id).status.value == "failed")
        assert "deleted" in r.get_job(c2.id).error

    def test_a_failed_clip_can_be_fixed_and_the_scene_requeued(self, h):
        r = h.runner()
        c1, c2 = self._chain(h, r, 2)
        r.enqueue_jobs([c1.id, c2.id])
        h.finish(c1, h.module.JobStatus.FAILED)
        assert wait_for(lambda: statuses(r, c1, c2) == ["failed", "failed"])

        h.outcomes.pop(c1.id)
        h.gates[c1.id].clear()
        r.enqueue_jobs([c1.id, c2.id])
        assert wait_for(lambda: statuses(r, c1, c2) == ["running", "queued"])
        h.finish(c1)
        assert wait_for(lambda: statuses(r, c1, c2) == ["completed", "running"])

    def test_the_stored_reference_stays_deferred(self, h):
        r = h.runner()
        c1, c2 = self._chain(h, r, 2)
        r.enqueue_jobs([c1.id, c2.id])
        h.finish(c1)
        assert wait_for(lambda: r.get_job(c2.id).status.value == "running")
        assert r.get_job(c2.id).references == frame_of(c1)


class TestStartingDirectly:
    def test_refuses_while_the_frame_source_is_unfinished(self, h):
        r = h.runner()
        c1, c2 = h.make(r, "clip 1"), None
        c2 = h.make(r, "clip 2", references=frame_of(c1))
        with pytest.raises(ValueError, match="clip 1.*hasn't finished.*pending"):
            r.start_job(c2.id)
        assert r.get_job(c2.id).status.value == "pending"
        assert h.ran == []

    def test_starts_once_the_frame_source_completed(self, h):
        r = h.runner()
        c1 = h.make(r, "clip 1")
        c1.status = h.module.JobStatus.COMPLETED
        c2 = h.make(r, "clip 2", references=frame_of(c1))
        r.start_job(c2.id)
        assert r.get_job(c2.id).status.value == "running"

    def test_refuses_when_the_frame_source_was_deleted(self, h):
        r = h.runner()
        c1 = h.make(r, "clip 1")
        c2 = h.make(r, "clip 2", references=frame_of(c1))
        r.delete_job(c1.id)
        with pytest.raises(ValueError, match="no longer exists"):
            r.start_job(c2.id)

    def test_a_direct_start_takes_a_queue_slot(self, h):
        r = h.runner()
        running, queued = h.make(r, "running"), h.make(r, "queued")
        r.start_job(running.id)
        r.enqueue_job(queued.id)
        time.sleep(0.05)
        assert statuses(r, running, queued) == ["running", "queued"]
        h.finish(running)
        assert wait_for(lambda: r.get_job(queued.id).status.value == "running")

    def test_a_queued_job_cannot_also_be_started(self, h):
        r = h.runner()
        blocker, waiting = h.make(r, "blocker"), h.make(r, "waiting")
        r.enqueue_jobs([blocker.id, waiting.id])
        with pytest.raises(ValueError, match="already queued"):
            r.start_job(waiting.id)


class TestQueueManagement:
    def test_dequeue_returns_a_job_to_pending_and_lets_others_go(self, h):
        r = h.runner()
        a, b, c = (h.make(r, n) for n in "abc")
        r.enqueue_jobs([a.id, b.id, c.id])
        assert wait_for(lambda: r.get_job(a.id).status.value == "running")
        r.dequeue_job(b.id)
        assert r.get_job(b.id).status.value == "pending"
        assert r.get_job(b.id).queued_at is None
        h.finish(a)
        assert wait_for(lambda: statuses(r, a, b, c) == ["completed", "pending", "running"])

    def test_stopping_a_queued_job_dequeues_it(self, h):
        r = h.runner()
        a, b = h.make(r, "a"), h.make(r, "b")
        r.enqueue_jobs([a.id, b.id])
        assert wait_for(lambda: r.get_job(a.id).status.value == "running")
        assert r.stop_job(b.id).status.value == "pending"

    def test_dequeue_of_a_job_not_queued_is_refused(self, h):
        r = h.runner()
        a = h.make(r, "a")
        with pytest.raises(ValueError, match="not queued"):
            r.dequeue_job(a.id)
        with pytest.raises(ValueError, match="not found"):
            r.dequeue_job("nope")

    def test_enqueue_is_all_or_nothing(self, h):
        r = h.runner()
        a, done = h.make(r, "a"), h.make(r, "done")
        done.status = h.module.JobStatus.COMPLETED
        with pytest.raises(ValueError, match="already completed"):
            r.enqueue_jobs([a.id, done.id])
        assert r.get_job(a.id).status.value == "pending"
        with pytest.raises(ValueError, match="not found"):
            r.enqueue_jobs([a.id, "nope"])
        assert h.ran == []

    def test_a_job_cannot_be_queued_twice(self, h):
        r = h.runner()
        blocker, a = h.make(r, "blocker"), h.make(r, "a")
        r.enqueue_jobs([blocker.id, a.id])
        with pytest.raises(ValueError, match="already queued"):
            r.enqueue_job(a.id)
        with pytest.raises(ValueError, match="more than once"):
            r.enqueue_jobs([a.id, a.id])

    def test_loops_of_last_frames_are_refused(self, h):
        r = h.runner()
        a = h.make(r, "a")
        b = h.make(r, "b", references=frame_of(a))
        r.update_job_config(a.id, {"references": frame_of(b)})
        with pytest.raises(ValueError, match="loop"):
            r.enqueue_jobs([a.id, b.id])
        assert statuses(r, a, b) == ["pending", "pending"]

    def test_queued_jobs_cannot_be_edited(self, h):
        r = h.runner()
        blocker, a = h.make(r, "blocker"), h.make(r, "a")
        r.enqueue_jobs([blocker.id, a.id])
        with pytest.raises(ValueError, match="queued"):
            r.update_job_config(a.id, {"prompt": "changed"})

    def test_requeueing_clears_the_previous_error(self, h):
        r = h.runner()
        a = h.make(r, "a")
        h.finish(a, h.module.JobStatus.FAILED)
        r.enqueue_job(a.id)
        assert wait_for(lambda: r.get_job(a.id).status.value == "failed")
        h.outcomes.pop(a.id)
        h.gates[a.id].clear()
        r.enqueue_job(a.id)
        assert wait_for(lambda: r.get_job(a.id).status.value == "running")
        assert r.get_job(a.id).error is None


class TestRestart:
    def test_queued_jobs_resume_in_order_after_a_restart(self, h, job_runner_module):
        first = h.runner()
        blocker = h.make(first, "blocker")
        a, b = h.make(first, "a"), h.make(first, "b")
        first.enqueue_jobs([blocker.id, b.id, a.id])
        assert wait_for(lambda: h.ran == [blocker.id])  # actually started, not just handed to a worker

        h.ran.clear()
        second = h.runner()  # same database
        # The job that was running is failed, exactly as before queues existed...
        assert second.get_job(blocker.id).status.value == "failed"
        assert "restarted" in second.get_job(blocker.id).error
        # ...and the queue carries on where it was.
        assert wait_for(lambda: h.ran[:1] == [b.id])
        assert second.get_job(a.id).status.value == "queued"

    def test_edits_survive_a_restart(self, h):
        first = h.runner()
        job = h.make(first, "a", fps=24, negative_prompt="neg")
        first.update_job_config(job.id, {"prompt": "edited", "references": frame_of(job), "fps": 12})

        second = h.runner()
        restored = second.get_job(job.id)
        assert restored.prompt == "edited"
        assert restored.references == frame_of(job)
        assert restored.fps == 12
        assert restored.negative_prompt == "neg"


class TestTrimming:
    """trim_job/restore_job_video against a real (small, synthetic) video file."""

    FPS = 24

    def _write_video(self, path, n=48):
        import av
        import numpy as np

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        container = av.open(path, mode="w")
        try:
            stream = container.add_stream("h264", rate=self.FPS)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(n):
                array = np.full((48, 64, 3), min(i * 5, 255), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def _completed(self, h, r, n=48):
        job = h.make(r, "clip")
        r.start_job(job.id)
        h.finish(job)
        assert wait_for(lambda: r.get_job(job.id).status.value == "completed")
        live = r.get_job(job.id)
        path = str(h.tmp_path / "videos" / f"{job.id}.mp4")
        self._write_video(path, n)
        live.output_path = path
        live.num_frames = n
        return live

    def test_trim_updates_output_path_and_frame_count(self, h):
        r = h.runner()
        job = self._completed(h, r)  # 2s @ 24fps
        original_path = job.output_path
        trimmed = r.trim_job(job.id, start_seconds=0.5, end_seconds=1.5)
        assert trimmed.output_path != original_path
        assert trimmed.output_path.endswith("edited.mp4")
        assert trimmed.num_frames == self.FPS  # 1s kept

    def test_grading_updates_output_path_without_changing_frame_count(self, h):
        r = h.runner()
        job = self._completed(h, r)
        graded = r.trim_job(job.id, start_seconds=0.0, end_seconds=None, brightness=40.0)
        assert graded.output_path.endswith("edited.mp4")
        assert graded.num_frames == self.FPS * 2  # full length kept, only color changed

    def test_grading_and_a_trim_requested_together_both_take_effect(self, h):
        import av

        r = h.runner()
        job = self._completed(h, r)
        edited = r.trim_job(job.id, start_seconds=0.5, end_seconds=1.5, brightness=80.0)
        assert edited.num_frames == self.FPS  # the range applied
        with av.open(edited.output_path) as c:
            frame = next(c.decode(c.streams.video[0]))
        # Source frame at 0.5s (index 12) is level 60; +80 brightness should clearly lighten it.
        assert int(frame.to_ndarray(format="rgb24")[0, 0, 0]) > 60 + 40

    def test_trim_persists_across_a_restart(self, h):
        first = h.runner()
        job = self._completed(h, first)
        trimmed = first.trim_job(job.id, start_seconds=0.5, end_seconds=1.5)

        second = h.runner()
        restored = second.get_job(job.id)
        assert restored.output_path == trimmed.output_path
        assert restored.num_frames == trimmed.num_frames

    def test_trimming_again_is_relative_to_the_original_not_the_last_trim(self, h):
        r = h.runner()
        job = self._completed(h, r)
        r.trim_job(job.id, start_seconds=0.5, end_seconds=1.5)
        again = r.trim_job(job.id, start_seconds=0.0, end_seconds=2.0)
        assert again.num_frames == self.FPS * 2

    def test_restore_undoes_a_trim(self, h):
        r = h.runner()
        job = self._completed(h, r)
        r.trim_job(job.id, start_seconds=0.5, end_seconds=1.5)
        restored = r.restore_job_video(job.id)
        # Points at the untouched backup, not literally the job's original path --
        # trim_job() never overwrites or renames what H3 actually generated.
        assert restored.output_path.endswith("original.mp4")
        assert restored.num_frames == self.FPS * 2

    def test_restore_refuses_a_job_never_edited(self, h):
        r = h.runner()
        job = self._completed(h, r)
        with pytest.raises(FrameError, match="not been edited"):
            r.restore_job_video(job.id)

    def test_restore_undoes_a_grade_too(self, h):
        import av

        r = h.runner()
        job = self._completed(h, r)
        graded = r.trim_job(job.id, start_seconds=0.0, end_seconds=None, brightness=80.0)
        with av.open(graded.output_path) as c:
            graded_level = int(next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24")[0, 0, 0])
        assert graded_level > 60  # the grade visibly lightened the first frame (source level 0)

        restored = r.restore_job_video(job.id)
        with av.open(restored.output_path) as c:
            restored_level = int(next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24")[0, 0, 0])
        assert restored_level == 0  # back to the source's actual first-frame level

    def test_trim_refuses_an_unknown_job(self, h):
        r = h.runner()
        with pytest.raises(ValueError, match="not found"):
            r.trim_job("nope", 0.0, 1.0)

    def test_trim_refuses_a_pending_job(self, h):
        r = h.runner()
        job = h.make(r, "pending")
        with pytest.raises(FrameError, match="No output"):
            r.trim_job(job.id, 0.0, 1.0)
