# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for inference ownership, queueing and failure recovery."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo_studio.database import Database
from fastvideo_studio.job_runner import JobRunner, JobStatus


@pytest.fixture
def runner(tmp_path):
    result = JobRunner(str(tmp_path / 'out'), str(tmp_path / 'logs'), Database(tmp_path / 'jobs.db'))
    yield result
    result._shutdown()


def wait_done(job):
    deadline = time.monotonic() + 10
    while job.status.value not in ('completed', 'failed', 'stopped'):
        assert time.monotonic() < deadline, job.to_dict()
        time.sleep(0.01)


def test_failed_load_cannot_destroy_another_jobs_parallel_state(runner, monkeypatch):
    from fastvideo import VideoGenerator
    from fastvideo.distributed import parallel_state
    entered = threading.Event()
    release = threading.Event()
    bad_entered = threading.Event()
    # Exercise the actual global-state destroy/get path without initializing CUDA.
    monkeypatch.setattr(parallel_state, '_TP', SimpleNamespace(rank_in_group=0, destroy=lambda: None))
    for key in ('_SP', '_DP', '_NODE'):
        monkeypatch.setattr(parallel_state, key, None)

    def generate(**kwargs):
        entered.set()
        assert release.wait(5)
        assert parallel_state.get_tp_rank() == 0
        Path(kwargs['output_path'], 'result.mp4').write_bytes(b'video')

    def build(model, **kwargs):
        if model == 'bad':
            bad_entered.set()
            parallel_state.destroy_model_parallel()
            raise ValueError('checkpoint rejected')
        return SimpleNamespace(generate_video=generate, shutdown=lambda: None)

    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    a = runner.create_job('a', 'good', 'a')
    b = runner.create_job('b', 'bad', 'b')
    runner.start_job(a.id)
    assert entered.wait(5)
    try:
        runner.start_job(b.id)
        assert not bad_entered.wait(0.2), 'second task loaded inside the first task lifecycle'
        assert b.status.value == 'queued'
    finally:
        release.set()
    wait_done(a)
    wait_done(b)
    assert a.status == JobStatus.COMPLETED
    assert b.status == JobStatus.FAILED
    assert 'checkpoint rejected' in b.error


def test_switch_releases_previous_generator_before_loading(runner, monkeypatch):
    from fastvideo import VideoGenerator
    events = []
    def build(model, **kwargs):
        events.append(('load', model))
        return SimpleNamespace(shutdown=lambda: events.append(('stop', model)))
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    a = runner._get_or_create_generator('a', 't2v', 1)
    assert runner._get_or_create_generator('a', 't2v', 1) is a
    runner._get_or_create_generator('b', 't2v', 1)
    assert events == [('load', 'a'), ('stop', 'a'), ('load', 'b')]
    assert len(runner._generators) == 1


def test_queue_cancel_and_duplicate_start(runner, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def run(job):
        calls.append(job.id)
        job.status = JobStatus.RUNNING
        entered.set()
        assert release.wait(5)
        job.status = JobStatus.COMPLETED
    monkeypatch.setattr(runner, '_run_job', run)
    a = runner.create_job('a', 'm', 'a')
    b = runner.create_job('b', 'm', 'b')
    runner.start_job(a.id)
    assert entered.wait(5)
    try:
        runner.start_job(b.id)
        with pytest.raises(ValueError):
            runner.start_job(b.id)
        with pytest.raises(ValueError):
            runner.update_job_config(b.id, {'prompt': 'changed'})
        runner.stop_job(b.id)
        assert b.status == JobStatus.STOPPED
    finally:
        release.set()
    wait_done(a)
    runner._shutdown()
    assert calls == ['a']


def test_shutdown_releases_cached_generator(runner, monkeypatch):
    from fastvideo import VideoGenerator
    stopped = []
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', lambda *a, **k: SimpleNamespace(shutdown=lambda: stopped.append(1)))
    runner._get_or_create_generator('a', 't2v', 1)
    runner._shutdown()
    assert stopped == [1]


def test_single_gpu_studio_uses_process_executor(monkeypatch):
    from fastvideo_studio.inference_worker import StudioVideoGenerator
    from fastvideo.worker.multiproc_executor import MultiprocExecutor
    seen = {}
    monkeypatch.setattr(StudioVideoGenerator, '__init__', lambda self, **kw: seen.update(kw))
    StudioVideoGenerator.from_fastvideo_args(SimpleNamespace(num_gpus=1, distributed_executor_backend='mp'))
    assert seen['executor_class'] is MultiprocExecutor


def test_cancel_during_load_skips_generation_and_preserves_cache(runner, monkeypatch):
    from fastvideo import VideoGenerator
    entered, release = threading.Event(), threading.Event()
    generated = []
    def build(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(generate_video=lambda **kw: generated.append(kw), shutdown=lambda: None)
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    a = runner.create_job('a', 'good', 'a')
    runner.start_job(a.id)
    assert entered.wait(5)
    runner.stop_job(a.id)
    release.set()
    wait_done(a)
    assert a.status == JobStatus.STOPPED
    assert not generated
    assert a._log_buf.phase == 'stopped'


def test_generation_failure_evicts_cache_and_next_job_recovers(runner, monkeypatch):
    from fastvideo import VideoGenerator
    events = []
    def generate(**kw):
        if kw['prompt'] == 'bad':
            raise RuntimeError('injected generation failure')
        Path(kw['output_path'], 'result.mp4').write_bytes(b'video')
    def build(*args, **kwargs):
        events.append('load')
        return SimpleNamespace(generate_video=generate, shutdown=lambda: events.append('stop'))
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    bad = runner.create_job('bad', 'model', 'bad')
    good = runner.create_job('good', 'model', 'good')
    runner.start_job(bad.id)
    runner.start_job(good.id)
    wait_done(bad)
    wait_done(good)
    assert bad.status == JobStatus.FAILED
    assert good.status == JobStatus.COMPLETED
    assert events == ['load', 'stop', 'load']


def test_restart_marks_persisted_queued_job_failed(runner):
    job = runner.create_job('interrupted', 'm', 'prompt')
    runner._db.update_job(job.id, {'status': 'queued'})
    runner._jobs.clear()
    runner._load_jobs()
    loaded = runner.get_job(job.id)
    assert loaded.status == JobStatus.FAILED
    assert 'queued' in loaded.error
    assert runner._db.get_all_jobs()[0]['status'] == 'failed'


def test_shutdown_waits_for_inflight_load_before_closing_manager(runner, monkeypatch):
    from fastvideo import VideoGenerator
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    def build(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(shutdown=stopped.set)
    monkeypatch.setattr(VideoGenerator, 'from_pretrained', build)
    a = runner.create_job('a', 'm', 'a')
    runner.start_job(a.id)
    assert entered.wait(5)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (runner._shutdown(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(0.2)
    finally:
        release.set()
        closer.join(10)
    assert closed.is_set()
    assert stopped.is_set()
    assert not runner._worker.is_alive()
    assert not runner._generators
    assert a.status == JobStatus.STOPPED


def test_stop_silent_training_process(runner):
    import subprocess
    import sys
    job = runner.create_job('training', 'm', 'a', job_type='finetuning')
    job.status = JobStatus.RUNNING
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                               stdout=subprocess.PIPE, start_new_session=True)
    job._process = process
    try:
        runner.stop_job(job.id)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def test_delete_stops_silent_training_process(runner):
    import subprocess
    import sys
    job = runner.create_job('deleted', 'm', 'a', job_type='finetuning')
    job.status = JobStatus.RUNNING
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                               stdout=subprocess.PIPE, start_new_session=True)
    job._process = process
    try:
        assert runner.delete_job(job.id)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def test_training_stop_kills_children_after_leader_exits(runner, monkeypatch, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import psutil
    from fastvideo_studio import job_runner
    monkeypatch.setattr(job_runner, '_TRAINING_STOP_TIMEOUT_S', 0.2)
    pid_file = tmp_path / 'child.pid'
    child_code = "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); open(" + repr(str(pid_file)) + ", 'w').write(str(os.getpid())); time.sleep(60)"
    leader_code = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', " + repr(child_code) + "]); time.sleep(60)"
    process = subprocess.Popen([sys.executable, '-c', leader_code], stdout=subprocess.PIPE, start_new_session=True)
    job = runner.create_job('group', 'm', 'a', job_type='finetuning')
    job.status = JobStatus.RUNNING
    job._process = process
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child_pid = int(pid_file.read_text())
        runner.stop_job(job.id)
        assert process.poll() is not None
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert process.stdout.read() == b''
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def test_cancelled_queue_entry_cannot_execute_restarted_job_twice(runner, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def run(job):
        calls.append(job.id)
        job.status = JobStatus.RUNNING
        if job.id == 'a':
            entered.set()
            assert release.wait(5)
        job.status = JobStatus.COMPLETED
    monkeypatch.setattr(runner, '_run_job', run)
    a = runner.create_job('a', 'm', 'a')
    b = runner.create_job('b', 'm', 'b')
    runner.start_job(a.id)
    assert entered.wait(5)
    try:
        runner.start_job(b.id)
        runner.stop_job(b.id)
        runner.start_job(b.id)
    finally:
        release.set()
    wait_done(b)
    runner._shutdown()
    assert calls == ['a', 'b']
