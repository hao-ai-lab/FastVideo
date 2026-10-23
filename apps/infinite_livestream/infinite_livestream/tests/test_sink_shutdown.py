"""Use real OS pipes to cover shutdown under encoder backpressure, without GPUs."""

import asyncio
import os
import select
import signal
import subprocess
import sys
import threading

import pytest

from infinite_livestream import sink as sink_module
from infinite_livestream.sink import HlsSink, _PipeWriter


class ObservedPipe:
    def __init__(self, pipe):
        self.pipe = pipe
        self.writing = threading.Event()

    def write(self, payload):
        self.writing.set()
        return self.pipe.write(payload)


@pytest.mark.skipif(os.name != "posix", reason="the HLS sink uses POSIX pipes")
@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_stop_unblocks_full_pipes_and_reaps_process(tmp_path, monkeypatch, ignore_terminate):
    monkeypatch.setattr(sink_module.shutil, "which", lambda name: "/test/ffmpeg")
    monkeypatch.setattr(sink_module, "_PROCESS_EXIT_TIMEOUT_S", 0.2, raising=False)
    sink = HlsSink(tmp_path)
    audio_read, audio_write = os.pipe()
    child = (
        "import signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_terminate else "")
        + "print('ready', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", child], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, bufsize=0, pass_fds=(audio_read,))
    os.close(audio_read)
    sink._process = process
    audio_pipe = os.fdopen(audio_write, "wb", buffering=0)
    sink._audio_pipe = audio_pipe
    writers = [_PipeWriter("video-test", 1), _PipeWriter("audio-test", 1)]
    sink._video_writer, sink._audio_writer = writers
    errors = []
    ticks = []
    stopper = None
    try:
        assert select.select([process.stdout], [], [], 5)[0], "child failed to become ready"
        assert process.stdout.readline() == b"ready\n"
        for writer, pipe in zip(writers, (process.stdin, sink._audio_pipe)):
            observed = ObservedPipe(pipe)
            writer.attach(observed)
            writer.start()
            writer.submit(b"x" * (2 * 1024 * 1024))
            assert observed.writing.wait(2), "writer never reached the pipe"
            writer.submit(b"queued")
            assert writer.queue.full()

        async def stop_with_heartbeat():
            task = asyncio.create_task(sink.stop())
            while not task.done():
                ticks.append(True)
                await asyncio.sleep(0.01)
            await task

        def stop():
            try:
                asyncio.run(stop_with_heartbeat())
            except BaseException as error:
                errors.append(error)

        # A separate thread makes the timeout effective even if a regression
        # blocks the event loop inside a synchronous queue/pipe operation.
        stopper = threading.Thread(target=stop, daemon=True)
        stopper.start()
        stopper.join(timeout=5)
        assert not stopper.is_alive(), "shutdown blocked on a full pipe or queue"
        assert errors == []
        expected_signal = signal.SIGKILL if ignore_terminate else signal.SIGTERM
        assert process.returncode == -expected_signal
        assert all(not writer.is_alive() for writer in writers)
        assert process.stdin.closed and audio_pipe.closed
        assert sink._audio_pipe is None
        if ignore_terminate:
            assert len(ticks) > 1, "waiting for FFmpeg blocked the event loop"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if stopper is not None:
            stopper.join(timeout=5)
        for writer in writers:
            writer.close()
            if writer.ident is not None:
                writer.join(timeout=2)
        process.stdin.close()
        process.stdout.close()
        if sink._audio_pipe is not None:
            sink._audio_pipe.close()


def test_writer_preserves_the_payload_across_short_writes():
    payload = b"one complete media frame"
    written = bytearray()
    complete = threading.Event()

    class ShortPipe:
        def write(self, data):
            count = min(3, len(data))
            written.extend(data[:count])
            if len(written) == len(payload):
                complete.set()
            return count

    writer = _PipeWriter("short-write-test", 1)
    writer.attach(ShortPipe())
    writer.start()
    try:
        writer.submit(payload)
        assert complete.wait(2), "a short write discarded the rest of the media payload"
        assert written == payload
    finally:
        writer.close()
        writer.join(timeout=2)
    assert not writer.is_alive()
