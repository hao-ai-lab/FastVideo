"""Frame ownership must survive buffering, repetition, short writes, and drops."""

import asyncio
import queue
import threading

import numpy as np
import pytest

from infinite_livestream.metadata import encode_id3
from infinite_livestream.pacer import Pacer
from infinite_livestream.sink import AudioFormat, VideoFormat, _PipeWriter


def test_pacer_repeats_and_drops_identity_with_its_frame(monkeypatch):
    from infinite_livestream import pacer as module
    monkeypatch.setattr(module, "_BUFFER_SECONDS", 2 / 24)
    displayed = []

    class Sink:
        async def start(self, *args):
            pass

        def send_video(self, frame, metadata):
            displayed.append((int(frame[0, 0, 0]), metadata))
            if len(displayed) == 4:
                raise asyncio.CancelledError

        def send_audio(self, samples):
            pass

    pacer = Pacer(Sink(), VideoFormat(2, 2, 24), AudioFormat(48000, 1))
    for value in (1, 2, 3):
        pacer.submit_video(np.full((2, 2, 3), value, dtype=np.uint8), str(value).encode())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pacer.run())
    assert displayed == [(2, b"2"), (3, b"3"), (3, b"3"), (3, b"3")]
    assert pacer.dropped_frames == 1


def test_writer_commits_only_the_surviving_complete_frame():
    ledger = queue.Queue()
    written = bytearray()

    class ShortPipe:
        def write(self, data):
            # No record may be visible while any bytes remain unwritten.
            assert ledger.empty()
            written.extend(data[:2])
            return min(2, len(data))

    writer = _PipeWriter("metadata-test", 1)
    writer.attach(ShortPipe(), ledger)
    writer.submit(b"discarded", b"old-title")
    assert writer.submit(b"kept frame", b"new-title") == 1
    writer.start()
    try:
        assert ledger.get(timeout=2) == b"new-title"
        assert written == b"kept frame"
        assert ledger.empty()
    finally:
        writer.close()
        writer.join(timeout=2)


def test_partial_frame_failure_never_commits_a_title():
    ledger = queue.Queue()

    class BrokenPipe:
        first = True

        def write(self, data):
            if self.first:
                self.first = False
                return 1
            raise BrokenPipeError

    writer = _PipeWriter("broken-metadata-test", 1)
    writer.attach(BrokenPipe(), ledger)
    writer.start()
    try:
        writer.submit(b"incomplete", b"must not appear")
        assert writer.broken.wait(2)
        assert ledger.empty()
    finally:
        writer.close()
        writer.join(timeout=2)


def test_replaced_encoder_cannot_receive_the_previous_writes_title():
    entered, release = threading.Event(), threading.Event()
    old_ledger, new_ledger = queue.Queue(), queue.Queue()

    class OldPipe:
        def write(self, data):
            entered.set()
            assert release.wait(2)
            return len(data)

    class NewPipe:
        def write(self, data):
            return len(data)

    writer = _PipeWriter("restart-metadata-test", 2)
    writer.attach(OldPipe(), old_ledger)
    writer.start()
    try:
        writer.submit(b"old frame", b"old title")
        assert entered.wait(2)
        writer.attach(NewPipe(), new_ledger)
        writer.submit(b"new frame", b"new title")
        release.set()
        assert old_ledger.get(timeout=2) == b"old title"
        assert new_ledger.get(timeout=2) == b"new title"
        assert old_ledger.empty() and new_ledger.empty()
    finally:
        release.set()
        writer.close()
        writer.join(timeout=2)


def test_title_encoding_preserves_unicode_and_explicit_black_state():
    # UTF-8 text must survive as data, including characters meaningful in HTML.
    value = {"clip_id": "a", "title": "猫 <script> & café"}
    assert "猫 <script> & café".encode() in encode_id3(value)
    assert b'"clip":null' in encode_id3(None)
