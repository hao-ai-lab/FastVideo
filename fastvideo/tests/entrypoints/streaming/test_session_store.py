# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming SessionStore and BlobStore.

Covers:

* ``store`` / ``snapshot`` / ``drop`` lifecycle for the in-memory store
* ``hydrate`` with and without an explicit session id
* blob store insert / get / drop semantics
* thread-safety under concurrent writes (smoke)
* round-trip a LTX-2 continuation through snapshot + hydrate across a
  session boundary (the "export and resume" flow the PR plan calls out)
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from fastvideo.api.schema import ContinuationState
from fastvideo.entrypoints.streaming.session_store import (
    BlobStore,
    InMemoryBlobStore,
    InMemorySessionStore,
    SessionStore,
)
from fastvideo.pipelines.basic.ltx2.continuation import (
    LTX2_CONTINUATION_KIND,
    LTX2ContinuationState,
)


class TestInMemoryBlobStore:

    def test_is_blob_store(self):
        assert isinstance(InMemoryBlobStore(), BlobStore)

    def test_put_then_get_returns_same_bytes(self):
        store = InMemoryBlobStore()
        blob_id = store.put(b"hello")
        assert store.get(blob_id) == b"hello"

    def test_put_returns_distinct_ids(self):
        store = InMemoryBlobStore()
        id_a = store.put(b"a")
        id_b = store.put(b"b")
        assert id_a != id_b

    def test_get_missing_raises_keyerror(self):
        store = InMemoryBlobStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")

    def test_drop_removes_blob(self):
        store = InMemoryBlobStore()
        blob_id = store.put(b"payload")
        store.drop(blob_id)
        assert blob_id not in store
        with pytest.raises(KeyError):
            store.get(blob_id)

    def test_drop_missing_is_noop(self):
        store = InMemoryBlobStore()
        store.drop("not-there")  # no raise

    def test_contains(self):
        store = InMemoryBlobStore()
        blob_id = store.put(b"x")
        assert blob_id in store
        assert "other" not in store


class TestInMemorySessionStore:

    def test_is_session_store(self):
        assert isinstance(InMemorySessionStore(), SessionStore)

    def test_store_then_snapshot(self):
        store = InMemorySessionStore()
        state = ContinuationState(kind="ltx2.v1", payload={"x": 1})
        store.store("sess-1", state)
        assert store.snapshot("sess-1") is state

    def test_snapshot_missing_returns_none(self):
        store = InMemorySessionStore()
        assert store.snapshot("missing") is None

    def test_store_overwrites_prior_state(self):
        store = InMemorySessionStore()
        first = ContinuationState(kind="ltx2.v1", payload={"v": 1})
        second = ContinuationState(kind="ltx2.v1", payload={"v": 2})
        store.store("s", first)
        store.store("s", second)
        assert store.snapshot("s").payload["v"] == 2

    def test_hydrate_assigns_new_session_id(self):
        store = InMemorySessionStore()
        state = ContinuationState(kind="ltx2.v1", payload={})
        sid = store.hydrate(state)
        assert sid
        assert store.snapshot(sid) is state

    def test_hydrate_with_explicit_session_id(self):
        store = InMemorySessionStore()
        state = ContinuationState(kind="ltx2.v1", payload={})
        sid = store.hydrate(state, session_id="pinned-id")
        assert sid == "pinned-id"
        assert store.snapshot("pinned-id") is state

    def test_drop_forgets_session(self):
        store = InMemorySessionStore()
        store.store("s", ContinuationState(kind="ltx2.v1", payload={}))
        store.drop("s")
        assert store.snapshot("s") is None
        assert "s" not in store

    def test_iter_yields_session_ids(self):
        store = InMemorySessionStore()
        store.store("a", ContinuationState(kind="ltx2.v1", payload={}))
        store.store("b", ContinuationState(kind="ltx2.v1", payload={}))
        assert sorted(store) == ["a", "b"]

    def test_len(self):
        store = InMemorySessionStore()
        assert len(store) == 0
        store.store("x", ContinuationState(kind="ltx2.v1", payload={}))
        assert len(store) == 1

    def test_concurrent_store_is_safe(self):
        """Smoke-check the lock: 200 parallel stores settle to 200 ids."""
        store = InMemorySessionStore()

        def write(i: int) -> None:
            store.store(
                f"s-{i}",
                ContinuationState(kind="ltx2.v1", payload={"i": i}),
            )

        threads = [threading.Thread(target=write, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(store) == 200


class TestSnapshotHydrateRoundTrip:
    """Session boundary: snapshot + hydrate preserves the full LTX-2 state."""

    def test_end_to_end_ltx2_session_migration(self):
        blob_store = InMemoryBlobStore()
        sessions = InMemorySessionStore()

        typed = LTX2ContinuationState(
            segment_index=4,
            video_frames=[
                np.full((32, 32, 3), i * 5, dtype=np.uint8) for i in range(3)
            ],
            audio_latents=torch.randn(1, 4, 8, 32, dtype=torch.float32),
            audio_sample_rate=24000,
            audio_conditioning_num_frames=5,
            video_position_offset_sec=0.25,
        )
        envelope = typed.to_continuation_state(blob_store=blob_store)

        sessions.store("session-a", envelope)
        snapshot = sessions.snapshot("session-a")
        assert snapshot is not None
        assert snapshot.kind == LTX2_CONTINUATION_KIND

        # Simulate a migration: drop the first session, hydrate a new one
        # from the snapshot, and reconstruct the typed state.
        sessions.drop("session-a")
        new_sid = sessions.hydrate(snapshot)
        assert new_sid != "session-a"
        rebuilt = sessions.snapshot(new_sid)
        assert rebuilt is snapshot

        restored = LTX2ContinuationState.from_continuation_state(
            rebuilt, blob_store=blob_store)
        assert restored.segment_index == typed.segment_index
        assert restored.audio_sample_rate == typed.audio_sample_rate
        torch.testing.assert_close(
            restored.audio_latents, typed.audio_latents)
        assert len(restored.video_frames) == len(typed.video_frames)
