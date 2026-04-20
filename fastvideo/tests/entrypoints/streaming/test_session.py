# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle tests."""
from __future__ import annotations

import time

import pytest

from fastvideo.entrypoints.streaming.session import (
    InvalidSessionTransition,
    Session,
    SessionManager,
    SessionRejected,
    SessionState,
)


class TestSessionStateMachine:

    def test_starts_initializing(self):
        s = Session()
        assert s.state is SessionState.INITIALIZING

    def test_legal_sequence(self):
        s = Session()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.GPU_BINDING)
        s.transition(SessionState.ACTIVE)
        s.transition(SessionState.COMPLETE)
        assert s.state is SessionState.COMPLETE

    def test_active_self_loop_allowed(self):
        s = Session()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.GPU_BINDING)
        s.transition(SessionState.ACTIVE)
        s.transition(SessionState.ACTIVE)  # re-asserting is fine
        assert s.state is SessionState.ACTIVE

    def test_illegal_backwards_transition(self):
        s = Session()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.GPU_BINDING)
        s.transition(SessionState.ACTIVE)
        with pytest.raises(InvalidSessionTransition):
            s.transition(SessionState.INITIALIZING)

    def test_cannot_leave_terminal_state(self):
        s = Session()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.GPU_BINDING)
        s.transition(SessionState.ACTIVE)
        s.transition(SessionState.COMPLETE)
        with pytest.raises(InvalidSessionTransition):
            s.transition(SessionState.ACTIVE)

    def test_error_terminal(self):
        s = Session()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.ERROR)
        with pytest.raises(InvalidSessionTransition):
            s.transition(SessionState.ACTIVE)

    def test_transition_updates_activity(self):
        s = Session()
        prior = s.last_activity
        time.sleep(0.001)
        s.transition(SessionState.QUEUED)
        assert s.last_activity > prior

    def test_segment_cap(self):
        s = Session()
        s.generated_segment_count = 5
        assert s.segment_cap_reached(5) is True
        assert s.segment_cap_reached(6) is False


class TestSessionManager:

    def test_create_assigns_unique_ids(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=60, max_sessions=2)
        a = mgr.create()
        b = mgr.create()
        assert a.id != b.id
        assert len(mgr) == 2

    def test_max_sessions_enforced(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=60, max_sessions=1)
        mgr.create()
        with pytest.raises(SessionRejected):
            mgr.create()

    def test_close_releases_slot(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=60, max_sessions=1)
        s = mgr.create()
        mgr.close(s.id)
        assert len(mgr) == 0
        # Now can create again.
        mgr.create()

    def test_reap_timed_out_flags_idle_sessions(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=1, max_sessions=4)
        s = mgr.create()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.GPU_BINDING)
        s.transition(SessionState.ACTIVE)
        s.last_activity = time.monotonic() - 10  # 10s ago, past the 1s budget
        dead = mgr.reap_timed_out()
        assert s.id in dead

    def test_reap_skips_terminal_states(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=1, max_sessions=4)
        s = mgr.create()
        s.transition(SessionState.QUEUED)
        s.transition(SessionState.ERROR)
        s.last_activity = time.monotonic() - 10
        assert s.id not in mgr.reap_timed_out()

    def test_active_sessions_filter(self):
        mgr = SessionManager(
            segment_cap=4, session_timeout_seconds=60, max_sessions=4)
        a = mgr.create()
        a.transition(SessionState.QUEUED)
        a.transition(SessionState.GPU_BINDING)
        a.transition(SessionState.ACTIVE)
        b = mgr.create()  # INITIALIZING
        assert mgr.active_sessions() == [a]
        assert b not in mgr.active_sessions()
