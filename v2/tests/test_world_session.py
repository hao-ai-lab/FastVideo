"""Interactive world-model session — the realtime/interactive plane.

The Session plane had zero coverage. These tests drive the causal ``chunk_rollout`` loop as a
long-lived session and assert the seams a one-shot ``generate()`` can't reach:

  * **persistent cross-request world state** — each ``act`` continues from the last (the carried KV);
  * **history-dependence** — the same action with a different history yields a different world;
  * **transactional step-boundary cancellation** — a cancelled act leaves the session resumable;
  * **frame streaming** — chunks stream as they complete;
  * **no cross-session smearing** — interleaving two sessions is bit-identical to running them apart
    (the interleave guarantee, now for sessions).
"""
from __future__ import annotations

import numpy as np
import pytest

from v2.recipes import build_default_engine
from v2.core.request.cancel import Cancelled
from v2.runtime import WorldModelSession

MID = "wan-causal-sf-1.3b"


def _frames_video(result):
    return np.asarray(result.outputs["latents"])


def test_session_persists_world_state_across_acts():
    eng = build_default_engine()
    s = WorldModelSession(eng, MID, window=2)
    s.act("walk forward", seed=1)
    assert s.step == 1 and len(s.world_context) == 2          # window-capped persistent chunks
    assert s.session.kv_handle is s.world_context             # the persistent state lives on the Session
    s.act("turn left", seed=1)
    assert s.step == 2 and len(s.world_context) == 2          # still carrying world state
    assert len(s.frames()) == 14                              # 7 chunks/act × 2 acts streamed


def test_continuation_depends_on_history():
    """The world is stateful: identical action+seed but a different prior history → different frames;
    identical history → bit-identical (deterministic)."""
    eng = build_default_engine()
    a = WorldModelSession(eng, MID)
    a.act("go north", seed=5)
    va = a.act("go north", seed=7)
    b = WorldModelSession(eng, MID)
    b.act("go south", seed=5)                                 # different history
    vb = b.act("go north", seed=7)                            # same final action+seed
    assert not np.array_equal(_frames_video(va), _frames_video(vb))
    c = WorldModelSession(eng, MID)
    c.act("go north", seed=5)
    vc = c.act("go north", seed=7)                            # same history as `a`
    assert np.array_equal(_frames_video(va), _frames_video(vc))


def test_cancel_mid_rollout_is_transactional():
    eng = build_default_engine()
    s = WorldModelSession(eng, MID)
    s.act("start", seed=1)
    ctx_before, step_before = [c.copy() for c in s.world_context], s.step

    def cancel_at_step_2(sess, n):
        if n == 2:
            sess.cancel()

    with pytest.raises(Cancelled):
        s.act("continue", seed=2, step_hook=cancel_at_step_2)
    # the persistent world state is unchanged → the session is resumable from the last good frame
    assert s.step == step_before
    assert len(s.world_context) == len(ctx_before)
    assert all(np.array_equal(x, y) for x, y in zip(s.world_context, ctx_before))


def test_two_sessions_do_not_smear():
    """Interleaving a second session's acts between this session's acts must not change this session's
    output — per-session state isolation (the interleave guarantee for the session plane)."""
    eng = build_default_engine()
    a = WorldModelSession(eng, MID)
    a.act("x", seed=1)
    b = WorldModelSession(eng, MID)                           # a second, interleaved session
    b.act("y", seed=2)
    b.act("z", seed=3)
    interleaved = _frames_video(a.act("w", seed=4))

    solo = WorldModelSession(eng, MID)
    solo.act("x", seed=1)
    alone = _frames_video(solo.act("w", seed=4))
    assert np.array_equal(interleaved, alone)


def test_window_bounds_persistent_state():
    eng = build_default_engine()
    s = WorldModelSession(eng, MID, window=1)
    for i in range(3):
        s.act(f"act {i}", seed=i)
    assert len(s.world_context) == 1                          # window caps the carried world state
