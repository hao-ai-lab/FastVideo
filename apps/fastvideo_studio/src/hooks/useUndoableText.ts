'use client';

import * as React from 'react';

const MAX_HISTORY = 100;
// How long a pause in typing has to be before it becomes its own undo step.
// Keeps undo from removing text one keystroke at a time while still giving
// each tag/preset insertion (the actual ask) a guaranteed, immediate step.
const TYPING_COALESCE_MS = 500;

/**
 * Text state with undo/redo. Typing is coalesced into bursts; discrete edits
 * (`commit`, e.g. a tag insertion) always get their own step. `textRef`
 * mirrors `text` synchronously because history bookkeeping needs the true
 * current value the instant an edit happens, not whatever the last render
 * closed over.
 */
export function useUndoableText(initial: string) {
  const [text, setTextState] = React.useState(initial);
  const textRef = React.useRef(initial);
  const [past, setPast] = React.useState<string[]>([]);
  const [future, setFuture] = React.useState<string[]>([]);
  // Start of the in-progress typing burst that hasn't become a history step
  // yet; flushed into `past` once typing pauses or a discrete edit happens.
  const typingBaseRef = React.useRef<string | null>(null);
  const typingTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const setText = React.useCallback((next: string) => {
    textRef.current = next;
    setTextState(next);
  }, []);

  const pushHistory = React.useCallback((previous: string) => {
    setPast((p) => (p.length >= MAX_HISTORY ? [...p.slice(1), previous] : [...p, previous]));
    setFuture([]);
  }, []);

  const clearTypingTimer = React.useCallback(() => {
    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current);
      typingTimerRef.current = null;
    }
  }, []);

  const flushTypingCheckpoint = React.useCallback(() => {
    clearTypingTimer();
    if (typingBaseRef.current != null && typingBaseRef.current !== textRef.current) {
      pushHistory(typingBaseRef.current);
    }
    typingBaseRef.current = null;
  }, [clearTypingTimer, pushHistory]);

  /** Free typing: coalesced into one undo step per burst. */
  const type = React.useCallback(
    (next: string) => {
      if (typingBaseRef.current == null) typingBaseRef.current = textRef.current;
      setText(next);
      clearTypingTimer();
      typingTimerRef.current = setTimeout(flushTypingCheckpoint, TYPING_COALESCE_MS);
    },
    [setText, clearTypingTimer, flushTypingCheckpoint],
  );

  /** Discrete edit (tag/preset insertion): always its own undo step. */
  const commit = React.useCallback(
    (next: string) => {
      flushTypingCheckpoint();
      pushHistory(textRef.current);
      setText(next);
    },
    [flushTypingCheckpoint, pushHistory, setText],
  );

  /** Returns the restored text (so the caller can place the caret), or null if nothing to undo. */
  function undo(): string | null {
    flushTypingCheckpoint();
    if (past.length === 0) return null;
    const prev = past[past.length - 1];
    setPast(past.slice(0, -1));
    setFuture([...future, textRef.current]);
    setText(prev);
    return prev;
  }

  function redo(): string | null {
    if (future.length === 0) return null;
    const next = future[future.length - 1];
    setFuture(future.slice(0, -1));
    setPast([...past, textRef.current]);
    setText(next);
    return next;
  }

  /** Replace the text and drop all history (e.g. when an editor is reopened). */
  const reset = React.useCallback(
    (value: string) => {
      setText(value);
      setPast([]);
      setFuture([]);
      typingBaseRef.current = null;
      clearTypingTimer();
    },
    [setText, clearTypingTimer],
  );

  React.useEffect(() => clearTypingTimer, [clearTypingTimer]);

  return { text, type, commit, undo, redo, reset, canUndo: past.length > 0, canRedo: future.length > 0 };
}
