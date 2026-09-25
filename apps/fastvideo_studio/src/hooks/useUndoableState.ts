'use client';

import * as React from 'react';

const MAX_HISTORY = 100;
// Edits made under the same key, this close together, merge into one undo step.
export const COALESCE_MS = 500;

interface Snapshot<T> {
  present: T;
  past: T[];
  future: T[];
}

/**
 * Undo/redo over a whole state value. Every `set` is one undo step, except
 * that consecutive sets with the same `key` inside COALESCE_MS merge -- pass a
 * key for typing so a burst is one step, and none for discrete changes (moves,
 * adds, selects) so each is its own. Values are treated as immutable, so
 * snapshots share structure and are cheap to keep.
 */
export function useUndoableState<T>(initial: T) {
  const [snapshot, setSnapshot] = React.useState<Snapshot<T>>({ present: initial, past: [], future: [] });
  // Mirrors the snapshot synchronously so several sets in one event chain correctly.
  const ref = React.useRef(snapshot);
  const last = React.useRef<{ key: string | null; at: number }>({ key: null, at: 0 });

  const apply = React.useCallback((next: Snapshot<T>) => {
    ref.current = next;
    setSnapshot(next);
  }, []);

  const set = React.useCallback(
    (update: T | ((prev: T) => T), key: string | null = null) => {
      const cur = ref.current;
      const next = typeof update === 'function' ? (update as (prev: T) => T)(cur.present) : update;
      if (Object.is(next, cur.present)) return;

      const now = Date.now();
      const merges = key !== null && last.current.key === key && now - last.current.at < COALESCE_MS;
      last.current = { key, at: now };

      if (merges) {
        apply({ ...cur, present: next });
      } else {
        apply({ present: next, past: [...cur.past.slice(-(MAX_HISTORY - 1)), cur.present], future: [] });
      }
    },
    [apply],
  );

  const undo = React.useCallback(() => {
    const cur = ref.current;
    if (cur.past.length === 0) return false;
    last.current = { key: null, at: 0 };
    apply({ present: cur.past[cur.past.length - 1], past: cur.past.slice(0, -1), future: [cur.present, ...cur.future] });
    return true;
  }, [apply]);

  const redo = React.useCallback(() => {
    const cur = ref.current;
    if (cur.future.length === 0) return false;
    last.current = { key: null, at: 0 };
    apply({ present: cur.future[0], past: [...cur.past, cur.present], future: cur.future.slice(1) });
    return true;
  }, [apply]);

  /** Replace the value and forget all history (e.g. when an editor reopens). */
  const reset = React.useCallback(
    (value: T) => {
      last.current = { key: null, at: 0 };
      apply({ present: value, past: [], future: [] });
    },
    [apply],
  );

  return {
    state: snapshot.present,
    set,
    undo,
    redo,
    reset,
    canUndo: snapshot.past.length > 0,
    canRedo: snapshot.future.length > 0,
  };
}
