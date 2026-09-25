import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { COALESCE_MS, useUndoableState } from './useUndoableState';

describe('useUndoableState', () => {
  let now = 1_000_000;
  beforeEach(() => {
    now = 1_000_000;
    vi.spyOn(Date, 'now').mockImplementation(() => now);
  });
  afterEach(() => vi.restoreAllMocks());

  const tick = (ms: number) => {
    now += ms;
  };

  it('makes each unkeyed change its own step', () => {
    const { result } = renderHook(() => useUndoableState({ n: 0 }));
    act(() => result.current.set({ n: 1 }));
    act(() => result.current.set({ n: 2 }));
    expect(result.current.state.n).toBe(2);

    act(() => void result.current.undo());
    expect(result.current.state.n).toBe(1);
    act(() => void result.current.undo());
    expect(result.current.state.n).toBe(0);
    expect(result.current.canUndo).toBe(false);
  });

  it('merges same-key edits made close together into one step', () => {
    const { result } = renderHook(() => useUndoableState('a'));
    act(() => result.current.set('ab', 'text'));
    tick(100);
    act(() => result.current.set('abc', 'text'));
    tick(100);
    act(() => result.current.set('abcd', 'text'));

    act(() => void result.current.undo());
    expect(result.current.state).toBe('a');
    expect(result.current.canUndo).toBe(false);
  });

  it('starts a new step after a pause, or when the key changes', () => {
    const { result } = renderHook(() => useUndoableState('a'));
    act(() => result.current.set('ab', 'text'));
    tick(COALESCE_MS + 1);
    act(() => result.current.set('abc', 'text'));
    act(() => result.current.set('abcX', 'other'));

    act(() => void result.current.undo());
    expect(result.current.state).toBe('abc');
    act(() => void result.current.undo());
    expect(result.current.state).toBe('ab');
    act(() => void result.current.undo());
    expect(result.current.state).toBe('a');
  });

  it('never merges an unkeyed change into typing, or typing into it', () => {
    const { result } = renderHook(() => useUndoableState('a'));
    act(() => result.current.set('ab', 'text'));
    act(() => result.current.set('AB'));
    act(() => result.current.set('ABc', 'text'));

    act(() => void result.current.undo());
    expect(result.current.state).toBe('AB');
    act(() => void result.current.undo());
    expect(result.current.state).toBe('ab');
  });

  it('typing after an undo begins a fresh step instead of merging into the undone one', () => {
    const { result } = renderHook(() => useUndoableState('a'));
    act(() => result.current.set('ab', 'text'));
    act(() => void result.current.undo());
    act(() => result.current.set('ax', 'text'));
    expect(result.current.canUndo).toBe(true);
    act(() => void result.current.undo());
    expect(result.current.state).toBe('a');
  });

  it('redoes, and a new change clears what could be redone', () => {
    const { result } = renderHook(() => useUndoableState(0));
    act(() => result.current.set(1));
    act(() => result.current.set(2));
    act(() => void result.current.undo());
    act(() => void result.current.undo());
    expect(result.current.canRedo).toBe(true);

    act(() => void result.current.redo());
    expect(result.current.state).toBe(1);
    act(() => result.current.set(9));
    expect(result.current.canRedo).toBe(false);
    expect(result.current.redo()).toBe(false);
  });

  it('returns false when there is nothing to undo or redo', () => {
    const { result } = renderHook(() => useUndoableState(0));
    expect(result.current.undo()).toBe(false);
    expect(result.current.redo()).toBe(false);
  });

  it('chains functional updates made in the same event', () => {
    const { result } = renderHook(() => useUndoableState(0));
    act(() => {
      result.current.set((n) => n + 1);
      result.current.set((n) => n + 1);
    });
    expect(result.current.state).toBe(2);
    act(() => void result.current.undo());
    expect(result.current.state).toBe(1);
  });

  it('ignores a set that produces the same value', () => {
    const { result } = renderHook(() => useUndoableState('x'));
    act(() => result.current.set('x'));
    expect(result.current.canUndo).toBe(false);
  });

  it('reset replaces the value and drops all history', () => {
    const { result } = renderHook(() => useUndoableState(0));
    act(() => result.current.set(1));
    act(() => void result.current.undo());
    act(() => result.current.reset(7));
    expect(result.current.state).toBe(7);
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });

  it('keeps at most 100 steps', () => {
    const { result } = renderHook(() => useUndoableState(0));
    for (let i = 1; i <= 130; i += 1) act(() => result.current.set(i));
    let undone = 0;
    while (result.current.canUndo) {
      act(() => void result.current.undo());
      undone += 1;
    }
    expect(undone).toBe(100);
    expect(result.current.state).toBe(30);
  });
});
