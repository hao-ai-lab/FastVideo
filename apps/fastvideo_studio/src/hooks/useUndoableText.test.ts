import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useUndoableText } from './useUndoableText';

describe('useUndoableText', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('coalesces a typing burst into one undo step', () => {
    const { result } = renderHook(() => useUndoableText('a'));
    act(() => {
      result.current.type('ab');
      result.current.type('abc');
    });
    act(() => {
      vi.advanceTimersByTime(600);
    });
    expect(result.current.canUndo).toBe(true);

    let restored: string | null = null;
    act(() => {
      restored = result.current.undo();
    });
    expect(restored).toBe('a');
    expect(result.current.text).toBe('a');
    expect(result.current.canUndo).toBe(false);
  });

  it('gives a committed edit its own step and supports redo', () => {
    const { result } = renderHook(() => useUndoableText('hi'));
    act(() => result.current.commit('hi <pause>'));
    expect(result.current.text).toBe('hi <pause>');

    act(() => {
      result.current.undo();
    });
    expect(result.current.text).toBe('hi');
    expect(result.current.canRedo).toBe(true);

    let redone: string | null = null;
    act(() => {
      redone = result.current.redo();
    });
    expect(redone).toBe('hi <pause>');
    expect(result.current.text).toBe('hi <pause>');
  });

  it('flushes pending typing before a commit so undo steps stay separate', () => {
    const { result } = renderHook(() => useUndoableText(''));
    act(() => result.current.type('hello'));
    act(() => result.current.commit('hello <pause>'));

    act(() => {
      result.current.undo();
    });
    expect(result.current.text).toBe('hello');
    act(() => {
      result.current.undo();
    });
    expect(result.current.text).toBe('');
  });

  it('returns null when there is nothing to undo or redo', () => {
    const { result } = renderHook(() => useUndoableText('x'));
    let undone: string | null = 'sentinel';
    let redone: string | null = 'sentinel';
    act(() => {
      undone = result.current.undo();
      redone = result.current.redo();
    });
    expect(undone).toBeNull();
    expect(redone).toBeNull();
  });

  it('reset replaces the text and drops history', () => {
    const { result } = renderHook(() => useUndoableText('a'));
    act(() => result.current.commit('b'));
    act(() => result.current.reset('fresh'));
    expect(result.current.text).toBe('fresh');
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });
});
