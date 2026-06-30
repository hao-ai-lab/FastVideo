import { describe, it, expect, vi } from 'vitest';

import { createManagedStore } from './createManagedStore';

describe('createManagedStore', () => {
  it('get returns the initial state', () => {
    const store = createManagedStore({ a: 1 });
    expect(store.get()).toEqual({ a: 1 });
  });

  it('set replaces state and notifies subscribers', () => {
    const store = createManagedStore({ a: 1 });
    const cb = vi.fn();
    store.subscribe(cb);
    store.set({ a: 2 });
    expect(store.get()).toEqual({ a: 2 });
    expect(cb).toHaveBeenCalledWith({ a: 2 });
  });

  it('update derives the next state from the current state', () => {
    const store = createManagedStore({ count: 1 });
    store.update((s) => ({ count: s.count + 1 }));
    expect(store.get().count).toBe(2);
  });

  it('patch merges a partial (object form and function form)', () => {
    const store = createManagedStore({ a: 1, b: 2 });
    store.patch({ b: 20 });
    expect(store.get()).toEqual({ a: 1, b: 20 });
    store.patch((s) => ({ a: s.a + 100 }));
    expect(store.get()).toEqual({ a: 101, b: 20 });
  });

  it('unsubscribe stops further notifications', () => {
    const store = createManagedStore({ a: 1 });
    const cb = vi.fn();
    const unsubscribe = store.subscribe(cb);
    unsubscribe();
    store.set({ a: 2 });
    expect(cb).not.toHaveBeenCalled();
  });

  it('applies deriveState on init and on every set', () => {
    const store = createManagedStore({ n: 3 }, (s) => ({ n: Math.min(s.n, 5) }));
    expect(store.get().n).toBe(3);
    store.set({ n: 10 });
    expect(store.get().n).toBe(5);
  });
});
