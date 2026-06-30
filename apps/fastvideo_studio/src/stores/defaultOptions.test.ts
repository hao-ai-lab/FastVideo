import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock only the backend client used by the store. The shared pure-TS helpers
// in '@/lib/defaultOptions' (DEFAULT_OPTIONS, load/saveDefaultOptions) stay real.
vi.mock('@/lib/api', () => ({
  getSettings: vi.fn(),
  updateSettings: vi.fn(() => Promise.resolve({})),
}));

import { getSettings, updateSettings } from '@/lib/api';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';
import {
  defaultOptionsStore,
  initDefaultOptions,
  updateOption,
  resetToDefaults,
} from './defaultOptions';

const STORAGE_KEY = 'fastvideo-default-options';
const flushMicrotasks = () => new Promise((r) => setTimeout(r, 0));

// Node's experimental global localStorage is unreliable under vitest
// (clear/removeItem throw), so install a clean in-memory Storage per test.
function makeLocalStorage(): Storage {
  const data: Record<string, string> = {};
  return {
    getItem: (k) =>
      Object.prototype.hasOwnProperty.call(data, k) ? data[k] : null,
    setItem: (k, v) => {
      data[k] = String(v);
    },
    removeItem: (k) => {
      delete data[k];
    },
    clear: () => {
      Object.keys(data).forEach((k) => delete data[k]);
    },
    key: (i) => Object.keys(data)[i] ?? null,
    get length() {
      return Object.keys(data).length;
    },
  } as Storage;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('localStorage', makeLocalStorage());
  defaultOptionsStore.set({ options: { ...DEFAULT_OPTIONS } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('defaultOptions store', () => {
  it('initDefaultOptions merges server settings but keeps apiServerBaseUrl local', async () => {
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, apiServerBaseUrl: 'http://local:9999/api' },
    });
    vi.mocked(getSettings).mockResolvedValue({
      ...DEFAULT_OPTIONS,
      numFrames: 121,
      apiServerBaseUrl: 'http://server-should-be-ignored/api',
    } as never);

    initDefaultOptions();
    await flushMicrotasks();

    const opts = defaultOptionsStore.get().options;
    expect(opts.numFrames).toBe(121); // server value applied
    expect(opts.apiServerBaseUrl).toBe('http://local:9999/api'); // local preserved
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).numFrames).toBe(121); // persisted
  });

  it('initDefaultOptions falls back to local storage when the server call fails', async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...DEFAULT_OPTIONS, seed: 777 }),
    );
    vi.mocked(getSettings).mockRejectedValue(new Error('boom'));

    initDefaultOptions();
    await flushMicrotasks();

    expect(defaultOptionsStore.get().options.seed).toBe(777);
  });

  it('updateOption persists a non-local key to the backend', () => {
    updateOption('numFrames', 49);
    expect(defaultOptionsStore.get().options.numFrames).toBe(49);
    expect(updateSettings).toHaveBeenCalledWith({ numFrames: 49 });
  });

  it('updateOption keeps apiServerBaseUrl local (no backend call)', () => {
    updateOption('apiServerBaseUrl', 'http://x/api');
    expect(defaultOptionsStore.get().options.apiServerBaseUrl).toBe('http://x/api');
    expect(updateSettings).not.toHaveBeenCalled();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).apiServerBaseUrl).toBe(
      'http://x/api',
    );
  });

  it('resetToDefaults restores defaults and pushes them to the backend', () => {
    defaultOptionsStore.set({ options: { ...DEFAULT_OPTIONS, numFrames: 200 } });
    resetToDefaults();
    expect(defaultOptionsStore.get().options).toEqual(DEFAULT_OPTIONS);
    expect(updateSettings).toHaveBeenCalledWith(DEFAULT_OPTIONS);
  });
});
