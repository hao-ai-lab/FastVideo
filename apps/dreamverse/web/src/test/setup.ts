import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { WebSocket as MockWebSocket } from 'mock-socket';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  cleanup();
});

globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;
if (typeof window !== 'undefined') {
  window.WebSocket = MockWebSocket as unknown as typeof WebSocket;
}

if (!globalThis.crypto?.randomUUID) {
  Object.defineProperty(globalThis, 'crypto', {
    value: {
      ...(globalThis.crypto || {}),
      randomUUID: () => '11111111-1111-4111-8111-111111111111',
    },
    configurable: true,
  });
}

if (!URL.createObjectURL) {
  URL.createObjectURL = vi.fn(() => 'blob:mock-url');
}
if (!URL.revokeObjectURL) {
  URL.revokeObjectURL = vi.fn();
}

if (!globalThis.ResizeObserver) {
  class ResizeObserverMock implements ResizeObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }

  globalThis.ResizeObserver =
    ResizeObserverMock as unknown as typeof ResizeObserver;
}

const matchMediaMock = (query: string): MediaQueryList => ({
  matches: false,
  media: query,
  onchange: null,
  addListener() {},
  removeListener() {},
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent() {
    return false;
  },
} as MediaQueryList);

if (!globalThis.matchMedia) {
  Object.defineProperty(globalThis, 'matchMedia', {
    configurable: true,
    writable: true,
    value: matchMediaMock,
  });
}

if (typeof window !== 'undefined' && window.matchMedia !== matchMediaMock) {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: matchMediaMock,
  });
}

if (typeof window !== 'undefined' && !('maxTouchPoints' in window.navigator)) {
  Object.defineProperty(window.navigator, 'maxTouchPoints', {
    configurable: true,
    get: () => 0,
  });
}

Object.defineProperty(HTMLMediaElement.prototype, 'play', {
  configurable: true,
  writable: true,
  value: vi.fn(() => Promise.resolve()),
});

Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
  configurable: true,
  writable: true,
  value: vi.fn(),
});

Object.defineProperty(HTMLMediaElement.prototype, 'load', {
  configurable: true,
  writable: true,
  value: vi.fn(),
});
