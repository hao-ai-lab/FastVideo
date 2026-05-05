import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ReplicaMonitorPage from './internal/f8a3991c/replica-monitor/page';

describe('Monitor route', () => {
  beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('renders monitor page instead of the regular app shell and polls every 15 seconds', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          replicas: [
            {
              url: 'http://r1:8009',
              healthy: true,
              active_sessions: 2,
              pending_sessions: 1,
              max_available_sessions: 4,
              prompt_provider_success_counts: {
                cerebras_ifm: 9,
                cerebras: 2,
                groq: 1,
              },
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          replicas: [
            {
              url: 'http://r1:8009',
              healthy: true,
              active_sessions: 3,
              pending_sessions: 0,
              max_available_sessions: 4,
              prompt_provider_success_counts: {
                cerebras_ifm: 10,
                cerebras: 2,
                groq: 1,
              },
            },
          ],
        }),
      });
    vi.stubGlobal('fetch', fetchMock);
    const intervalCallbacks: Array<() => void | Promise<void>> = [];
    vi.spyOn(globalThis, 'setInterval').mockImplementation(
      ((callback: TimerHandler) => {
        intervalCallbacks.push(callback as () => void | Promise<void>);
        return 1 as unknown as ReturnType<typeof setInterval>;
      }) as unknown as typeof setInterval,
    );
    vi.spyOn(globalThis, 'clearInterval').mockImplementation(
      (() => {}) as typeof clearInterval,
    );

    render(<ReplicaMonitorPage />);

    expect(
      await screen.findByRole('heading', { name: 'Replica Session Monitor' }),
    ).toBeInTheDocument();
    expect(await screen.findByText('http://r1:8009')).toBeInTheDocument();
    expect(await screen.findByText('4')).toBeInTheDocument();
    expect(
      await screen.findByText('IFM 9 / Cerebras 2 / Groq 1'),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Preset-driven video continuation' }),
    ).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await intervalCallbacks[0]?.();
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  });

  it('renders error state when monitor fetch fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'boom' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<ReplicaMonitorPage />);

    expect(
      await screen.findByText('boom'),
    ).toBeInTheDocument();
  });
});
