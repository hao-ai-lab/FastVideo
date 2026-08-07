import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import EngineConsole from './EngineConsole';
import { getEngineLogs } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  getEngineLogs: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(getEngineLogs).mockResolvedValue({
    lines: ['[engine] booted', '[engine] worker heartbeat ok'],
    total: 2,
  });
});

describe('EngineConsole', () => {
  it('is collapsed by default and does not fetch', () => {
    render(<EngineConsole />);

    expect(
      screen.getByRole('button', { name: 'Engine output' }),
    ).toHaveAttribute('aria-expanded', 'false');
    expect(getEngineLogs).not.toHaveBeenCalled();
  });

  it('shows log lines and polls with the cursor while open', async () => {
    vi.useFakeTimers();
    try {
      render(<EngineConsole />);
      fireEvent.click(screen.getByRole('button', { name: 'Engine output' }));

      // Flush the immediate poll fired on expand.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(getEngineLogs).toHaveBeenCalledWith(0);
      expect(screen.getByText(/\[engine\] booted/)).toBeInTheDocument();

      // The 2s interval polls again, from the previous total.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      expect(getEngineLogs).toHaveBeenLastCalledWith(2);

      // Collapsing stops the polling.
      fireEvent.click(screen.getByRole('button', { name: 'Engine output' }));
      const calls = vi.mocked(getEngineLogs).mock.calls.length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10000);
      });
      expect(getEngineLogs).toHaveBeenCalledTimes(calls);
    } finally {
      vi.useRealTimers();
    }
  });

  it('clear view empties the scrollback locally', async () => {
    const user = userEvent.setup();
    render(<EngineConsole />);

    await user.click(screen.getByRole('button', { name: 'Engine output' }));
    expect(await screen.findByText(/\[engine\] booted/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Clear view' }));
    expect(screen.queryByText(/\[engine\] booted/)).not.toBeInTheDocument();
    expect(screen.getByText('Waiting for engine output…')).toBeInTheDocument();
  });
});
