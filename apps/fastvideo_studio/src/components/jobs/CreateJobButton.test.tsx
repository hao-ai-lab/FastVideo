import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobButton from './CreateJobButton';
import { getDatasets, getModels } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  uploadImage: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(getModels).mockResolvedValue([
    { id: 'wan/t2v-1.3b', label: 'Wan T2V' },
  ]);
  vi.mocked(getDatasets).mockResolvedValue([]);
});

describe('CreateJobButton', () => {
  it('opens the workload menu on click and selects an item', async () => {
    const user = userEvent.setup();
    render(<CreateJobButton jobType="inference" />);

    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    await user.click(screen.getByRole('menuitem', { name: /I2V/i }));

    expect(
      screen.getByRole('dialog', { name: 'New Inference Job (I2V)' }),
    ).toBeInTheDocument();
  });

  it('opens and operates the workload menu from the keyboard', async () => {
    const user = userEvent.setup();
    render(<CreateJobButton jobType="inference" />);

    const trigger = screen.getByRole('button', { name: 'Create Job' });
    trigger.focus();
    await user.keyboard('{Enter}');

    const firstItem = await screen.findByRole('menuitem', { name: /T2V/i });
    expect(firstItem).toHaveFocus();
    await user.keyboard('{Enter}');

    expect(
      screen.getByRole('dialog', { name: 'New Inference Job (T2V)' }),
    ).toBeInTheDocument();
    await user.keyboard('{Escape}');
    await waitFor(() =>
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument(),
    );
    await waitFor(() =>
      expect(document.body.style.pointerEvents).not.toBe('none'),
    );
    expect(trigger).toHaveFocus();
  });

  it.each(['inference', 'finetuning', 'distillation'] as const)(
    'restores page interaction after closing the real %s dialog',
    async (jobType) => {
      const user = userEvent.setup();
      render(<CreateJobButton jobType={jobType} />);
      const trigger = screen.getByRole('button', { name: 'Create Job' });

      // Keep the real Dialog mounted: mocking it hides conflicting Radix layers.
      for (let attempt = 0; attempt < 2; attempt++) {
        await user.click(trigger);
        await user.click(screen.getAllByRole('menuitem')[0]);
        const dialog = screen.getByRole('dialog');
        await user.click(
          within(dialog).getByRole('button', { name: 'Close' }),
        );
        await waitFor(() =>
          expect(screen.queryByRole('dialog')).not.toBeInTheDocument(),
        );
        await waitFor(() =>
          expect(document.body.style.pointerEvents).not.toBe('none'),
        );
        expect(trigger).toHaveFocus();
      }
    },
  );
});
