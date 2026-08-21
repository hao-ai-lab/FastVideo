import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import CreateJobButton from './CreateJobButton';

vi.mock('./CreateJobModal', () => ({
  default: ({
    isOpen,
    workloadType,
  }: {
    isOpen: boolean;
    workloadType: string;
  }) =>
    isOpen ? (
      <div role="dialog" data-workload-type={workloadType}>
        Create job form
      </div>
    ) : null,
}));

describe('CreateJobButton', () => {
  it('opens the workload menu on click and selects an item', async () => {
    const user = userEvent.setup();
    render(<CreateJobButton jobType="inference" />);

    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    await user.click(screen.getByRole('menuitem', { name: /I2V/i }));

    expect(screen.getByRole('dialog')).toHaveAttribute(
      'data-workload-type',
      'i2v',
    );
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

    expect(screen.getByRole('dialog')).toHaveAttribute(
      'data-workload-type',
      't2v',
    );
  });
});
