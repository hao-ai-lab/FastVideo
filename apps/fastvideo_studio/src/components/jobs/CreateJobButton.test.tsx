import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { toast } from 'sonner';
import { describe, expect, it, vi } from 'vitest';

import CreateJobButton from './CreateJobButton';

vi.mock('sonner', () => ({ toast: { error: vi.fn(), warning: vi.fn() } }));

vi.mock('./CreateJobModal', () => ({
  default: ({
    isOpen,
    workloadType,
    prefillJob,
  }: {
    isOpen: boolean;
    workloadType: string;
    prefillJob?: { name?: string; prompt: string } | null;
  }) =>
    isOpen ? (
      <div
        role="dialog"
        data-workload-type={workloadType}
        data-prefill-name={prefillJob?.name ?? ''}
        data-prefill-prompt={prefillJob?.prompt ?? ''}
      >
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

    // The dialog opens a frame after the menu selection (see CreateJobButton's
    // openModal: deferred via requestAnimationFrame to dodge a Radix
    // DropdownMenu + Dialog pointer-events race), so wait for it rather than
    // asserting synchronously.
    expect(await screen.findByRole('dialog')).toHaveAttribute(
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

    expect(await screen.findByRole('dialog')).toHaveAttribute(
      'data-workload-type',
      't2v',
    );
  });

  describe('importing a job file', () => {
    const file = (contents: string, name = 'job.json') =>
      new File([contents], name, { type: 'application/json' });

    it('opens the form for the file\'s workload with the job pre-filled', async () => {
      const user = userEvent.setup();
      render(<CreateJobButton jobType="inference" />);

      await user.click(screen.getByRole('button', { name: 'Create Job' }));
      await user.click(screen.getByRole('menuitem', { name: /Import job JSON/ }));
      await user.upload(
        screen.getByLabelText('Import job JSON file'),
        file(
          JSON.stringify({
            model_id: 'MiniMaxAI/MiniMax-H3',
            name: 'from-file',
            workload_type: 'i2v',
            prompt: ['summary:', 'Hi.'],
          }),
        ),
      );

      const dialog = await screen.findByRole('dialog');
      expect(dialog).toHaveAttribute('data-workload-type', 'i2v');
      expect(dialog).toHaveAttribute('data-prefill-name', 'from-file');
      expect(dialog).toHaveAttribute('data-prefill-prompt', 'summary:\nHi.');
    });

    it('reports an unreadable file instead of opening the form', async () => {
      const user = userEvent.setup();
      render(<CreateJobButton jobType="inference" />);

      await user.click(screen.getByRole('button', { name: 'Create Job' }));
      await user.click(screen.getByRole('menuitem', { name: /Import job JSON/ }));
      await user.upload(screen.getByLabelText('Import job JSON file'), file('not json', 'bad.json'));

      await vi.waitFor(() => expect(toast.error).toHaveBeenCalled());
      expect(vi.mocked(toast.error).mock.calls.at(-1)?.[0]).toContain("bad.json");
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('a normal open after an import starts clean', async () => {
      const user = userEvent.setup();
      render(<CreateJobButton jobType="inference" />);

      await user.click(screen.getByRole('button', { name: 'Create Job' }));
      await user.click(screen.getByRole('menuitem', { name: /Import job JSON/ }));
      await user.upload(
        screen.getByLabelText('Import job JSON file'),
        file(JSON.stringify({ model_id: 'm', name: 'from-file', prompt: 'p' })),
      );
      expect(await screen.findByRole('dialog')).toHaveAttribute('data-prefill-name', 'from-file');

      await user.click(screen.getByRole('button', { name: 'Create Job' }));
      await user.click(await screen.findByRole('menuitem', { name: /T2V/i }));
      await vi.waitFor(() => expect(screen.getByRole('dialog')).toHaveAttribute('data-prefill-name', ''));
    });

    it('is not offered on the training pages', async () => {
      const user = userEvent.setup();
      render(<CreateJobButton jobType="finetuning" />);
      await user.click(screen.getByRole('button', { name: 'Create Job' }));
      expect(screen.queryByRole('menuitem', { name: /Import job JSON/ })).not.toBeInTheDocument();
    });
  });
});
