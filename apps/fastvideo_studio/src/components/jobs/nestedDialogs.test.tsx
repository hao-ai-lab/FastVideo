import * as React from 'react';
import { flushSync } from 'react-dom';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { createJob, getDatasets, getModels } from '@/lib/api';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';

vi.mock('sonner', () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));
vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  uploadImage: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const H3 = { id: 'minimax-h3/i2v', label: 'MiniMax H3', type: 'i2v' };

// A browser runs a microtask checkpoint -- so React flushes -- between the
// listeners of a single native event; jsdom's synchronous dispatch never does.
// Radix decides whether a parent dialog should dismiss from the layer stack at
// the moment its listener runs, so without this the bug can't show up here.
const wrapped = new WeakMap<object, EventListener>();
beforeEach(() => {
  const realAdd = Document.prototype.addEventListener;
  const realRemove = Document.prototype.removeEventListener;
  vi.spyOn(document, 'addEventListener').mockImplementation(function (
    this: Document,
    type: string,
    listener: unknown,
    options?: unknown,
  ) {
    if (typeof listener === 'function' && ['keydown', 'pointerdown', 'click'].includes(type)) {
      const flushing: EventListener = (e) => {
        (listener as EventListener)(e);
        flushSync(() => {});
      };
      wrapped.set(listener, flushing);
      return realAdd.call(this, type, flushing, options as AddEventListenerOptions);
    }
    return realAdd.call(this, type, listener as EventListener, options as AddEventListenerOptions);
  } as typeof document.addEventListener);
  vi.spyOn(document, 'removeEventListener').mockImplementation(function (
    this: Document,
    type: string,
    listener: unknown,
    options?: unknown,
  ) {
    return realRemove.call(
      this,
      type,
      (wrapped.get(listener as object) ?? listener) as EventListener,
      options as EventListenerOptions,
    );
  } as typeof document.removeEventListener);

  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue([H3]);
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(createJob).mockResolvedValue({ id: 'j' } as never);
});

function Harness() {
  const [open, setOpen] = React.useState(true);
  return (
    <>
      <div data-testid="state">{open ? 'open' : 'closed'}</div>
      <CreateJobModal
        isOpen={open}
        onClose={() => setOpen(false)}
        onSuccess={() => {}}
        jobType="inference"
        workloadType="i2v"
      />
    </>
  );
}

type User = ReturnType<typeof userEvent.setup>;
type Closer = (user: User, dialog: HTMLElement) => Promise<void>;

async function renderH3Modal(user: User) {
  render(<Harness />);
  await screen.findByRole('option', { name: /MiniMax H3/ });
  await user.selectOptions(screen.getByLabelText('Model'), H3.id);
}

// Radix marks every dialog under the topmost one aria-hidden, which also hides its
// accessible name, so the job dialog is identified by its title text instead.
const expectParentStillOpen = () => {
  expect(screen.getByTestId('state')).toHaveTextContent('open');
  expect(screen.getByText(/New Inference Job/)).toBeInTheDocument();
};
// [0] is the job dialog's overlay; the last is the topmost dialog's.
const topOverlay = () => [...document.querySelectorAll('.bg-black\\/70')].at(-1) as Element;

const CLOSERS: [string, Closer][] = [
  ['Cancel', (u, d) => u.click(within(d).getByRole('button', { name: 'Cancel' }))],
  ['the X button', (u, d) => u.click(within(d).getByRole('button', { name: 'Close' }))],
  ['Escape', (u) => u.keyboard('{Escape}')],
  ['a click on the overlay', (u) => u.click(topOverlay())],
];

describe('closing a dialog opened from the job form', () => {
  describe.each([
    ['shot list', 'Open shot list', 'Shot list', [...CLOSERS, ['Apply', (u: User, d: HTMLElement) => u.click(within(d).getByRole('button', { name: 'Apply' }))] as [string, Closer]]],
    ['dialogue editor', 'Open dialogue editor', 'Dialogue editor', [...CLOSERS, ['Apply to field', (u: User, d: HTMLElement) => u.click(within(d).getByRole('button', { name: 'Apply to field' }))] as [string, Closer]]],
  ] as const)('%s', (_label, openButton, title, closers) => {
    it.each(closers)('closing with %s leaves the job form open', async (_name, close) => {
      const user = userEvent.setup();
      await renderH3Modal(user);
      await user.click(await screen.findByRole('button', { name: openButton }));
      const child = await screen.findByRole('dialog', { name: title });

      await close(user, child);

      await waitFor(() => expect(screen.queryByRole('dialog', { name: title })).not.toBeInTheDocument());
      expectParentStillOpen();
    });
  });

  it('the job form is still usable afterwards', async () => {
    const user = userEvent.setup();
    await renderH3Modal(user);
    await user.click(await screen.findByRole('button', { name: 'Open shot list' }));
    const child = await screen.findByRole('dialog', { name: 'Shot list' });
    await user.click(within(child).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Shot list' })).not.toBeInTheDocument());

    await user.type(screen.getByLabelText('Name (optional)'), 'still here');
    expect(screen.getByLabelText('Name (optional)')).toHaveValue('still here');
    // ...and the dialog can be opened again
    await user.click(screen.getByRole('button', { name: 'Open shot list' }));
    expect(await screen.findByRole('dialog', { name: 'Shot list' })).toBeInTheDocument();
  });

  describe('the script import inside the shot list', () => {
    it.each(CLOSERS.slice(0, 4))('closing it with %s leaves the shot list and the job form open', async (_name, close) => {
      const user = userEvent.setup();
      await renderH3Modal(user);
      await user.click(await screen.findByRole('button', { name: 'Open shot list' }));
      const shotList = await screen.findByRole('dialog', { name: 'Shot list' });
      await user.click(within(shotList).getByRole('button', { name: 'Import script…' }));
      const importDialog = await screen.findByRole('dialog', { name: 'Import script' });

      await close(user, importDialog);

      await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Import script' })).not.toBeInTheDocument());
      expect(screen.getByRole('dialog', { name: 'Shot list' })).toBeInTheDocument();
      expectParentStillOpen();
    });
  });
});
