import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { downloadBlob } from '@/lib/utils';

import DownloadCaptions from './DownloadCaptions';

vi.mock('@/lib/utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/utils')>();
  return { ...actual, downloadBlob: vi.fn() };
});

const props = {
  fileNames: ['b.mp4', 'a.mp4'],
  captions: { 'a.mp4': 'cap a', 'b.mp4': 'cap b' },
};

describe('DownloadCaptions', () => {
  it('opens the format menu on click and downloads the selection', async () => {
    const user = userEvent.setup();
    render(<DownloadCaptions {...props} />);

    await user.click(
      screen.getByRole('button', { name: 'Download Captions' }),
    );
    await user.click(screen.getByRole('menuitem', { name: 'JSON' }));

    expect(vi.mocked(downloadBlob)).toHaveBeenCalledWith(
      expect.any(Blob),
      'videos2caption.json',
    );
  });

  it('operates entirely from the keyboard', async () => {
    const user = userEvent.setup();
    render(<DownloadCaptions {...props} />);

    screen.getByRole('button', { name: 'Download Captions' }).focus();
    await user.keyboard('{Enter}');
    const first = await screen.findByRole('menuitem', { name: 'JSON' });
    expect(first).toHaveFocus();
    await user.keyboard('{ArrowDown}{Enter}');

    expect(vi.mocked(downloadBlob)).toHaveBeenCalledWith(
      expect.any(Blob),
      'videos.txt',
    );
  });
});
