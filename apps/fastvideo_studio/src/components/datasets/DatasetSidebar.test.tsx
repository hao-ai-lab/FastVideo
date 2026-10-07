import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

import DatasetSidebar from '@/components/datasets/DatasetSidebar';
import * as api from '@/lib/api';
import type { Dataset } from '@/lib/api';

vi.mock('@/lib/api');
vi.mock('sonner', () => ({
  toast: { error: vi.fn() },
}));

const mockedApi = vi.mocked(api);

const dataset: Dataset = {
  id: 'ds-1',
  name: 'My Dataset',
  created_at: 0,
};

beforeEach(() => {
  mockedApi.getDatasetFiles.mockResolvedValue({
    file_names: ['a.mp4', 'b.mp4'],
    captions: { 'a.mp4': 'cap a', 'b.mp4': '' },
  });
  mockedApi.getDatasetMediaUrl.mockImplementation(
    (id, fileName) => `http://test/${id}/${fileName}`,
  );
  mockedApi.updateDatasetCaption.mockResolvedValue(undefined);
});

describe('DatasetSidebar', () => {
  it('fills the mobile viewport without reserving main-content width', async () => {
    const onWidthChange = vi.fn();

    render(
      <DatasetSidebar
        dataset={dataset}
        isMobile
        onClose={() => {}}
        onWidthChange={onWidthChange}
      />,
    );

    const drawer = screen.getByRole('dialog', {
      name: 'My Dataset dataset details',
    });
    expect(drawer).toHaveStyle({ width: '100%', maxWidth: 'none' });
    expect(drawer).toHaveAttribute('aria-modal', 'true');
    expect(drawer).toHaveFocus();
    expect(onWidthChange).toHaveBeenCalledWith(0);
  });

  it('lists dataset files after loading', async () => {
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);

    // The first file's caption is rendered once loading resolves.
    expect(await screen.findByDisplayValue('cap a')).toBeInTheDocument();
    expect(mockedApi.getDatasetFiles).toHaveBeenCalledWith('ds-1');

    // One caption editor per returned file.
    const captionFields = screen.getAllByPlaceholderText('Caption');
    expect(captionFields).toHaveLength(2);

    // Media URLs are requested per visible file.
    expect(mockedApi.getDatasetMediaUrl).toHaveBeenCalledWith('ds-1', 'a.mp4');
    expect(mockedApi.getDatasetMediaUrl).toHaveBeenCalledWith('ds-1', 'b.mp4');
  });

  it('shows a fallback when a dataset preview cannot load', async () => {
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);

    const preview = await screen.findByLabelText('Preview of a.mp4');
    fireEvent.error(preview);

    expect(screen.getByText('Preview unavailable')).toBeInTheDocument();
    expect(screen.queryByLabelText('Preview of a.mp4')).not.toBeInTheDocument();
  });

  it('debounces caption save by 500ms', async () => {
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);
    const textarea = await screen.findByDisplayValue('cap a');

    vi.useFakeTimers();
    try {
      fireEvent.change(textarea, { target: { value: 'updated caption' } });

      // Nothing saved immediately or just before the debounce window closes.
      expect(mockedApi.updateDatasetCaption).not.toHaveBeenCalled();
      act(() => {
        vi.advanceTimersByTime(499);
      });
      expect(mockedApi.updateDatasetCaption).not.toHaveBeenCalled();

      // Saved exactly once after the full 500ms.
      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledTimes(1);
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledWith(
        'ds-1',
        'a.mp4',
        'updated caption',
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('coalesces rapid edits into a single debounced save', async () => {
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);
    const textarea = await screen.findByDisplayValue('cap a');

    vi.useFakeTimers();
    try {
      fireEvent.change(textarea, { target: { value: 'one' } });
      act(() => {
        vi.advanceTimersByTime(300);
      });
      fireEvent.change(textarea, { target: { value: 'two' } });
      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledTimes(1);
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledWith(
        'ds-1',
        'a.mp4',
        'two',
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows a failed save and lets the user retry it', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockedApi.updateDatasetCaption
      .mockRejectedValueOnce(new Error('network down'))
      .mockResolvedValueOnce(undefined);
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);
    const textarea = await screen.findByDisplayValue('cap a');

    vi.useFakeTimers();
    try {
      fireEvent.change(textarea, { target: { value: 'needs retry' } });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500);
      });

      expect(screen.getByText(/Not saved/)).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
      await act(async () => {
        await Promise.resolve();
      });

      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledTimes(2);
      expect(mockedApi.updateDatasetCaption).toHaveBeenLastCalledWith(
        'ds-1',
        'a.mp4',
        'needs retry',
      );
      expect(screen.getByText('Saved')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('debounces per file: editing another caption does not cancel a pending save', async () => {
    render(<DatasetSidebar dataset={dataset} onClose={() => {}} />);
    await screen.findByDisplayValue('cap a');
    const [textareaA, textareaB] = screen.getAllByPlaceholderText('Caption');

    vi.useFakeTimers();
    try {
      fireEvent.change(textareaA, { target: { value: 'new cap a' } });
      act(() => {
        vi.advanceTimersByTime(300);
      });
      // Editing b.mp4 inside a.mp4's debounce window must not drop a's save.
      fireEvent.change(textareaB, { target: { value: 'new cap b' } });
      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledTimes(2);
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledWith(
        'ds-1',
        'a.mp4',
        'new cap a',
      );
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledWith(
        'ds-1',
        'b.mp4',
        'new cap b',
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('flushes a pending save on unmount instead of dropping it', async () => {
    const { unmount } = render(
      <DatasetSidebar dataset={dataset} onClose={() => {}} />,
    );
    const textarea = await screen.findByDisplayValue('cap a');

    vi.useFakeTimers();
    try {
      fireEvent.change(textarea, { target: { value: 'edited just before close' } });
      unmount();

      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledTimes(1);
      expect(mockedApi.updateDatasetCaption).toHaveBeenCalledWith(
        'ds-1',
        'a.mp4',
        'edited just before close',
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
