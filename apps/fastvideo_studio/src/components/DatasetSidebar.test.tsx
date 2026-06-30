import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

import DatasetSidebar from '@/components/DatasetSidebar';
import * as api from '@/lib/api';
import type { Dataset } from '@/lib/api';

vi.mock('@/lib/api');

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
});
