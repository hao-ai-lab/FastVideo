import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import DatasetCard from '@/components/DatasetCard';
import { deleteDataset } from '@/lib/api';
import type { Dataset } from '@/lib/api';
import { setActiveDatasetId } from '@/stores/activeDataset';

vi.mock('@/lib/api', () => ({
  deleteDataset: vi.fn(),
}));

const mockedDeleteDataset = vi.mocked(deleteDataset);

const dataset: Dataset = {
  id: 'ds-1',
  name: 'My Dataset',
  created_at: 0,
  file_count: 3,
  size_bytes: 2048,
};

beforeEach(() => {
  setActiveDatasetId(null);
});

describe('DatasetCard', () => {
  it('renders the name, file count and human-readable size', () => {
    render(<DatasetCard dataset={dataset} onUpdated={() => {}} />);
    expect(screen.getByText('My Dataset')).toBeInTheDocument();
    expect(screen.getByText('3 files · 2.0 KB')).toBeInTheDocument();
  });

  it('uses the singular "file" label and byte units for a small dataset', () => {
    render(
      <DatasetCard
        dataset={{ ...dataset, file_count: 1, size_bytes: 512 }}
        onUpdated={() => {}}
      />,
    );
    expect(screen.getByText('1 file · 512 B')).toBeInTheDocument();
  });

  it('calls onSelect when the card body is clicked', () => {
    const onSelect = vi.fn();
    render(
      <DatasetCard dataset={dataset} onUpdated={() => {}} onSelect={onSelect} />,
    );
    fireEvent.click(screen.getByText('My Dataset'));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('deletes after confirmation and notifies the parent without selecting', async () => {
    const onUpdated = vi.fn();
    const onSelect = vi.fn();
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    mockedDeleteDataset.mockResolvedValue(undefined);

    render(
      <DatasetCard
        dataset={dataset}
        onUpdated={onUpdated}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    expect(confirmSpy).toHaveBeenCalledWith('Delete dataset "My Dataset"?');
    expect(mockedDeleteDataset).toHaveBeenCalledWith('ds-1');
    await waitFor(() => expect(onUpdated).toHaveBeenCalledTimes(1));
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('selects on keyboard activation of the card body but not of the Delete button', () => {
    const onSelect = vi.fn();
    render(
      <DatasetCard dataset={dataset} onUpdated={() => {}} onSelect={onSelect} />,
    );

    // Activating the Delete button must not bubble into a card selection.
    fireEvent.keyDown(screen.getByRole('button', { name: 'Delete' }), {
      key: 'Enter',
    });
    expect(onSelect).not.toHaveBeenCalled();

    // Activating the card body itself does select.
    fireEvent.keyDown(screen.getByText('My Dataset'), { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('does not delete when the user cancels the confirm dialog', () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    render(<DatasetCard dataset={dataset} onUpdated={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(mockedDeleteDataset).not.toHaveBeenCalled();
  });

  it('applies selected styling when it is the active dataset', () => {
    setActiveDatasetId('ds-1');
    const { container } = render(
      <DatasetCard dataset={dataset} onUpdated={() => {}} />,
    );
    expect(container.firstChild).toHaveClass('border-accent', 'bg-accent/5');
  });
});
