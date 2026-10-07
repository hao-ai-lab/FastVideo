import { act, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { HeaderActionsProvider } from '@/components/shell/HeaderActionsContext';
import { getDatasets, type Dataset } from '@/lib/api';

import DatasetsPage from './page';

vi.mock('@/lib/api', () => ({
  getDatasets: vi.fn(),
}));

vi.mock('@/components/datasets/AddDatasetButton', () => ({
  default: () => null,
}));

vi.mock('@/components/datasets/CreateDatasetModal', () => ({
  default: () => null,
}));

vi.mock('@/components/datasets/DatasetCard', () => ({
  default: ({ dataset }: { dataset: Dataset }) => <div>{dataset.name}</div>,
}));

function renderPage() {
  return render(
    <HeaderActionsProvider>
      <DatasetsPage />
    </HeaderActionsProvider>,
  );
}

describe('DatasetsPage', () => {
  it('shows loading content before the initial request settles', async () => {
    let resolveDatasets: (datasets: Dataset[]) => void = () => {};
    vi.mocked(getDatasets).mockReturnValue(
      new Promise<Dataset[]>((resolve) => {
        resolveDatasets = resolve;
      }),
    );

    renderPage();
    expect(screen.getByLabelText('Loading datasets')).toBeInTheDocument();
    expect(screen.queryByText('No datasets yet.')).not.toBeInTheDocument();

    act(() => resolveDatasets([]));
    expect(await screen.findByText('No datasets yet.')).toBeInTheDocument();
  });

  it('shows API failures separately from an empty list and retries', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(getDatasets).mockRejectedValueOnce(new Error('network down'));

    renderPage();
    expect(
      await screen.findByText(/Could not load datasets from the Studio API/),
    ).toBeInTheDocument();
    expect(screen.queryByText('No datasets yet.')).not.toBeInTheDocument();

    vi.mocked(getDatasets).mockResolvedValueOnce([]);
    fireEvent.click(screen.getByRole('button', { name: 'Try Again' }));
    expect(await screen.findByText('No datasets yet.')).toBeInTheDocument();
  });
});
