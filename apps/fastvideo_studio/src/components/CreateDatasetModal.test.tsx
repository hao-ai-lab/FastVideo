import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import CreateDatasetModal from '@/components/CreateDatasetModal';
import { createDataset } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  createDataset: vi.fn(),
  uploadRawDataset: vi.fn(),
}));

const mockedCreateDataset = vi.mocked(createDataset);

describe('CreateDatasetModal', () => {
  it('renders nothing when closed', () => {
    render(
      <CreateDatasetModal isOpen={false} onClose={() => {}} onSuccess={() => {}} />,
    );
    expect(screen.queryByText('Add Dataset — Raw')).not.toBeInTheDocument();
  });

  it('renders the form and the default JSON caption upload when open', () => {
    render(
      <CreateDatasetModal isOpen onClose={() => {}} onSuccess={() => {}} />,
    );
    expect(screen.getByText('Add Dataset — Raw')).toBeInTheDocument();
    expect(screen.getByText('Upload video files')).toBeInTheDocument();
    expect(screen.getByText('Upload videos2caption.json')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Create Dataset' }),
    ).toBeInTheDocument();
  });

  it('does not create a dataset when the name is empty', () => {
    const onSuccess = vi.fn();
    render(
      <CreateDatasetModal isOpen onClose={() => {}} onSuccess={onSuccess} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Create Dataset' }));
    expect(mockedCreateDataset).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });
});
