import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import GalleryPage from './page';
import { HeaderActionsProvider } from '@/components/HeaderActionsContext';
import { getJobsList } from '@/lib/api';
import type { Job } from '@/lib/types';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
}));

const makeJob = (overrides: Partial<Job> = {}): Job =>
  makeBaseJob({
    model_id: 'wan',
    prompt: 'a cat surfing a wave',
    status: 'completed',
    created_at: 1,
    finished_at: 2,
    output_path: '/out/clip.mp4',
    ...overrides,
  });

function renderGallery() {
  return render(
    <HeaderActionsProvider>
      <GalleryPage />
    </HeaderActionsProvider>,
  );
}

describe('GalleryPage', () => {
  it('renders a grid item for a completed inference job', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ prompt: 'a cat surfing a wave' }),
    ]);

    renderGallery();

    expect(await screen.findByText('a cat surfing a wave')).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('inference');
  });

  it('shows the empty state when no completed videos exist', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ status: 'running', output_path: null }),
    ]);

    renderGallery();

    expect(
      await screen.findByText('No completed videos yet'),
    ).toBeInTheDocument();
    expect(
      screen.queryByText('a cat surfing a wave'),
    ).not.toBeInTheDocument();
  });
});
