import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import GalleryPage from './page';
import { HeaderActionsProvider } from '@/components/HeaderActionsContext';
import { getJobsList } from '@/lib/api';
import type { Job } from '@/lib/types';

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
}));

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 'job-1',
    model_id: 'wan',
    prompt: 'a cat surfing a wave',
    job_type: 'inference',
    status: 'completed',
    created_at: 1,
    started_at: null,
    finished_at: 2,
    error: null,
    output_path: '/out/clip.mp4',
    log_file_path: null,
    num_inference_steps: 0,
    num_frames: 0,
    height: 0,
    width: 0,
    guidance_scale: 0,
    seed: 0,
    num_gpus: 0,
    progress: 0,
    progress_msg: '',
    phase: '',
    ...overrides,
  };
}

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
