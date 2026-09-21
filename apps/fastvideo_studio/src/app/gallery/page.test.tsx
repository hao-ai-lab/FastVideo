import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import GalleryPage from './page';
import { HeaderActionsProvider } from '@/components/shell/HeaderActionsContext';
import { getJobsList, getModels } from '@/lib/api';
import type { Job } from '@/lib/types';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  getModels: vi.fn(),
  getApiBaseUrl: () => 'http://test.local/api',
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  downloadJobVideo: vi.fn(),
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

beforeEach(() => {
  vi.mocked(getModels).mockResolvedValue([]);
});

describe('GalleryPage', () => {
  it('renders a grid item for a completed inference job', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ prompt: 'a cat surfing a wave' }),
    ]);

    renderGallery();

    expect(await screen.findByText('a cat surfing a wave')).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('inference');
    expect(getModels).toHaveBeenCalledWith(undefined);
  });

  it('shows native controls directly without autoplay or a separate dialog', async () => {
    vi.mocked(getJobsList).mockResolvedValue([makeJob()]);
    const { container } = renderGallery();

    const video = await screen.findByLabelText(
      'Generated video: a cat surfing a wave',
    );
    expect(video).toHaveAttribute('controls');
    expect(video).toHaveAttribute('preload', 'none');
    expect(video).toHaveAttribute('poster', 'http://test.local/api/jobs/job-1/thumbnail');
    expect(video).not.toHaveAttribute('autoplay');
    fireEvent.click(video);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(container.querySelector('video')).toBe(video);
    expect(screen.queryByText('Watch video')).not.toBeInTheDocument();
  });

  it('keeps the original and download available when inline playback fails', async () => {
    vi.mocked(getJobsList).mockResolvedValue([makeJob()]);
    renderGallery();
    const video = await screen.findByLabelText('Generated video: a cat surfing a wave');

    fireEvent.error(video);
    expect(screen.getByText('Preview unavailable')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open original/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download video' })).toBeInTheDocument();
  });

  it('shows model/configuration and combines the catalog dropdown with prompt text', async () => {
    vi.mocked(getModels).mockResolvedValue([{ id: 'Wan2.1', label: 'Wan 2.1' }]);
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'one', name: 'Sunset clip', model_id: 'Wan2.1', prompt: 'A cat surfing', num_inference_steps: 30, seed: 123 }),
      makeJob({ id: 'two', model_id: 'Wan2.1', prompt: 'A dog surfing' }),
      makeJob({ id: 'three', model_id: 'OtherModel', prompt: 'A cat sleeping' }),
    ]);
    renderGallery();
    expect(await screen.findByText('Sunset clip')).toBeInTheDocument();
    expect(screen.getByText('30 steps')).toBeInTheDocument();
    expect(screen.getByText('Seed 123')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Wan 2.1 (Wan2.1)' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'OtherModel' })).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Status' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Model name' }), { target: { value: 'Wan2.1' } });
    fireEvent.change(screen.getByLabelText('Prompt contains'), { target: { value: ' CAT ' } });
    expect(screen.getByText('1 of 3 jobs')).toBeInTheDocument();
    expect(screen.getByText('Sunset clip')).toBeInTheDocument();
    expect(screen.queryByText('A dog surfing')).not.toBeInTheDocument();
    expect(screen.queryByText('A cat sleeping')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(screen.getByText('3 jobs')).toBeInTheDocument();
  });

  it('caps inline previews to 50 and makes older results accessible on the next page', async () => {
    vi.mocked(getJobsList).mockResolvedValue(
      Array.from({ length: 51 }, (_, index) => makeJob({ id: `job-${index}`, name: `Result ${index}` })),
    );
    const { container } = renderGallery();
    expect(await screen.findByText('Page 1 of 2 · Up to 50 previews per page')).toBeInTheDocument();
    await waitFor(() => expect(container.querySelectorAll('video')).toHaveLength(50));
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    await waitFor(() => expect(container.querySelectorAll('video')).toHaveLength(1));
    expect(screen.getByText('Result 50')).toBeInTheDocument();
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
