import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import JobResultPreview from '@/components/jobs/JobResultPreview';
import { downloadJobVideo } from '@/lib/api';
import { downloadBlob } from '@/lib/utils';
import { makeJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getApiBaseUrl: () => 'http://test.local/api',
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  downloadJobVideo: vi.fn(),
}));

vi.mock('@/lib/utils', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/utils')>(),
  downloadBlob: vi.fn(),
}));

describe('JobResultPreview', () => {
  it('downloads from a thumbnail without opening the player or the surrounding job card', async () => {
    const blob = new Blob(['video']);
    vi.mocked(downloadJobVideo).mockResolvedValue(blob);
    const onCardClick = vi.fn();
    const { container } = render(
      <div onClick={onCardClick}>
        <JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />
      </div>,
    );
    const button = screen.getByRole('button', { name: 'Download video' });
    expect(button).toHaveAttribute('title', 'Download video');
    expect(container.querySelector('button button')).toBeNull();
    fireEvent.click(button);
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'job_job-1.mp4'));
    expect(onCardClick).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(container.querySelector('video')).toBeNull();
  });

  it('keeps preview clicks separate from the surrounding job card', () => {
    const onCardClick = vi.fn();
    render(
      <div onClick={onCardClick}>
        <JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />
      </div>,
    );
    fireEvent.click(screen.getByRole('button', { name: /Preview result:/ }));
    fireEvent.click(screen.getByRole('dialog'));
    expect(onCardClick).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('downloads a full result through the API instead of navigating to a cross-origin media URL', async () => {
    const blob = new Blob(['video']);
    vi.mocked(downloadJobVideo).mockResolvedValue(blob);
    render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    fireEvent.click(screen.getByRole('button', { name: /Preview result:/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Download video' }));
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'job_job-1.mp4'));
    expect(downloadJobVideo).toHaveBeenCalledWith('job-1');
  });

  it('reports a missing output without closing the preview', async () => {
    vi.mocked(downloadJobVideo).mockRejectedValue(new Error('The output was deleted.'));
    render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    fireEvent.click(screen.getByRole('button', { name: /Preview result:/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Download video' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The output was deleted.');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('shows download failures next to the thumbnail and allows retrying', async () => {
    const blob = new Blob(['image']);
    vi.mocked(downloadJobVideo)
      .mockRejectedValueOnce(new Error('The output is unavailable.'))
      .mockResolvedValueOnce(blob);
    render(<JobResultPreview job={makeJob({ output_path: '/out/image.PNG' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download image' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The output is unavailable.');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download image' }));
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'job_job-1.png'));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('disables the download controls while the shared request is in flight', async () => {
    let completeDownload!: (blob: Blob) => void;
    vi.mocked(downloadJobVideo).mockReturnValue(new Promise((resolve) => { completeDownload = resolve; }));
    render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download video' }));
    expect(screen.getByRole('button', { name: 'Download video' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: /Preview result:/ }));
    expect(screen.getByRole('button', { name: 'Download video' })).toBeDisabled();
    completeDownload(new Blob(['video']));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Download video' })).toBeEnabled());
    expect(downloadJobVideo).toHaveBeenCalledTimes(1);
  });

  it('uses an image for image outputs and skips thumbnails when the queue cap is reached', () => {
    const { container } = render(
      <JobResultPreview job={makeJob({ output_path: '/out/image.PNG' })} thumbnailEnabled={false} />,
    );
    expect(container.querySelectorAll('img')).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: /Preview result:/ }));
    expect(screen.getByAltText('a prompt')).toHaveAttribute('src', 'http://test.local/api/jobs/job-1/video');
    expect(screen.getByRole('dialog').querySelector('video')).toBeNull();
    expect(screen.getByRole('button', { name: 'Download image' })).toBeInTheDocument();
  });
});
