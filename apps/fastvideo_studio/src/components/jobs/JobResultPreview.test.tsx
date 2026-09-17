import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

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

afterEach(() => vi.unstubAllGlobals());

describe('JobResultPreview', () => {
  it('renders native inline controls with a cached poster and no autoplay or eager media loading', () => {
    const { container } = render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    const video = screen.getByLabelText('Generated video: a prompt');
    expect(video.tagName).toBe('VIDEO');
    expect(video).toHaveAttribute('controls');
    expect(video).toHaveAttribute('playsinline');
    expect(video).toHaveAttribute('preload', 'none');
    expect(video).toHaveAttribute('poster', 'http://test.local/api/jobs/job-1/thumbnail');
    expect(video).toHaveAttribute('src', 'http://test.local/api/jobs/job-1/video');
    expect(video).not.toHaveAttribute('autoplay');
    expect(container.querySelector('img')).toBeNull();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByText(/Watch video|View image/)).not.toBeInTheDocument();
  });

  it('mounts videos near the viewport and releases media buffers when scrolled away', () => {
    let notify!: IntersectionObserverCallback;
    const observe = vi.fn();
    const disconnect = vi.fn();
    let options: IntersectionObserverInit | undefined;
    vi.stubGlobal('IntersectionObserver', class {
      constructor(callback: IntersectionObserverCallback, init?: IntersectionObserverInit) {
        notify = callback;
        options = init;
      }
      observe = observe;
      disconnect = disconnect;
    });
    const { container, unmount } = render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    expect(observe).toHaveBeenCalledOnce();
    expect(options).toEqual({ rootMargin: '200px' });
    expect(container.querySelector('video, img')).toBeNull();
    const intersect = (isIntersecting: boolean) => act(() => {
      notify([{ isIntersecting } as IntersectionObserverEntry], {} as IntersectionObserver);
    });

    intersect(true);
    const video = screen.getByLabelText('Generated video: a prompt') as HTMLVideoElement;
    const pause = vi.spyOn(video, 'pause');
    const load = vi.spyOn(video, 'load');
    expect(video).toHaveAttribute('preload', 'none');
    intersect(false);
    expect(container.querySelector('video')).toBeNull();
    expect(pause).toHaveBeenCalledOnce();
    expect(load).toHaveBeenCalledOnce();
    expect(video).not.toHaveAttribute('src');
    intersect(true);
    expect(screen.getByLabelText('Generated video: a prompt')).toHaveAttribute('src', 'http://test.local/api/jobs/job-1/video');
    unmount();
    expect(disconnect).toHaveBeenCalledOnce();
  });

  it('retains the media source after StrictMode replays cleanup in development', () => {
    render(<StrictMode><JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} /></StrictMode>);
    expect(screen.getByLabelText('Generated video: a prompt')).toHaveAttribute('src', 'http://test.local/api/jobs/job-1/video');
  });

  it('keeps native media and image links separate from the surrounding job card', () => {
    const onCardClick = vi.fn();
    render(
      <div onClick={onCardClick}>
        <JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />
        <JobResultPreview job={makeJob({ id: 'image-job', output_path: '/out/image.png' })} />
      </div>,
    );
    fireEvent.click(screen.getByLabelText('Generated video: a prompt'));
    fireEvent.click(screen.getByRole('img'));
    expect(onCardClick).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('pauses other Studio results when playing a video, leaving unrelated videos alone', () => {
    render(
      <>
        <JobResultPreview job={makeJob({ id: 'first', prompt: 'first', output_path: '/out/first.mp4' })} />
        <JobResultPreview job={makeJob({ id: 'second', prompt: 'second', output_path: '/out/second.mp4' })} />
        <video aria-label="Unrelated video" />
      </>,
    );
    const first = screen.getByLabelText('Generated video: first') as HTMLVideoElement;
    const second = screen.getByLabelText('Generated video: second') as HTMLVideoElement;
    const unrelated = screen.getByLabelText('Unrelated video') as HTMLVideoElement;
    const firstPause = vi.spyOn(first, 'pause');
    const secondPause = vi.spyOn(second, 'pause');
    const unrelatedPause = vi.spyOn(unrelated, 'pause');
    fireEvent.play(second);
    expect(firstPause).toHaveBeenCalledOnce();
    expect(secondPause).not.toHaveBeenCalled();
    expect(unrelatedPause).not.toHaveBeenCalled();
  });

  it('renders image outputs directly using a lazy cached thumbnail linked to the original', () => {
    const { container } = render(<JobResultPreview job={makeJob({ output_path: '/out/image.PNG' })} />);
    const image = screen.getByRole('img', { name: 'a prompt' });
    expect(image).toHaveAttribute('src', 'http://test.local/api/jobs/job-1/thumbnail');
    expect(image).toHaveAttribute('loading', 'lazy');
    expect(image).toHaveAttribute('decoding', 'async');
    expect(screen.getByRole('link', { name: 'Open original image' })).toHaveAttribute('href', 'http://test.local/api/jobs/job-1/video');
    expect(container.querySelector('video')).toBeNull();
    expect(screen.getByRole('button', { name: 'Download image' })).toBeInTheDocument();
  });

  it.each(['/out/clip.mp4', '/out/image.PNG'])('mounts no media beyond the cap for %s, retaining original access', (output_path) => {
    const onCardClick = vi.fn();
    const { container } = render(
      <div onClick={onCardClick}>
        <JobResultPreview job={makeJob({ output_path })} thumbnailEnabled={false} />
      </div>,
    );
    expect(container.querySelector('img, video')).toBeNull();
    expect(screen.getByText('Result available')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: 'Open original' });
    expect(link).toHaveAttribute('href', 'http://test.local/api/jobs/job-1/video');
    fireEvent.click(link);
    expect(onCardClick).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /Download/ })).toBeInTheDocument();
  });

  it.each(['/out/clip.mp4', '/out/image.PNG'])('offers retry and original access if inline media fails for %s', (output_path) => {
    const { container } = render(<JobResultPreview job={makeJob({ output_path })} />);
    fireEvent.error(container.querySelector('video, img')!);
    expect(screen.getByRole('status')).toHaveTextContent('Preview unavailable');
    expect(screen.getByRole('link', { name: 'Open original' })).toBeInTheDocument();
    expect(container.querySelector('video, img')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Retry preview' }));
    expect(container.querySelector('video, img')).not.toBeNull();
  });

  it('downloads the full result through the API without opening a modal or the surrounding card', async () => {
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
    expect(button).toHaveClass('right-2', 'top-2');
    expect(container.querySelector('button button')).toBeNull();
    fireEvent.click(button);
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'job_job-1.mp4'));
    expect(downloadJobVideo).toHaveBeenCalledWith('job-1');
    expect(onCardClick).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('shows download failures next to the media and allows retrying', async () => {
    const blob = new Blob(['image']);
    vi.mocked(downloadJobVideo)
      .mockRejectedValueOnce(new Error('The output is unavailable.'))
      .mockResolvedValueOnce(blob);
    render(<JobResultPreview job={makeJob({ output_path: '/out/image.PNG' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download image' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The output is unavailable.');
    fireEvent.click(screen.getByRole('button', { name: 'Download image' }));
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith(blob, 'job_job-1.png'));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('disables downloads while the shared request is in flight', async () => {
    let completeDownload!: (blob: Blob) => void;
    vi.mocked(downloadJobVideo).mockReturnValue(new Promise((resolve) => { completeDownload = resolve; }));
    render(<JobResultPreview job={makeJob({ output_path: '/out/clip.mp4' })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Download video' }));
    expect(screen.getByRole('button', { name: 'Download video' })).toBeDisabled();
    completeDownload(new Blob(['video']));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Download video' })).toBeEnabled());
    expect(downloadJobVideo).toHaveBeenCalledTimes(1);
  });
});
