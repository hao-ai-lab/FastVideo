import { act, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import JobDetailsSidebar from './JobDetailsSidebar';
import { getJobLogs } from '@/lib/api';
import type { Job } from '@/lib/types';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getJobLogs: vi.fn(),
  downloadJobLog: vi.fn(),
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
}));

const makeJob = (overrides: Partial<Job> = {}): Job =>
  makeBaseJob({
    status: 'running',
    log_file_path: '/logs/job-1.log',
    ...overrides,
  });

describe('JobDetailsSidebar', () => {
  it('fills the mobile viewport without reserving main-content width', async () => {
    vi.mocked(getJobLogs).mockResolvedValue({
      lines: [],
      total: 0,
      progress: 0,
      progress_msg: '',
      phase: '',
    });
    const onWidthChange = vi.fn();

    render(
      <JobDetailsSidebar
        job={makeJob({ status: 'completed' })}
        isMobile
        onClose={vi.fn()}
        onWidthChange={onWidthChange}
      />,
    );

    const drawer = screen.getByRole('dialog', { name: 'Job details' });
    expect(drawer).toHaveStyle({ width: '100%', maxWidth: 'none' });
    expect(drawer).toHaveAttribute('aria-modal', 'true');
    expect(drawer).toHaveFocus();
    expect(onWidthChange).toHaveBeenCalledWith(0);
  });

  it('plays completed inference output inline; running jobs get no player', async () => {
    vi.mocked(getJobLogs).mockResolvedValue({
      lines: [],
      total: 0,
      progress: 0,
      progress_msg: '',
      phase: '',
    });

    const { rerender } = render(
      <JobDetailsSidebar
        job={makeJob({
          status: 'completed',
          output_path: '/outputs/job-1.mp4',
          prompt: 'a cat surfing a wave',
        })}
        onClose={vi.fn()}
      />,
    );

    const video = screen.getByLabelText('Generated video: a cat surfing a wave');
    expect(video.tagName).toBe('VIDEO');
    expect(video).toHaveAttribute('controls');
    expect(video).toHaveAttribute(
      'src',
      'http://test.local/api/jobs/job-1/video',
    );

    rerender(
      <JobDetailsSidebar
        job={makeJob({ status: 'running', output_path: null })}
        onClose={vi.fn()}
      />,
    );
    expect(screen.queryByLabelText(/Generated video/)).not.toBeInTheDocument();
  });

  it('renders log lines streamed from the job log poll', async () => {
    vi.mocked(getJobLogs).mockResolvedValue({
      lines: ['boot sequence started', 'loading model weights'],
      total: 2,
      progress: 0,
      progress_msg: '',
      phase: '',
    });

    render(
      <JobDetailsSidebar job={makeJob({ status: 'running' })} onClose={vi.fn()} />,
    );

    expect(await screen.findByText(/boot sequence started/)).toBeInTheDocument();
    expect(screen.getByText(/loading model weights/)).toBeInTheDocument();
    expect(getJobLogs).toHaveBeenCalledWith('job-1', 0);
  });

  it('keeps polling while the job is running', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(getJobLogs).mockResolvedValue({
        lines: [],
        total: 0,
        progress: 0,
        progress_msg: '',
        phase: '',
      });

      render(
        <JobDetailsSidebar
          job={makeJob({ status: 'running' })}
          onClose={vi.fn()}
        />,
      );

      // Flush the immediate poll fired on mount.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      const initialCalls = vi.mocked(getJobLogs).mock.calls.length;

      // Two 2s interval ticks should fire while the job is running.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4000);
      });

      expect(vi.mocked(getJobLogs).mock.calls.length).toBeGreaterThan(
        initialCalls,
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('stops polling once the job is completed', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(getJobLogs).mockResolvedValue({
        lines: ['final line'],
        total: 1,
        progress: 0,
        progress_msg: '',
        phase: '',
      });

      render(
        <JobDetailsSidebar
          job={makeJob({ status: 'completed' })}
          onClose={vi.fn()}
        />,
      );

      // The component fetches once on mount even for terminal jobs.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(getJobLogs).toHaveBeenCalledTimes(1);

      // No interval is registered, so advancing the clock must not re-poll.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10000);
      });
      expect(getJobLogs).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
