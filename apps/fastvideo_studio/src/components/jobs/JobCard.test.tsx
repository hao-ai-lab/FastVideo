import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import JobCard from '@/components/jobs/JobCard';
import type { CreateJobModalProps } from '@/components/jobs/CreateJobModal';
import {
  deleteJob,
  downloadJobVideo,
  startJob,
  stopJob,
} from '@/lib/api';
import type { Job } from '@/lib/types';
import { activeJobStore, setActiveJobId } from '@/stores/activeJob';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getApiBaseUrl: () => 'http://test.local/api',
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  startJob: vi.fn(),
  stopJob: vi.fn(),
  deleteJob: vi.fn(),
  downloadJobVideo: vi.fn(),
}));

vi.mock('@/components/jobs/CreateJobModal', () => ({
  default: ({ readOnly, onClose }: CreateJobModalProps) => (
    <div role="dialog" aria-label={readOnly ? 'View configuration' : 'Edit configuration'}>
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

vi.mock('@/lib/utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/utils')>();
  return {
    ...actual,
    downloadBlob: vi.fn(),
  };
});

const makeJob = (overrides: Partial<Job> = {}): Job =>
  makeBaseJob({
    model_id: 'Wan2.1-T2V',
    prompt: 'a cat surfing a wave',
    created_at: 1_700_000_000,
    num_inference_steps: 50,
    num_frames: 81,
    height: 480,
    width: 832,
    guidance_scale: 5,
    seed: 42,
    num_gpus: 1,
    ...overrides,
  });

beforeEach(() => {
  setActiveJobId(null);
  vi.mocked(startJob).mockResolvedValue({} as Job);
  vi.mocked(stopJob).mockResolvedValue({} as Job);
  vi.mocked(deleteJob).mockResolvedValue(undefined);
  vi.mocked(downloadJobVideo).mockResolvedValue(new Blob());
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  vi.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('JobCard', () => {
  it('keeps configuration and job action buttons as semantic siblings', () => {
    render(<JobCard job={makeJob()} />);

    const selectButton = screen.getByRole('button', { name: 'View configuration: Wan2.1-T2V' });
    const deleteButton = screen.getByRole('button', { name: 'Delete' });

    expect(selectButton).toHaveTextContent('Wan2.1-T2V');
    expect(selectButton).not.toContainElement(deleteButton);
    expect(selectButton).not.toHaveAttribute('aria-pressed');
  });

  it('renders the model, prompt, status and inference meta', () => {
    render(<JobCard job={makeJob()} />);
    expect(screen.getByText('Wan2.1-T2V')).toBeInTheDocument();
    expect(screen.getByText('a cat surfing a wave')).toBeInTheDocument();
    expect(screen.getByText('pending')).toBeInTheDocument();
    expect(screen.getByText('81 frames')).toBeInTheDocument();
    expect(screen.getByText('832×480')).toBeInTheDocument();
  });

  it('shows the workload type (not frames) for non-inference jobs', () => {
    render(
      <JobCard
        job={makeJob({ job_type: 'finetuning', workload_type: 'lora_t2v' })}
      />,
    );
    expect(screen.getByText('lora t2v')).toBeInTheDocument();
    expect(screen.queryByText('81 frames')).not.toBeInTheDocument();
  });

  it('starts a pending job and notifies the parent', async () => {
    const onJobUpdated = vi.fn();
    render(<JobCard job={makeJob({ status: 'pending' })} onJobUpdated={onJobUpdated} />);
    await userEvent.click(screen.getByRole('button', { name: 'Start' }));
    await waitFor(() => expect(startJob).toHaveBeenCalledWith('job-1'));
    expect(onJobUpdated).toHaveBeenCalled();
  });

  it('stops a running job', async () => {
    render(<JobCard job={makeJob({ status: 'running', started_at: Date.now() })} />);
    await userEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await waitFor(() => expect(stopJob).toHaveBeenCalledWith('job-1'));
  });

  it('restarts a failed job via startJob', async () => {
    render(<JobCard job={makeJob({ status: 'failed' })} />);
    await userEvent.click(screen.getByRole('button', { name: 'Restart' }));
    await waitFor(() => expect(startJob).toHaveBeenCalledWith('job-1'));
  });

  it('deletes when confirmed and skips when cancelled', async () => {
    const { rerender } = render(<JobCard job={makeJob()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(deleteJob).toHaveBeenCalledWith('job-1'));

    vi.mocked(deleteJob).mockClear();
    vi.mocked(window.confirm).mockReturnValue(false);
    rerender(<JobCard job={makeJob()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(deleteJob).not.toHaveBeenCalled();
  });

  it('downloads from the media icon without opening configuration or logs', async () => {
    render(
      <JobCard
        job={makeJob({ status: 'completed', output_path: '/out/video.mp4' })}
      />,
    );
    await userEvent.click(
      screen.getByRole('button', { name: 'Download video' }),
    );
    await waitFor(() => expect(downloadJobVideo).toHaveBeenCalledWith('job-1'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(activeJobStore.get().activeJobId).toBeNull();
  });

  it.each(['pending', 'failed', 'running', 'completed'])('views a %s job without opening its logs', async (status) => {
    render(<JobCard job={makeJob({ status })} />);
    await userEvent.click(screen.getByText('Wan2.1-T2V'));
    expect(screen.getByRole('dialog', { name: 'View configuration' })).toBeInTheDocument();
    expect(activeJobStore.get().activeJobId).toBeNull();
  });

  it('opens configuration with the keyboard or a click on card whitespace', async () => {
    const { container } = render(<JobCard job={makeJob()} />);
    screen.getByRole('button', { name: 'View configuration: Wan2.1-T2V' }).focus();
    await userEvent.keyboard('{Enter}');
    expect(screen.getByRole('dialog', { name: 'View configuration' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    fireEvent.click(container.querySelector('article')!);
    expect(screen.getByRole('dialog', { name: 'View configuration' })).toBeInTheDocument();
  });

  it('toggles the sidebar only with Details & logs', async () => {
    render(<JobCard job={makeJob()} />);
    const details = screen.getByRole('button', { name: 'Details & logs' });
    await userEvent.click(details);
    expect(activeJobStore.get().activeJobId).toBe('job-1');
    expect(details).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await userEvent.click(details);
    expect(activeJobStore.get().activeJobId).toBeNull();
  });

  it('keeps editing and previewing separate from configuration viewing', async () => {
    const { rerender } = render(<JobCard job={makeJob()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Edit' }));
    expect(screen.getByRole('dialog', { name: 'Edit configuration' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'View configuration' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    rerender(<JobCard job={makeJob({ status: 'completed', output_path: '/out/video.mp4' })} />);
    const video = screen.getByLabelText('Generated video: a cat surfing a wave');
    expect(video).toHaveAttribute('controls');
    await userEvent.click(video);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(activeJobStore.get().activeJobId).toBeNull();
  });
});
