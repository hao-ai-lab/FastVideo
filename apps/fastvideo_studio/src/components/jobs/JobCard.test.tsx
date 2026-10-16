import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import JobCard from '@/components/jobs/JobCard';
import {
  deleteJob,
  downloadJobVideo,
  startJob,
  stopJob,
} from '@/lib/api';
import type { Job } from '@/lib/types';
import { activeJobStore, setActiveJobId } from '@/stores/activeJob';
import { OMNI_RECIPE, makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  startJob: vi.fn(),
  stopJob: vi.fn(),
  deleteJob: vi.fn(),
  downloadJobVideo: vi.fn(),
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
  it('keeps selection and job action buttons as semantic siblings', () => {
    render(<JobCard job={makeJob()} />);

    const selectButton = screen.getByRole('button', { pressed: false });
    const deleteButton = screen.getByRole('button', { name: 'Delete' });

    expect(selectButton).toHaveTextContent('Wan2.1-T2V');
    expect(selectButton).not.toContainElement(deleteButton);
  });

  it('renders the model, prompt, status and inference meta', () => {
    render(<JobCard job={makeJob()} />);
    expect(screen.getByText('Wan2.1-T2V')).toBeInTheDocument();
    expect(screen.getByText('a cat surfing a wave')).toBeInTheDocument();
    expect(screen.getByText('pending')).toBeInTheDocument();
    expect(screen.getByText('81 frames')).toBeInTheDocument();
    expect(screen.getByText('480×832')).toBeInTheDocument();
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

  it('downloads the video for a completed inference job', async () => {
    render(
      <JobCard
        job={makeJob({ status: 'completed', output_path: '/out/video.mp4' })}
      />,
    );
    await userEvent.click(
      screen.getByRole('button', { name: 'Download Video' }),
    );
    await waitFor(() => expect(downloadJobVideo).toHaveBeenCalledWith('job-1'));
  });

  it('selects the job when the card body is clicked', async () => {
    render(<JobCard job={makeJob()} />);
    await userEvent.click(screen.getByText('Wan2.1-T2V'));
    expect(activeJobStore.get().activeJobId).toBe('job-1');
  });

  it('shows a failed job\'s error with its category', () => {
    render(
      <JobCard
        job={makeJob({
          status: 'failed',
          error: 'This recipe needs sm_100a; found RTX 6000 Ada (sm_89).',
          error_category: 'kernel',
        })}
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Kernel / hardware error: This recipe needs sm_100a',
    );
  });

  it('summarises a FastH3-Omni job\'s recipe and references', () => {
    render(
      <JobCard
        job={makeJob({
          recipe: OMNI_RECIPE,
          references: [
            { source: '/u/a.png', media_type: 'image' },
            { source: '/u/b.mp4', media_type: 'video' },
          ],
        })}
      />,
    );
    expect(screen.getByText('8 forwards · VSA 0.9 · tile 128')).toBeInTheDocument();
    expect(screen.getByText('2 references')).toBeInTheDocument();
  });
});
