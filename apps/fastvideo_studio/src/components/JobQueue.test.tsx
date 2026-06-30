import { act, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import JobQueue from '@/components/JobQueue';
import { getJobsList } from '@/lib/api';
import type { Job, JobType } from '@/lib/types';
import { setActiveJobId } from '@/stores/activeJob';
import { triggerRefresh } from '@/stores/jobsRefresh';

vi.mock('@/lib/api', () => ({
  getJobsList: vi.fn(),
  startJob: vi.fn(),
  stopJob: vi.fn(),
  deleteJob: vi.fn(),
  downloadJobVideo: vi.fn(),
}));

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 'job-1',
    model_id: 'Wan2.1-T2V',
    prompt: 'a prompt',
    job_type: 'inference',
    workload_type: 't2v',
    status: 'completed',
    created_at: 1_700_000_000,
    started_at: null,
    finished_at: null,
    error: null,
    output_path: null,
    log_file_path: null,
    num_inference_steps: 50,
    num_frames: 81,
    height: 480,
    width: 832,
    guidance_scale: 5,
    seed: 42,
    num_gpus: 1,
    progress: 0,
    progress_msg: '',
    phase: '',
    ...overrides,
  };
}

beforeEach(() => {
  setActiveJobId(null);
  vi.mocked(getJobsList).mockResolvedValue([]);
});

describe('JobQueue', () => {
  it('shows an empty placeholder and fetches for the single job type', async () => {
    render(<JobQueue jobType="inference" />);
    expect(
      await screen.findByText('No inference jobs yet. Create one above.'),
    ).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('inference');
  });

  it('renders a JobCard for each fetched job', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'a', model_id: 'Model-A' }),
    ]);
    render(<JobQueue jobType="distillation" />);
    expect(await screen.findByText('Model-A')).toBeInTheDocument();
  });

  it('merges all job types when jobTypesForList is provided', async () => {
    vi.mocked(getJobsList).mockImplementation((t?: JobType) =>
      Promise.resolve(
        t === ('lora' as JobType)
          ? [makeJob({ id: 'l', model_id: 'Lora-Model', created_at: 2 })]
          : [makeJob({ id: 'f', model_id: 'Full-Model', created_at: 1 })],
      ),
    );
    render(
      <JobQueue
        jobType="finetuning"
        jobTypesForList={['finetuning', 'lora'] as JobType[]}
      />,
    );
    expect(await screen.findByText('Lora-Model')).toBeInTheDocument();
    expect(screen.getByText('Full-Model')).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('finetuning');
    expect(getJobsList).toHaveBeenCalledWith('lora');
  });

  it('refetches when a refresh is triggered', async () => {
    render(<JobQueue jobType="inference" />);
    await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(1));
    act(() => triggerRefresh());
    await waitFor(() => expect(getJobsList).toHaveBeenCalledTimes(2));
  });
});
