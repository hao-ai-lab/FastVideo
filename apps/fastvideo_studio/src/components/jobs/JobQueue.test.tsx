import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import JobQueue from '@/components/jobs/JobQueue';
import { getJobsList } from '@/lib/api';
import type { Job, JobType } from '@/lib/types';
import { setActiveJobId } from '@/stores/activeJob';
import { triggerRefresh } from '@/stores/jobsRefresh';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getApiBaseUrl: () => 'http://test.local/api',
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  getJobsList: vi.fn(),
  startJob: vi.fn(),
  stopJob: vi.fn(),
  deleteJob: vi.fn(),
  downloadJobVideo: vi.fn(),
}));

const makeJob = (overrides: Partial<Job> = {}): Job =>
  makeBaseJob({
    model_id: 'Wan2.1-T2V',
    status: 'completed',
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
  vi.mocked(getJobsList).mockResolvedValue([]);
});

describe('JobQueue', () => {
  it('shows a loading placeholder before the initial request settles', async () => {
    let resolveJobs: (jobs: Job[]) => void = () => {};
    vi.mocked(getJobsList).mockReturnValue(
      new Promise<Job[]>((resolve) => {
        resolveJobs = resolve;
      }),
    );

    render(<JobQueue jobType="inference" />);

    expect(screen.getByLabelText('Loading jobs')).toBeInTheDocument();
    expect(
      screen.queryByText('No inference jobs yet. Create one above.'),
    ).not.toBeInTheDocument();

    act(() => resolveJobs([]));
    expect(
      await screen.findByText('No inference jobs yet. Create one above.'),
    ).toBeInTheDocument();
  });

  it('shows request failures separately from an empty queue and retries', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(getJobsList).mockRejectedValueOnce(new Error('network down'));
    render(<JobQueue jobType="inference" />);

    expect(
      await screen.findByText(/Could not load jobs from the Studio API/),
    ).toBeInTheDocument();
    expect(
      screen.queryByText('No inference jobs yet. Create one above.'),
    ).not.toBeInTheDocument();

    vi.mocked(getJobsList).mockResolvedValueOnce([]);
    fireEvent.click(screen.getByRole('button', { name: 'Try Again' }));
    expect(
      await screen.findByText('No inference jobs yet. Create one above.'),
    ).toBeInTheDocument();
  });

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

  it('caps thumbnails at 50 while keeping every completed result available', async () => {
    vi.mocked(getJobsList).mockResolvedValue(
      Array.from({ length: 51 }, (_, index) => makeJob({ id: `job-${index}`, name: `Result ${index}`, output_path: '/out/clip.mp4' })),
    );
    const { container } = render(<JobQueue jobType="inference" />);
    expect(await screen.findByText('51 jobs')).toBeInTheDocument();
    expect(container.querySelectorAll('img')).toHaveLength(50);
    expect(screen.getAllByRole('button', { name: /Preview result:/ })).toHaveLength(51);
    expect(container.querySelectorAll('video')).toHaveLength(0);
  });

  it('combines status, model, and prompt filters without issuing a new API request', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'one', model_id: 'Wan2.1', prompt: 'A surfing cat' }),
      makeJob({ id: 'two', model_id: 'Other', prompt: 'A surfing cat' }),
      makeJob({ id: 'three', model_id: 'Wan2.1', prompt: 'A dog swimming' }),
      makeJob({ id: 'four', model_id: 'Wan2.1', prompt: 'A surfing cat', status: 'failed' }),
    ]);
    render(<JobQueue jobType="inference" />);
    await screen.findByText('4 jobs');
    expect(screen.getByRole('combobox', { name: 'Status' })).toHaveValue('');
    fireEvent.change(screen.getByLabelText('Model name'), { target: { value: 'wan' } });
    fireEvent.change(screen.getByLabelText('Prompt contains'), { target: { value: 'CAT' } });
    expect(screen.getByText('2 of 4 jobs')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Status' }), { target: { value: 'completed' } });
    expect(screen.getByText('1 of 4 jobs')).toBeInTheDocument();
    expect(screen.getByText('A surfing cat')).toBeInTheDocument();
    expect(screen.queryByText('Other')).not.toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledTimes(1);
  });

  it('shows no matches and clears every filter back to the full queue', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'one', model_id: 'Wan2.1', prompt: 'A surfing cat' }),
      makeJob({ id: 'two', model_id: 'Other', prompt: 'A dog swimming', status: 'failed' }),
    ]);
    render(<JobQueue jobType="inference" />);
    await screen.findByText('2 jobs');
    fireEvent.change(screen.getByRole('combobox', { name: 'Status' }), { target: { value: 'stopped' } });
    expect(screen.getByText('0 of 2 jobs')).toBeInTheDocument();
    expect(screen.getByText('No jobs match these filters.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Clear' })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Model name'), { target: { value: 'wan' } });
    fireEvent.change(screen.getByLabelText('Prompt contains'), { target: { value: 'cat' } });
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

    expect(screen.getByRole('combobox', { name: 'Status' })).toHaveValue('');
    expect(screen.getByLabelText('Model name')).toHaveValue('');
    expect(screen.getByLabelText('Prompt contains')).toHaveValue('');
    expect(screen.getByText('2 jobs')).toBeInTheDocument();
    expect(screen.getByText('Wan2.1')).toBeInTheDocument();
    expect(screen.getByText('Other')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Clear' })).not.toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledTimes(1);
  });
});
