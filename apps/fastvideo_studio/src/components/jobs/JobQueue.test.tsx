import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import JobQueue from '@/components/jobs/JobQueue';
import { getJobsList, getModels } from '@/lib/api';
import type { Job, JobType } from '@/lib/types';
import { activeJobStore, setActiveJobId } from '@/stores/activeJob';
import { triggerRefresh } from '@/stores/jobsRefresh';
import { makeJob as makeBaseJob } from '@/test/factories';

vi.mock('@/lib/api', () => ({
  getApiBaseUrl: () => 'http://test.local/api',
  getJobVideoUrl: (id: string) => `http://test.local/api/jobs/${id}/video`,
  getJobsList: vi.fn(),
  getModels: vi.fn(),
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
  vi.mocked(getModels).mockResolvedValue([]);
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
    expect(getModels).toHaveBeenCalledWith(undefined);
  });

  it('renders a JobCard for each fetched job', async () => {
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'a', model_id: 'Model-A' }),
    ]);
    render(<JobQueue jobType="distillation" />);
    expect(await screen.findByRole('button', { name: 'View configuration: Model-A' })).toBeInTheDocument();
    expect(getModels).toHaveBeenCalledWith('t2v');
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
    expect(await screen.findByRole('button', { name: 'View configuration: Lora-Model' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View configuration: Full-Model' })).toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledWith('finetuning');
    expect(getJobsList).toHaveBeenCalledWith('lora');
    expect(getModels).toHaveBeenCalledWith('t2v');
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
    fireEvent.change(screen.getByRole('combobox', { name: 'Model name' }), { target: { value: 'Wan2.1' } });
    fireEvent.change(screen.getByLabelText('Prompt contains'), { target: { value: 'CAT' } });
    expect(screen.getByText('2 of 4 jobs')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Status' }), { target: { value: 'completed' } });
    expect(screen.getByText('1 of 4 jobs')).toBeInTheDocument();
    expect(screen.getByText('A surfing cat')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View configuration: Other' })).not.toBeInTheDocument();
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

    fireEvent.change(screen.getByRole('combobox', { name: 'Model name' }), { target: { value: 'Wan2.1' } });
    fireEvent.change(screen.getByLabelText('Prompt contains'), { target: { value: 'cat' } });
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

    expect(screen.getByRole('combobox', { name: 'Status' })).toHaveValue('');
    expect(screen.getByLabelText('Model name')).toHaveValue('');
    expect(screen.getByLabelText('Prompt contains')).toHaveValue('');
    expect(screen.getByText('2 jobs')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View configuration: Wan2.1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View configuration: Other' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Clear' })).not.toBeInTheDocument();
    expect(getJobsList).toHaveBeenCalledTimes(1);
  });

  it('propagates a polled failure and later error updates to the selected job', async () => {
    vi.useFakeTimers();
    try {
      const runningJob = makeJob({ status: 'running' });
      const failedJob = {
        ...runningJob,
        status: 'failed',
        error: 'Dataset path is not an existing directory',
      };
      vi.mocked(getJobsList)
        .mockResolvedValueOnce([runningJob])
        .mockResolvedValue([failedJob]);

      render(<JobQueue jobType="finetuning" />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      act(() => setActiveJobId(runningJob.id));
      expect(activeJobStore.get().activeJob?.status).toBe('running');

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(activeJobStore.get().activeJob).toEqual(failedJob);

      const updatedJob = { ...failedJob, error: 'Preprocess your dataset first.' };
      vi.mocked(getJobsList).mockResolvedValue([updatedJob]);
      await act(async () => triggerRefresh());
      expect(activeJobStore.get().activeJob).toEqual(updatedJob);
    } finally {
      vi.useRealTimers();
    }
  });

  it('merges the create-job catalog with recorded models and filters by exact ID', async () => {
    vi.mocked(getModels).mockResolvedValue([
      { id: 'wan', label: 'Wan 2.1' },
      { id: 'catalog-only', label: 'Latest model' },
    ]);
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'one', model_id: 'wan' }),
      makeJob({ id: 'two', model_id: 'wan-extra' }),
      makeJob({ id: 'three', model_id: 'old/custom' }),
    ]);
    render(<JobQueue jobType="inference" />);
    const select = await screen.findByRole('combobox', { name: 'Model name' });
    expect(select).toHaveValue('');
    expect(within(select).getAllByRole('option')).toHaveLength(5);
    expect(within(select).getByRole('option', { name: 'Wan 2.1 (wan)' })).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'Latest model (catalog-only)' })).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'old/custom' })).toBeInTheDocument();

    fireEvent.change(select, { target: { value: 'wan' } });
    expect(screen.getByText('1 of 3 jobs')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View configuration: wan' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View configuration: wan-extra' })).not.toBeInTheDocument();

    fireEvent.change(select, { target: { value: 'catalog-only' } });
    expect(screen.getByText('No jobs match these filters.')).toBeInTheDocument();
    expect(getModels).toHaveBeenCalledTimes(1);
  });

  it('keeps recorded models filterable when the catalog request fails', async () => {
    vi.mocked(getModels).mockRejectedValue(new Error('catalog unavailable'));
    vi.mocked(getJobsList).mockResolvedValue([
      makeJob({ id: 'one', model_id: 'old/custom' }),
      makeJob({ id: 'two', model_id: 'other' }),
    ]);
    render(<JobQueue jobType="inference" />);
    const select = await screen.findByRole('combobox', { name: 'Model name' });
    fireEvent.change(select, { target: { value: 'old/custom' } });
    expect(screen.getByText('1 of 2 jobs')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View configuration: old/custom' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
