import * as React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { createJob, getDatasets, getModels, uploadImage } from '@/lib/api';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getDatasets: vi.fn(),
  uploadImage: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const MODELS = [
  { id: 'wan/t2v-1.3b', label: 'Wan T2V', type: 't2v' },
  { id: 'wan/t2v-14b', label: 'Wan T2V Large', type: 't2v' },
];

beforeEach(() => {
  // Reset the shared options store to a known baseline for test isolation.
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue(MODELS);
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(uploadImage).mockResolvedValue({ path: '/uploads/x.png' });
  vi.mocked(createJob).mockResolvedValue({ id: 'job-1' } as never);
});

function renderModal(
  overrides: Partial<React.ComponentProps<typeof CreateJobModal>> = {},
) {
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  render(
    <CreateJobModal
      isOpen
      onClose={onClose}
      onSuccess={onSuccess}
      jobType="inference"
      workloadType="t2v"
      {...overrides}
    />,
  );
  return { onClose, onSuccess };
}

describe('CreateJobModal', () => {
  it('renders the form fields for an inference job', async () => {
    renderModal();

    expect(
      await screen.findByText('New Inference Job (T2V)'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toBeInTheDocument();
    expect(screen.getByLabelText('Prompt')).toBeInTheDocument();
    expect(screen.getByLabelText('Negative Prompt')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Create Job' }),
    ).toBeInTheDocument();

    // The model dropdown is populated once getModels resolves.
    expect(
      await screen.findByRole('option', {
        name: 'Wan T2V (wan/t2v-1.3b)',
      }),
    ).toBeInTheDocument();
  });

  it('seeds fields from the options store and submits an inference payload', async () => {
    // Non-default store values prove the open-time seeding effect ran (the
    // useState defaults are 50 / 480).
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numInferenceSteps: 25, height: 720 },
    });

    const user = userEvent.setup();
    const { onClose, onSuccess } = renderModal();

    // Wait for models to load so the default model is selected.
    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });

    await user.type(
      screen.getByLabelText('Prompt'),
      'a raccoon in sunflowers',
    );
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload).toMatchObject({
      model_id: 'wan/t2v-1.3b',
      prompt: 'a raccoon in sunflowers',
      workload_type: 't2v',
      job_type: 'inference',
      num_inference_steps: 25,
      height: 720,
      num_frames: 81,
      width: 832,
      guidance_scale: 5,
      seed: 1024,
    });

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('submits a dmd_t2v distillation payload including the DMD fields', async () => {
    vi.mocked(getDatasets).mockResolvedValue([
      { id: 'ds1', name: 'My Dataset', created_at: 0 },
    ]);

    const user = userEvent.setup();
    const { onSuccess } = renderModal({
      jobType: 'distillation',
      workloadType: 'dmd_t2v',
    });

    // Models + datasets load asynchronously on open. The model/dataset option
    // labels also appear in the Real/Fake Score Model and Validation Dataset
    // selects, so scope each wait to the relevant select.
    await within(screen.getByLabelText('Model')).findByRole('option', {
      name: 'Wan T2V (wan/t2v-1.3b)',
    });
    const datasetSelect = screen.getByLabelText('Dataset *');
    await within(datasetSelect).findByRole('option', { name: 'My Dataset' });

    await user.type(screen.getByLabelText('Description'), 'distill run');
    await user.selectOptions(datasetSelect, 'ds1');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload).toMatchObject({
      workload_type: 'dmd_t2v',
      job_type: 'distillation',
      data_path: 'My Dataset',
      lora_rank: 32,
      // DMD-specific fields added to CreateJobRequest for this modal.
      dmd_use_vsa: false,
      dmd_vsa_sparsity: 0.8,
      dmd_denoising_steps: '1000,757,522',
      min_timestep_ratio: 0.02,
      max_timestep_ratio: 0.98,
      real_score_guidance_scale: 3.5,
      generator_update_interval: 5,
      real_score_model_path: 'wan/t2v-1.3b',
      fake_score_model_path: 'wan/t2v-1.3b',
    });
    // Inference-only keys must be absent for a training job.
    expect(payload).not.toHaveProperty('num_inference_steps');

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });
});
