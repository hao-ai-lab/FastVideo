import * as React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import { createJob, getDatasets, getModels, getInferenceOptions, uploadImage } from '@/lib/api';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getInferenceOptions: vi.fn(),
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
  vi.mocked(getInferenceOptions).mockResolvedValue({ attention_backends: ["TORCH_SDPA", "VIDEO_SPARSE_ATTN_H3"], defaults: {}, source: null });
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
  it('shows a model loading error instead of an empty model list', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(getModels).mockRejectedValueOnce(new Error('network down'));

    renderModal();

    expect(
      await screen.findByText(/Models could not be loaded/),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toHaveAttribute(
      'aria-invalid',
      'true',
    );
  });

  it('keeps the form open and reports job creation failures', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(createJob).mockRejectedValueOnce(new Error('API rejected job'));
    const user = userEvent.setup();
    const { onClose, onSuccess } = renderModal();

    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    await user.type(screen.getByLabelText('Prompt'), 'a careful test prompt');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    expect(
      await screen.findByText(/API rejected job.*then try again/),
    ).toBeInTheDocument();
    expect(onSuccess).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('reports image upload failures next to the file input', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(uploadImage).mockRejectedValueOnce(new Error('Upload failed'));
    const user = userEvent.setup();
    renderModal({ workloadType: 'i2v' });

    await screen.findByRole('option', { name: 'Wan T2V (wan/t2v-1.3b)' });
    const input = screen.getByLabelText('Image');
    await user.upload(
      input,
      new File(['image'], 'input.png', { type: 'image/png' }),
    );

    expect(
      await screen.findByText(/Upload failed.*Choose the image again/),
    ).toBeInTheDocument();
    expect(input).toHaveAttribute('aria-invalid', 'true');
  });

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
      // The dataset id is sent; the backend resolves it to the on-disk dir.
      data_path: 'ds1',
      lora_rank: 32,
      // DMD-specific fields added to CreateJobRequest for this modal.
      dmd_use_vsa: false,
      dmd_vsa_sparsity: 0.8,
      dmd_denoising_steps: '1000,757,522',
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


describe('attention configuration', () => {
  it('submits an explicit backend and tile size', async () => {
    const user = userEvent.setup();
    renderModal();
    await screen.findByRole('option', { name: 'TORCH_SDPA' });
    await user.selectOptions(screen.getByLabelText('Attention Backend'), 'TORCH_SDPA');
    await user.selectOptions(screen.getByLabelText('VSA Tile Size'), '64');
    await user.type(screen.getByLabelText('Prompt'), 'cat');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    expect(createJob).toHaveBeenCalledWith(expect.objectContaining({ attention_backend: 'TORCH_SDPA', vsa_tile_size: 64 }));
  });

  it('applies recipe defaults only when requested', async () => {
    vi.mocked(getInferenceOptions).mockResolvedValue({ attention_backends: ['TORCH_SDPA', 'VIDEO_SPARSE_ATTN_H3'],
      defaults: { attention_backend: 'VIDEO_SPARSE_ATTN_H3', vsa_tile_size: 64, vsa_sparsity: 0.8,
        num_inference_steps: 9, guidance_scale: 1, num_frames: 124, height: 768, width: 1344, fps: 24 },
      source: 'examples/serving/openai_fasth3_8step.yaml' });
    const user = userEvent.setup();
    renderModal();
    const apply = await screen.findByRole('button', { name: 'Apply recipe defaults' });
    expect(screen.getByLabelText('Attention Backend')).toHaveValue('');
    await user.click(apply);
    expect(screen.getByLabelText('Attention Backend')).toHaveValue('VIDEO_SPARSE_ATTN_H3');
    await user.type(screen.getByLabelText('Prompt'), 'cat');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));
    expect(createJob).toHaveBeenCalledWith(expect.objectContaining({ attention_backend: 'VIDEO_SPARSE_ATTN_H3',
      vsa_tile_size: 64, num_inference_steps: 9, guidance_scale: 1 }));
  });
});

it('previews model-resolved Auto and submits unset selections', async () => {
  vi.mocked(getInferenceOptions).mockResolvedValue({
    attention_backends: ['VIDEO_SPARSE_ATTN_H3'], defaults: {}, source: null,
    auto_attention_backend: 'VIDEO_SPARSE_ATTN_H3', auto_vsa_tile_size: 64,
  });
  const user = userEvent.setup();
  renderModal();
  await screen.findByRole('option', { name: 'Auto (VIDEO_SPARSE_ATTN_H3)' });
  expect(screen.getByRole('option', { name: 'Auto (64)' })).toBeInTheDocument();
  expect(screen.getByLabelText('Attention Backend')).toHaveValue('');
  await user.type(screen.getByLabelText('Prompt'), 'cat');
  await user.click(screen.getByRole('button', { name: 'Create Job' }));
  expect(createJob).toHaveBeenCalledWith(expect.objectContaining({ attention_backend: null, vsa_tile_size: null }));
});

it('updates Auto tile preview when the explicit backend overrides the server', async () => {
  vi.mocked(getInferenceOptions).mockResolvedValue({
    attention_backends: ['TORCH_SDPA', 'FLASH_ATTN', 'VIDEO_SPARSE_ATTN_H3'],
    defaults: { attention_backend: 'VIDEO_SPARSE_ATTN_H3', vsa_tile_size: 64 }, source: null,
    auto_attention_backend: 'TORCH_SDPA', auto_vsa_tile_size: null,
  });
  const user = userEvent.setup();
  renderModal();
  await screen.findByRole('option', { name: 'VIDEO_SPARSE_ATTN_H3' });
  expect(screen.getByRole('option', { name: 'Auto (checkpoint default)' })).toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText('Attention Backend'), 'VIDEO_SPARSE_ATTN_H3');
  expect(screen.getByRole('option', { name: 'Auto (64)' })).toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText('Attention Backend'), 'FLASH_ATTN');
  expect(screen.getByRole('option', { name: 'Auto (checkpoint default)' })).toBeInTheDocument();
});
