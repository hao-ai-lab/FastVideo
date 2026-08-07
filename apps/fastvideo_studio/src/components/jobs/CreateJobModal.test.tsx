import * as React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import CreateJobModal from './CreateJobModal';
import {
  createJob,
  getDatasets,
  getModelPresets,
  getModels,
  listGenerators,
  uploadImage,
  type GeneratorInfo,
} from '@/lib/api';
import { defaultOptionsStore } from '@/stores/defaultOptions';
import { DEFAULT_OPTIONS } from '@/lib/defaultOptions';

vi.mock('@/lib/api', () => ({
  createJob: vi.fn(),
  getModels: vi.fn(),
  getModelPresets: vi.fn(),
  getDatasets: vi.fn(),
  listGenerators: vi.fn(),
  uploadImage: vi.fn(),
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
}));

const MODELS = [
  { id: 'wan/t2v-1.3b', label: 'Wan T2V', type: 't2v' },
  { id: 'wan/t2v-14b', label: 'Wan T2V Large', type: 't2v' },
];

// A resident engine slot whose engine config differs from the persisted
// defaults on every field the modal adopts.
const WARM_SLOT: GeneratorInfo = {
  state: 'ready',
  model_id: 'wan/t2v-14b',
  workload_type: 't2v',
  num_gpus: 8,
  dit_cpu_offload: true,
  text_encoder_cpu_offload: true,
  vae_cpu_offload: true,
  image_encoder_cpu_offload: false,
  use_fsdp_inference: true,
  enable_torch_compile: true,
  vsa_sparsity: 0.5,
  tp_size: 1,
  sp_size: 8,
  error: null,
};

beforeEach(() => {
  // Reset the shared options store to a known baseline for test isolation.
  defaultOptionsStore.set({ options: DEFAULT_OPTIONS });
  vi.mocked(getModels).mockResolvedValue(MODELS);
  vi.mocked(getModelPresets).mockResolvedValue({});
  vi.mocked(getDatasets).mockResolvedValue([]);
  vi.mocked(listGenerators).mockResolvedValue([]);
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

  it('defaults to the warm resident model and adopts its engine config', async () => {
    vi.mocked(listGenerators).mockResolvedValue([WARM_SLOT]);

    const user = userEvent.setup();
    renderModal();

    // Without a warm model the default logic picks the first model
    // (wan/t2v-1.3b, covered by the seeding test above); the warm slot wins.
    await waitFor(() =>
      expect(screen.getByLabelText('Model')).toHaveValue('wan/t2v-14b'),
    );
    // The warm selection also triggers its presets fetch.
    await waitFor(() =>
      expect(getModelPresets).toHaveBeenCalledWith('wan/t2v-14b'),
    );

    await user.type(screen.getByLabelText('Prompt'), 'warm run');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    // The engine fields mirror the resident slot (not the persisted defaults:
    // num_gpus 1, all offloads/fsdp/compile false) so the job reuses the warm
    // instance instead of replacing it.
    expect(vi.mocked(createJob).mock.calls[0][0]).toMatchObject({
      model_id: 'wan/t2v-14b',
      num_gpus: 8,
      tp_size: 1,
      sp_size: 8,
      dit_cpu_offload: true,
      text_encoder_cpu_offload: true,
      vae_cpu_offload: true,
      image_encoder_cpu_offload: false,
      use_fsdp_inference: true,
      enable_torch_compile: true,
      vsa_sparsity: 0.5,
    });
  });

  it('restores persisted-default engine fields when switching away from the warm model', async () => {
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numGpus: 2, tpSize: 2 },
    });
    vi.mocked(listGenerators).mockResolvedValue([WARM_SLOT]);

    const user = userEvent.setup();
    renderModal();

    await waitFor(() =>
      expect(screen.getByLabelText('Model')).toHaveValue('wan/t2v-14b'),
    );
    await user.selectOptions(screen.getByLabelText('Model'), 'wan/t2v-1.3b');
    await user.type(screen.getByLabelText('Prompt'), 'cold run');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(createJob).mock.calls[0][0]).toMatchObject({
      model_id: 'wan/t2v-1.3b',
      num_gpus: 2,
      tp_size: 2,
      use_fsdp_inference: false,
      enable_torch_compile: false,
    });
  });

  it('populates sampling fields from the selected model presets; engine fields stay from defaults', async () => {
    defaultOptionsStore.set({
      options: { ...DEFAULT_OPTIONS, numGpus: 4, tpSize: 2, seed: 999 },
    });
    vi.mocked(getModelPresets).mockImplementation(async (id) =>
      id === 'wan/t2v-14b'
        ? {
            height: 720,
            width: 1280,
            num_frames: 121,
            fps: 30,
            num_inference_steps: 40,
            guidance_scale: 6,
            guidance_rescale: 0.5,
            negative_prompt: 'blurry, low quality',
            seed: 7,
          }
        : {},
    );

    const user = userEvent.setup();
    renderModal();

    await screen.findByRole('option', { name: 'Wan T2V Large (wan/t2v-14b)' });
    await user.selectOptions(screen.getByLabelText('Model'), 'wan/t2v-14b');
    // The negative prompt is the easiest preset-populated field to observe.
    await waitFor(() =>
      expect(screen.getByLabelText('Negative Prompt')).toHaveValue(
        'blurry, low quality',
      ),
    );

    await user.type(screen.getByLabelText('Prompt'), 'preset test');
    await user.click(screen.getByRole('button', { name: 'Create Job' }));

    await waitFor(() => expect(createJob).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload).toMatchObject({
      model_id: 'wan/t2v-14b',
      // Sampling fields come from the model presets…
      height: 720,
      width: 1280,
      num_frames: 121,
      fps: 30,
      num_inference_steps: 40,
      guidance_scale: 6,
      guidance_rescale: 0.5,
      negative_prompt: 'blurry, low quality',
      seed: 7,
      // …while engine fields still come from the persisted defaults.
      num_gpus: 4,
      tp_size: 2,
    });
  });
});
